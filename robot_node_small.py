#!/usr/bin/env python3

import os, sys, time, logging
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch
from threading import Lock, Thread, Event
from collections import deque
import hydra, dill
import scipy.spatial.transform as st

# Custom imports assumed to be available
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import (
    normalize, mat_to_rot6d, rot6d_to_mat,
    pose_to_mat, mat_to_pose
)
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        self.configure_logging()

        # Parameter declarations
        params = {
            'image_topic': '/image_compressed',
            'checkpoint_path': '/path/to/checkpoint.ckpt',
            'gripper_activation_threshold': 0.5,
            'gripper_activation_timesteps': 5,
            'policy_num_inference_steps': 16,
            'target_frame': 'base_link',
            'source_frame': 'tool0',
            'max_start_pose_attempts': 10,
            'start_pose_wait_duration': 1.0,
            'gripper_state_topic': '/gripper_state'
        }
        for key, val in params.items():
            self.declare_parameter(key, val)
        # Get parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.checkpoint_path = self.get_parameter('checkpoint_path').value
        self.gripper_threshold = self.get_parameter('gripper_activation_threshold').value
        self.gripper_timesteps = self.get_parameter('gripper_activation_timesteps').value
        self.num_inference_steps = self.get_parameter('policy_num_inference_steps').value
        self.target_frame = self.get_parameter('target_frame').value
        self.source_frame = self.get_parameter('source_frame').value
        self.max_start_pose_attempts = self.get_parameter('max_start_pose_attempts').value
        self.start_pose_wait_duration = self.get_parameter('start_pose_wait_duration').value
        self.gripper_state_topic = self.get_parameter('gripper_state_topic').value

        # Initialize variables
        self.bridge = CvBridge()
        self.lock = Lock()
        self.last_gripper_states = deque(maxlen=self.gripper_timesteps)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_pose_mat = None
        self.start_pose_ready = Event()

        # Buffers with history of 2
        self.camera_buffer = deque(maxlen=2)  # [3, H, W]
        self.robot0_eef_pos_buffer = deque(maxlen=2)  # [3]
        self.robot0_eef_rot_axis_angle_buffer = deque(maxlen=2)  # [6]
        self.robot0_eef_rot_axis_angle_wrt_start_buffer = deque(maxlen=2)  # [6]
        self.robot0_gripper_open_buffer = deque(maxlen=2)  # [1]

        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Load policy model
        self.load_policy()

        # Subscribers
        self.image_sub = self.create_subscription(CompressedImage, self.image_topic, self.image_callback, 1)
        self.gripper_state_sub = self.create_subscription(Bool, self.gripper_state_topic, self.gripper_state_callback, 10)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'desired_pose', 10)
        self.gripper_pub = self.create_publisher(Bool, 'gripper_command', 10)
        self.debug_image_pub = self.create_publisher(CompressedImage, 'debug_image_compressed', 10)

        # Start threads
        Thread(target=self.get_start_pose_thread, daemon=True).start()
        Thread(target=self.processing_loop, daemon=True).start()

        self.get_logger().info(f"Policy node initialized. Subscribed to {self.image_topic} and {self.gripper_state_topic}")

    def configure_logging(self):
        logging.getLogger().setLevel(logging.INFO)
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        for logger_name in ['torch', 'hydra', 'some_other_library']:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    def load_policy(self):
        ckpt = self.checkpoint_path
        if not ckpt.endswith('.ckpt'):
            ckpt = os.path.join(ckpt, 'checkpoints', 'latest.ckpt')
        try:
            payload = torch.load(open(ckpt, 'rb'), map_location='cpu', pickle_module=dill)
            self.get_logger().debug(f"Loaded payload from {ckpt}")
        except Exception as e:
            self.get_logger().error(f"Checkpoint load failed: {e}")
            sys.exit(1)
        cfg = payload['cfg']
        self.get_logger().info(f"Policy model: {cfg.policy.obs_encoder.model_name}")
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)
        self.policy = workspace.model.ema_model if getattr(cfg.training, 'use_ema', False) else workspace.model
        self.policy.num_inference_steps = self.num_inference_steps
        self.policy.eval().to(self.device)
        self.shape_meta = cfg.task.shape_meta
        try:
            ch, h, w = self.shape_meta['obs']['camera0_rgb']['shape']
            self.desired_channels, self.desired_height, self.desired_width = ch, h, w
            self.get_logger().info(f"Image expected: [C={ch}, H={h}, W={w}]")
        except KeyError as e:
            self.get_logger().error(f"Shape meta key missing: {e}")
            sys.exit(1)
        self.policy.reset()

    def get_start_pose_thread(self):
        attempts = 0
        while attempts < self.max_start_pose_attempts and rclpy.ok():
            try:
                if self.tf_buffer.can_transform(self.target_frame, self.source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                    trans = self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
                    pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z], dtype=np.float32)
                    quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
                    rot = st.Rotation.from_quat(quat).as_rotvec().astype(np.float32)
                    self.start_pose_mat = pose_to_mat(np.concatenate([pos, rot]))
                    # Initialize buffers
                    for _ in range(2):
                        self.camera_buffer.append(torch.zeros((self.desired_channels, self.desired_height, self.desired_width), dtype=torch.float32))
                        self.robot0_eef_pos_buffer.append(pos)
                        rot6d = mat_to_rot6d(st.Rotation.from_rotvec(rot).as_matrix())
                        self.robot0_eef_rot_axis_angle_buffer.append(rot6d)
                        self.robot0_eef_rot_axis_angle_wrt_start_buffer.append(rot6d)  # Initially same
                        self.robot0_gripper_open_buffer.append(0.0)
                    self.start_pose_ready.set()
                    self.get_logger().info("Start pose initialized.")
                    return
                else:
                    raise LookupException("Transform not available.")
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                attempts += 1
                self.get_logger().warn(f"TF2 error ({attempts}/{self.max_start_pose_attempts}): {e}")
                time.sleep(self.start_pose_wait_duration)
        self.get_logger().error("Failed to obtain start pose. Exiting.")
        sys.exit(1)

    def get_robot_eef_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z], dtype=np.float32)
            quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            rot = st.Rotation.from_quat(quat).as_rotvec().astype(np.float32)
            return pos, rot
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().error("Failed to get current EEF pose.")
            return None, None

    def image_callback(self, msg):
        with self.lock:
            try:
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
                if frame.ndim != 3 or frame.shape[2] != 3:
                    self.get_logger().error(f"Incorrect image shape: {frame.shape}")
                    return
                resized = cv2.resize(frame, (self.desired_width, self.desired_height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                tensor = torch.from_numpy(resized.transpose(2, 0, 1))  # [C, H, W]
                self.camera_buffer.append(tensor)
                self.get_logger().debug(f"Image resized to: {resized.shape}")
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge error: {e}")

    def gripper_state_callback(self, msg):
        with self.lock:
            state = 1.0 if msg.data else 0.0
            self.robot0_gripper_open_buffer.append(state)
            self.get_logger().debug(f"Gripper state updated: {state}")

    def processing_loop(self):
        if not self.start_pose_ready.wait(timeout=self.max_start_pose_attempts * self.start_pose_wait_duration + 1.0):
            self.get_logger().error("Start pose not ready in time. Exiting.")
            sys.exit(1)
        while rclpy.ok():
            with self.lock:
                if all(len(buf) == 2 for buf in [self.camera_buffer, self.robot0_eef_pos_buffer, self.robot0_eef_rot_axis_angle_buffer, self.robot0_eef_rot_axis_angle_wrt_start_buffer, self.robot0_gripper_open_buffer]):
                    obs = {
                        'camera0_rgb': torch.stack(list(self.camera_buffer)).unsqueeze(0).to(self.device),  # [1,2,3,H,W]
                        'robot0_eef_pos': torch.tensor(list(self.robot0_eef_pos_buffer)).unsqueeze(0).to(self.device),  # [1,2,3]
                        'robot0_eef_rot_axis_angle': torch.tensor(list(self.robot0_eef_rot_axis_angle_buffer)).unsqueeze(0).to(self.device),  # [1,2,6]
                        'robot0_eef_rot_axis_angle_wrt_start': torch.tensor(list(self.robot0_eef_rot_axis_angle_wrt_start_buffer)).unsqueeze(0).to(self.device),  # [1,2,6]
                        'robot0_gripper_open': torch.tensor(list(self.robot0_gripper_open_buffer)).unsqueeze(0).to(self.device)  # [1,2,1]
                    }
                else:
                    obs = None
            if obs:
                try:
                    action_pred, debug_info = self.process_frame(obs)
                    self.publish_outputs(action_pred, debug_info)
                except Exception as e:
                    self.get_logger().error(f"Processing error: {e}")
            time.sleep(0.01)

    def process_frame(self, obs_dict):
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            raw_action = result['action_pred'][0].cpu().numpy()[0]  # [action_dim]
            return self.process_policy_output(raw_action)

    def process_policy_output(self, raw_action):
        pos, rot6d, gripper = raw_action[:3], raw_action[3:9], raw_action[9]
        rot_mat = rot6d_to_mat(rot6d)
        rotvec = st.Rotation.from_matrix(rot_mat).as_rotvec()

        # Determine action_pose_repr
        repr_lower = self.action_pose_repr.lower()
        if repr_lower.startswith('abs'):
            new_pos, new_rotvec = pos, rotvec
            self.get_logger().debug("Action Pose: Absolute")
        elif repr_lower.startswith('rel'):
            start_pos = self.start_pose_mat[:3, 3]
            start_rot = st.Rotation.from_matrix(self.start_pose_mat[:3, :3]).as_rotvec()
            new_pos, new_rotvec = start_pos + pos, start_rot + rotvec
            self.get_logger().debug("Action Pose: Relative")
        elif repr_lower.startswith('del'):
            if len(self.robot0_eef_pos_buffer) < 1 or len(self.robot0_eef_rot_axis_angle_buffer) < 1:
                self.get_logger().error("Insufficient data for Delta action.")
                new_pos, new_rotvec = pos, rotvec
            else:
                current_pos = self.robot0_eef_pos_buffer[-1]
                current_rot6d = self.robot0_eef_rot_axis_angle_buffer[-1]
                current_rot_mat = rot6d_to_mat(current_rot6d)
                current_rotvec = st.Rotation.from_matrix(current_rot_mat).as_rotvec()
                new_pos, new_rotvec = current_pos + pos, current_rotvec + rotvec
                self.get_logger().debug("Action Pose: Delta")
        else:
            new_pos, new_rotvec = pos, rotvec
            self.get_logger().warn(f"Unknown action_pose_repr '{self.action_pose_repr}'. Using Absolute.")

        # Update relative rotation buffer
        rel_rot = self.compute_relative_rotation(new_rotvec)
        self.robot0_eef_rot_axis_angle_wrt_start_buffer.append(rel_rot)

        # Prepare debug info
        debug = {
            'delta_pos_cm': new_pos * 100,
            'delta_rot_rpy': st.Rotation.from_rotvec(new_rotvec).as_euler('xyz', degrees=True),
            'gripper_open_raw': gripper,
            'gripper_open_transformed': gripper > self.gripper_threshold
        }
        return {'position': new_pos, 'rotation': new_rotvec, 'gripper_open': gripper}, debug

    def compute_relative_rotation(self, current_rotvec):
        # Compute relative rotation w.r.t start pose
        rel_rotmat = convert_pose_mat_rep(pose_to_mat(np.concatenate([np.zeros(3), current_rotvec])), self.start_pose_mat, 'relative', False)[:3, :3]
        rel_rotvec = st.Rotation.from_matrix(rel_rotmat).as_rotvec().astype(np.float32)
        return mat_to_rot6d(rel_rotmat)

    def publish_outputs(self, action_pred, debug_info):
        # Publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.target_frame
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = action_pred['position']
        quat = st.Rotation.from_rotvec(action_pred['rotation']).as_quat()
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quat
        self.pose_pub.publish(pose_msg)

        # Gripper command logic
        self.last_gripper_states.append(action_pred['gripper_open'] > self.gripper_threshold)
        gripper_state = all(self.last_gripper_states)
        gripper_msg = Bool(data=gripper_state)
        self.gripper_pub.publish(gripper_msg)
        self.get_logger().debug(f"Gripper command: {'Open' if gripper_state else 'Close'}")

        # Publish debug image
        debug_img = self.create_debug_image(debug_info)
        try:
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img, dst_format='jpeg')
            self.debug_image_pub.publish(debug_msg)
            self.get_logger().debug("Published debug image.")
        except CvBridgeError as e:
            self.get_logger().error(f"Debug image publish failed: {e}")

    def create_debug_image(self, debug_info):
        with self.lock:
            if len(self.camera_buffer) > 0:
                img = (self.camera_buffer[-1].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            else:
                img = np.zeros((self.desired_height, self.desired_width, 3), dtype=np.uint8)
        # Overlay debug info
        cv2.putText(img, f"Delta Pos (cm): X={debug_info['delta_pos_cm'][0]:.2f}, Y={debug_info['delta_pos_cm'][1]:.2f}, Z={debug_info['delta_pos_cm'][2]:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(img, f"Delta Rot (deg): Roll={debug_info['delta_rot_rpy'][0]:.2f}, Pitch={debug_info['delta_rot_rpy'][1]:.2f}, Yaw={debug_info['delta_rot_rpy'][2]:.2f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(img, f"Gripper Open Raw: {debug_info['gripper_open_raw']:.4f}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(img, f"Gripper Open Transformed: {debug_info['gripper_open_transformed']}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return img

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main(sys.argv)
