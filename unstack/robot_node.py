#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
from threading import Lock, Thread
import threading

# Import your policy and helper functions
import hydra
import dill
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import (
    normalize, mat_to_rot6d, rot6d_to_mat,
    pose_to_mat, mat_to_pose
)
import scipy.spatial.transform as st

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')

        # Declare parameters
        self.declare_parameter('image_topic', '/image')
        self.declare_parameter('checkpoint_path', '/path/to/checkpoint.ckpt')
        self.declare_parameter('gripper_activation_threshold', 0.5)
        self.declare_parameter('gripper_activation_timesteps', 5)
        self.declare_parameter('policy_num_inference_steps', 16)
        self.declare_parameter('processing_rate', 10.0)  # Hz

        # Get parameters
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        self.gripper_threshold = self.get_parameter('gripper_activation_threshold').get_parameter_value().double_value
        self.gripper_timesteps = self.get_parameter('gripper_activation_timesteps').get_parameter_value().integer_value
        self.num_inference_steps = self.get_parameter('policy_num_inference_steps').get_parameter_value().integer_value
        self.processing_rate = self.get_parameter('processing_rate').get_parameter_value().double_value

        # Initialize variables
        self.bridge = CvBridge()
        self.lock = Lock()
        self.latest_image = None
        self.inference_running = False
        self.last_gripper_states = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the policy
        self.load_policy()

        # Subscribe to image topic
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            1  # Queue size of 1 to limit buffering
        )

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'desired_pose', 10)
        self.gripper_pub = self.create_publisher(Bool, 'gripper_command', 10)
        self.debug_image_pub = self.create_publisher(Image, 'debug_image', 10)

        # Start the processing thread
        self.processing_thread = Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info(f"Policy node initialized. Subscribed to {self.image_topic}")

    def load_policy(self):
        # Load the checkpoint
        ckpt_path = self.checkpoint_path
        if not ckpt_path.endswith('.ckpt'):
            ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
        payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']
        self.get_logger().info(f"Loaded policy model: {cfg.policy.obs_encoder.model_name}")

        # Load the policy
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model
        self.policy.num_inference_steps = self.num_inference_steps  # Adjust as needed
        self.policy.eval().to(self.device)

        # Store other necessary configurations
        self.shape_meta = cfg.task.shape_meta
        self.obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = cfg.task.pose_repr.action_pose_repr

        # Initialize the observation dictionary
        self.obs_dict = {}
        self.last_observation_time = None

        # Reset policy state if needed
        self.policy.reset()

    def image_callback(self, msg):
        with self.lock:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # Store the latest image
            self.latest_image = frame.copy()
            # Note: We don't process the image here to avoid blocking

    def processing_loop(self):
        rate = self.create_rate(self.processing_rate)
        while rclpy.ok():
            # Check if a new image is available and inference is not running
            with self.lock:
                if self.latest_image is not None and not self.inference_running:
                    frame = self.latest_image.copy()
                    self.latest_image = None  # Reset the latest image
                    self.inference_running = True  # Set inference running flag
                else:
                    frame = None

            if frame is not None:
                # Process the frame
                action_pred, debug_info = self.process_frame(frame)
                # Publish the outputs
                self.publish_outputs(action_pred, debug_info, frame)
                # Reset inference running flag
                with self.lock:
                    self.inference_running = False
            else:
                # No new image to process
                pass

            rate.sleep()

    def process_frame(self, frame):
        # Preprocess the image
        img = frame.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # Convert to (Channels, Height, Width)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # Prepare observation dictionary
        obs_dict = {f'camera0_rgb': img}

        # Include other necessary observations (e.g., robot state)
        # For simplicity, we're only using the image in this example

        # Run the policy to get action prediction
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            raw_action = result['action_pred'][0].detach().cpu().numpy()
            raw_action_t0 = raw_action[0]  # First action step
            # Process the policy output
            action_pred, debug_info = self.process_policy_output(raw_action_t0)
        return action_pred, debug_info

    def process_policy_output(self, raw_action_t):
        pos = raw_action_t[:3]
        rot6d = raw_action_t[3:9]
        gripper_open_raw = raw_action_t[9]

        # Convert 6D rotation back to rotation matrix
        rot_mat = rot6d_to_mat(rot6d)
        # Convert rotation matrix to rotation vector (axis-angle)
        rotvec = st.Rotation.from_matrix(rot_mat).as_rotvec()

        # Depending on the action_pose_repr, convert to absolute pose
        if self.action_pose_repr == 'delta':
            # Apply delta to current pose
            # For simplicity, assuming current pose is zero
            new_pos = pos
            new_rotvec = rotvec
        elif self.action_pose_repr == 'absolute':
            new_pos = pos
            new_rotvec = rotvec
        else:
            # Handle other representations if needed
            new_pos = pos
            new_rotvec = rotvec

        # Prepare action prediction
        action_pred = {
            'position': new_pos,
            'rotation': new_rotvec,
            'gripper_open': gripper_open_raw
        }

        # Prepare debug information
        delta_pos_cm = new_pos * 100.0  # Convert to cm
        delta_rot_rpy = st.Rotation.from_rotvec(new_rotvec).as_euler('xyz', degrees=True)
        gripper_open_transformed = gripper_open_raw > self.gripper_threshold

        debug_info = {
            'delta_pos_cm': delta_pos_cm,
            'delta_rot_rpy': delta_rot_rpy,
            'gripper_open_raw': gripper_open_raw,
            'gripper_open_transformed': gripper_open_transformed
        }

        return action_pred, debug_info

    def publish_outputs(self, action_pred, debug_info, frame):
        # Publish the desired pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_link'  # Adjust the frame as needed
        pose_msg.pose.position.x = action_pred['position'][0]
        pose_msg.pose.position.y = action_pred['position'][1]
        pose_msg.pose.position.z = action_pred['position'][2]
        # Convert rotation vector to quaternion
        rotvec = action_pred['rotation']
        quat = st.Rotation.from_rotvec(rotvec).as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

        # Gripper command logic
        gripper_open_raw = action_pred['gripper_open']
        gripper_open_transformed = gripper_open_raw > self.gripper_threshold
        self.last_gripper_states.append(gripper_open_transformed)
        if len(self.last_gripper_states) > self.gripper_timesteps:
            self.last_gripper_states.pop(0)
        # Apply gripper command if condition is met
        if all(self.last_gripper_states):
            gripper_msg = Bool()
            gripper_msg.data = True  # Gripper open command
            self.gripper_pub.publish(gripper_msg)
        else:
            gripper_msg = Bool()
            gripper_msg.data = False  # Gripper close command
            self.gripper_pub.publish(gripper_msg)

        # Publish debug image with overlays
        debug_frame = self.create_debug_image(frame, debug_info)
        debug_image_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='rgb8')
        self.debug_image_pub.publish(debug_image_msg)

    def create_debug_image(self, frame, debug_info):
        # Overlay the policy outputs on the frame
        delta_pos_cm = debug_info['delta_pos_cm']
        delta_rot_rpy = debug_info['delta_rot_rpy']
        gripper_open_raw = debug_info['gripper_open_raw']
        gripper_open_transformed = debug_info['gripper_open_transformed']

        # Prepare text lines
        line1 = f'Delta Pos (cm): {delta_pos_cm[0]:.2f}, {delta_pos_cm[1]:.2f}, {delta_pos_cm[2]:.2f}'
        line2 = f'Delta Rot (deg): {delta_rot_rpy[0]:.2f}, {delta_rot_rpy[1]:.2f}, {delta_rot_rpy[2]:.2f}'
        line3 = f'Gripper Open Raw: {gripper_open_raw:.4f}'
        line4 = f'Gripper Open Transformed: {gripper_open_transformed}'

        # Overlay text on the frame
        y0, dy = 30, 30
        for i, line in enumerate([line1, line2, line3, line4]):
            y = y0 + i * dy
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def destroy_node(self):
        # Clean up the processing thread
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
    main()
