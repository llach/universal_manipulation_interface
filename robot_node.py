#!/usr/bin/env python3

import os
import sys
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
import time
import logging  # Import Python's logging module
from collections import deque  # Import deque for buffering

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

# Import tf2_ros components
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')

        # Step 1: Configure Logging Levels
        self.configure_logging()

        # Initialize tf2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Declare parameters
        self.declare_parameter('image_topic', '/image_compressed')  # Default changed to /image_compressed
        self.declare_parameter('checkpoint_path', '/path/to/checkpoint.ckpt')
        self.declare_parameter('gripper_activation_threshold', 0.5)
        self.declare_parameter('gripper_activation_timesteps', 5)
        self.declare_parameter('policy_num_inference_steps', 16)
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('source_frame', 'tool0')
        self.declare_parameter('max_start_pose_attempts', 10)
        self.declare_parameter('start_pose_wait_duration', 1.0)  # seconds

        # Get parameters
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        self.gripper_threshold = self.get_parameter('gripper_activation_threshold').get_parameter_value().double_value
        self.gripper_timesteps = self.get_parameter('gripper_activation_timesteps').get_parameter_value().integer_value
        self.num_inference_steps = self.get_parameter('policy_num_inference_steps').get_parameter_value().integer_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.source_frame = self.get_parameter('source_frame').get_parameter_value().string_value
        self.max_start_pose_attempts = self.get_parameter('max_start_pose_attempts').get_parameter_value().integer_value
        self.start_pose_wait_duration = self.get_parameter('start_pose_wait_duration').get_parameter_value().double_value

        # Initialize variables
        self.bridge = CvBridge()
        self.lock = Lock()
        self.last_gripper_states = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_pose_mat = None  # To store the start pose matrix
        self.start_pose_ready = Event()  # Event to signal that start pose is ready

        # Initialize buffers for observations with history of 2
        self.camera_buffer = deque(maxlen=2)  # Stores [3, 224, 299] tensors
        self.robot0_eef_pos_buffer = deque(maxlen=2)  # Stores [3] numpy arrays
        self.robot0_eef_rot_axis_angle_buffer = deque(maxlen=2)  # Stores [6] numpy arrays
        self.robot0_gripper_open_buffer = deque(maxlen=2)  # Stores [1] floats
        self.robot0_eef_rot_axis_angle_wrt_start_buffer = deque(maxlen=2)  # Stores [6] numpy arrays

        # Load the policy
        self.load_policy()

        # Subscribe to compressed image topic
        self.image_sub = self.create_subscription(
            CompressedImage,  # Changed to CompressedImage
            self.image_topic,
            self.image_callback,
            1  # Queue size of 1 to limit buffering
        )

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'desired_pose', 10)
        self.gripper_pub = self.create_publisher(Bool, 'gripper_command', 10)
        self.debug_image_pub = self.create_publisher(CompressedImage, 'debug_image_compressed', 10)  # Changed to CompressedImage

        # Start the start pose retrieval thread
        self.start_pose_thread = Thread(target=self.get_start_pose_thread)
        self.start_pose_thread.daemon = True
        self.start_pose_thread.start()

        # Start the processing thread
        self.processing_thread = Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info(f"Policy node initialized. Subscribed to {self.image_topic}")

    def configure_logging(self):
        """
        Configures the logging levels to ensure that only this node's debug logs are shown.
        """
        # Step 1.1: Set the root logger to INFO to suppress debug logs from other libraries
        logging.getLogger().setLevel(logging.INFO)

        # Step 1.2: Set this node's logger to DEBUG
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # Step 1.3: (Optional) Reduce the verbosity of external libraries if necessary
        # Example: Suppress debug logs from 'torch' and 'hydra'
        external_loggers = ['torch', 'hydra', 'some_other_library']
        for logger_name in external_loggers:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    def load_policy(self):
        # Load the checkpoint
        ckpt_path = self.checkpoint_path
        if not ckpt_path.endswith('.ckpt'):
            ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
        try:
            payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
            self.get_logger().debug(f"Loaded payload from checkpoint: {ckpt_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load checkpoint: {e}")
            sys.exit(1)
        
        cfg = payload['cfg']
        self.get_logger().info(f"Loaded policy model: {cfg.policy.obs_encoder.model_name}")

        # Load the policy
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self.policy = workspace.model
        if getattr(cfg.training, 'use_ema', False):
            self.policy = workspace.ema_model
        self.policy.num_inference_steps = self.num_inference_steps  # Adjust as needed
        self.policy.eval().to(self.device)

        # Store other necessary configurations
        self.shape_meta = cfg.task.shape_meta
        self.obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = cfg.task.pose_repr.action_pose_repr

        # Extract desired image shape from shape_meta
        # Assuming shape_meta.obs.camera0_rgb.shape = [channels, height, width]
        try:
            camera_shape = self.shape_meta['obs']['camera0_rgb']['shape']
            if len(camera_shape) != 3:
                self.get_logger().error(f"Expected shape list of length 3 for camera0_rgb, got {len(camera_shape)}")
                sys.exit(1)
            self.desired_channels, self.desired_height, self.desired_width = camera_shape
            self.get_logger().info(f"Expected image shape: Channels={self.desired_channels}, Height={self.desired_height}, Width={self.desired_width}")
        except KeyError as e:
            self.get_logger().error(f"Missing key in shape_meta: {e}")
            sys.exit(1)
        except Exception as e:
            self.get_logger().error(f"Error parsing shape_meta: {e}")
            sys.exit(1)

        # Reset policy state if needed
        self.policy.reset()

    def get_start_pose_thread(self):
        """
        Thread function to retrieve and store the start pose.
        """
        attempts = 0
        while attempts < self.max_start_pose_attempts:
            try:
                # Wait until the transform is available
                can_transform = self.tf_buffer.can_transform(
                    self.target_frame,
                    self.source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                if not can_transform:
                    raise LookupException(f"Cannot transform from {self.source_frame} to {self.target_frame}")

                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    self.target_frame,   # target frame
                    self.source_frame,   # source frame
                    rclpy.time.Time(),  # Time stamp (latest available)
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )

                # Extract translation
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                z = transform.transform.translation.z
                eef_pos = np.array([x, y, z], dtype=np.float32)

                # Extract rotation as quaternion
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                rotation = st.Rotation.from_quat([qx, qy, qz, qw])

                # Convert rotation to axis-angle
                eef_rot_axis_angle = rotation.as_rotvec().astype(np.float32)

                # Convert to pose matrix
                start_pose = np.concatenate([eef_pos, eef_rot_axis_angle], axis=0)  # [6]
                self.start_pose_mat = pose_to_mat(start_pose)

                self.get_logger().debug(f"Start robot0_eef_pos: {eef_pos}")
                self.get_logger().debug(f"Start robot0_eef_rot_axis_angle: {eef_rot_axis_angle}")

                # Initialize buffers with the start pose duplicated
                # For camera_buffer, we'll initialize with two zero tensors
                for _ in range(2):
                    # Initialize camera_buffer with zero tensors of expected size
                    self.camera_buffer.append(torch.zeros((self.desired_channels, self.desired_height, self.desired_width), dtype=torch.float32))
                    self.robot0_eef_pos_buffer.append(eef_pos)
                    rot6d = mat_to_rot6d(st.Rotation.from_rotvec(eef_rot_axis_angle).as_matrix())
                    self.robot0_eef_rot_axis_angle_buffer.append(rot6d)
                    # Compute relative rotation w.r.t start pose (initially zero)
                    rel_rot6d = rot6d  # Assuming relative to start is zero initially
                    self.robot0_eef_rot_axis_angle_wrt_start_buffer.append(rel_rot6d)
                    self.robot0_gripper_open_buffer.append(0.0)  # Assuming gripper is initially closed

                # Signal that start pose is ready
                self.start_pose_ready.set()
                self.get_logger().info("Start pose initialized and buffers are set.")
                return

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                attempts += 1
                self.get_logger().warn(f"Attempt {attempts}/{self.max_start_pose_attempts}: TF2 Lookup Error while getting start pose: {e}")
                time.sleep(self.start_pose_wait_duration)

        self.get_logger().error(f"Failed to get start pose after {self.max_start_pose_attempts} attempts. Exiting.")
        sys.exit(1)

    def get_robot_eef_pose(self):
        """
        Retrieves the end-effector's current position and rotation from TF2.
        Returns:
            tuple: (eef_pos, eef_rot_axis_angle) where
                eef_pos (numpy.ndarray): [x, y, z] position in 'base_link' frame.
                eef_rot_axis_angle (numpy.ndarray): [x, y, z] rotation in axis-angle.
            tuple: (None, None) if the transform is unavailable.
        """
        try:
            # Lookup the latest transform from target_frame to source_frame
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame,   # target frame
                self.source_frame,   # source frame
                rclpy.time.Time(),  # Time stamp (latest available)
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Extract translation
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z
            eef_pos = np.array([x, y, z], dtype=np.float32)

            # Extract rotation as quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            rotation = st.Rotation.from_quat([qx, qy, qz, qw])

            # Convert rotation to axis-angle
            eef_rot_axis_angle = rotation.as_rotvec().astype(np.float32)

            self.get_logger().debug(f"Retrieved robot0_eef_pos: {eef_pos}")
            self.get_logger().debug(f"Retrieved robot0_eef_rot_axis_angle: {eef_rot_axis_angle}")

            return eef_pos, eef_rot_axis_angle

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"TF2 Lookup Error: {e}")
            return None, None

    def image_callback(self, msg):
        with self.lock:
            try:
                # Convert ROS CompressedImage message to OpenCV image
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
                # Verify that the image has three channels
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    self.get_logger().error(f"Received image has incorrect shape: {frame.shape}. Expected 3 channels.")
                    return
                # Resize the image to match the policy's expected input shape
                resized_frame = cv2.resize(frame, (self.desired_width, self.desired_height), interpolation=cv2.INTER_AREA)
                # Normalize and convert the image to float32
                resized_frame = resized_frame.astype(np.float32) / 255.0
                # Convert to tensor and append to camera_buffer
                frame_tensor = torch.from_numpy(np.transpose(resized_frame, (2, 0, 1)))  # [3, 224, 299]
                self.camera_buffer.append(frame_tensor)
                self.get_logger().debug("Received and stored a new compressed image.")
                self.get_logger().debug(f"Image shape after conversion and resizing: {resized_frame.shape}")
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge Error: {e}")
                return

    def processing_loop(self):
        # Wait until the start pose is ready
        if not self.start_pose_ready.wait(timeout=self.max_start_pose_attempts * self.start_pose_wait_duration + 1.0):
            self.get_logger().error("Start pose was not set within the expected time. Exiting.")
            sys.exit(1)

        while rclpy.ok():
            with self.lock:
                # Check if all buffers have 2 samples
                if (len(self.camera_buffer) == 2 and
                    len(self.robot0_eef_pos_buffer) == 2 and
                    len(self.robot0_eef_rot_axis_angle_buffer) == 2 and
                    len(self.robot0_eef_rot_axis_angle_wrt_start_buffer) == 2 and
                    len(self.robot0_gripper_open_buffer) == 2):
                    
                    # Build observation dictionary from buffers
                    obs_dict = {
                        'camera0_rgb': torch.stack(list(self.camera_buffer), dim=0).unsqueeze(0).to(self.device),  # [1, 2, 3, 224, 299]
                        'robot0_eef_pos': torch.tensor(list(self.robot0_eef_pos_buffer), dtype=torch.float32).unsqueeze(0).to(self.device),  # [1, 2, 3]
                        'robot0_eef_rot_axis_angle': torch.tensor(list(self.robot0_eef_rot_axis_angle_buffer), dtype=torch.float32).unsqueeze(0).to(self.device),  # [1, 2, 6]
                        'robot0_eef_rot_axis_angle_wrt_start': torch.tensor(list(self.robot0_eef_rot_axis_angle_wrt_start_buffer), dtype=torch.float32).unsqueeze(0).to(self.device),  # [1, 2, 6]
                        'robot0_gripper_open': torch.tensor(list(self.robot0_gripper_open_buffer), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)  # [1, 2, 1]
                    }

                    # self.get_logger().debug(f"Observation Dictionary:")
                    # for key, value in obs_dict.items():
                    #     self.get_logger().debug(f"  {key}: shape={value.shape}, data={value}")

                else:
                    # Not enough data to process
                    obs_dict = None

            if obs_dict is not None:
                # Process the buffered observations
                action_pred, debug_info = self.process_frame(obs_dict)
                # Publish the outputs
                self.publish_outputs(action_pred, debug_info)
            else:
                # No new data to process
                pass

            time.sleep(0.01)  # Small sleep to prevent tight loop

    def process_frame(self, obs_dict):
        # Run the policy to get action prediction
        with torch.no_grad():
            try:
                result = self.policy.predict_action(obs_dict)
                raw_action = result['action_pred'][0].detach().cpu().numpy()
                raw_action_t0 = raw_action[0]  # First action step
                # Process the policy output
                action_pred, debug_info = self.process_policy_output(raw_action_t0)
            except Exception as e:
                self.get_logger().error(f"Policy inference error: {e}")
                raise e  # Re-raise the exception after logging

        # Debug: Print processed policy output if verbosity is DEBUG
        if self.get_logger().is_enabled_for(rclpy.logging.LoggingSeverity.DEBUG):
            self.get_logger().debug(f"Processed Policy Output: {action_pred}")
            self.get_logger().debug(f"Debug Info: {debug_info}")

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

    def publish_outputs(self, action_pred, debug_info):
        # Publish the desired pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.target_frame  # Adjust the frame as needed  
        pose_msg.pose.position.x = float(action_pred['position'][0])
        pose_msg.pose.position.y = float(action_pred['position'][1])
        pose_msg.pose.position.z = float(action_pred['position'][2])
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
            self.get_logger().debug("Published Gripper Open Command.")
        else:
            gripper_msg = Bool()
            gripper_msg.data = False  # Gripper close command
            self.gripper_pub.publish(gripper_msg)
            self.get_logger().debug("Published Gripper Close Command.")

        # Publish debug image with overlays
        debug_frame = self.create_debug_image(debug_info)
        try:
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(debug_frame, dst_format='jpeg')
            self.debug_image_pub.publish(debug_image_msg)
            self.get_logger().debug("Published Debug Image with Overlays.")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error while publishing debug image: {e}")

    def create_debug_image(self, debug_info):
        # Overlay the policy outputs on the latest camera image for better visualization
        if len(self.camera_buffer) > 0:
            latest_camera_tensor = self.camera_buffer[-1].cpu().numpy()  # [3, 224, 299]
            # Convert tensor to image format
            latest_camera_image = np.transpose(latest_camera_tensor, (1, 2, 0))  # [224, 299, 3]
            latest_camera_image = (latest_camera_image * 255).astype(np.uint8)
        else:
            # Create a blank image if no camera data is available
            latest_camera_image = np.zeros((self.desired_height, self.desired_width, 3), dtype=np.uint8)

        # Overlay the policy outputs on the frame
        delta_pos_cm = debug_info['delta_pos_cm']
        delta_rot_rpy = debug_info['delta_rot_rpy']
        gripper_open_raw = debug_info['gripper_open_raw']
        gripper_open_transformed = debug_info['gripper_open_transformed']

        # Prepare text lines
        line1 = f'Delta Pos (cm): X={delta_pos_cm[0]:.2f}, Y={delta_pos_cm[1]:.2f}, Z={delta_pos_cm[2]:.2f}'
        line2 = f'Delta Rot (deg): Roll={delta_rot_rpy[0]:.2f}, Pitch={delta_rot_rpy[1]:.2f}, Yaw={delta_rot_rpy[2]:.2f}'
        line3 = f'Gripper Open Raw: {gripper_open_raw:.4f}'
        line4 = f'Gripper Open Transformed: {gripper_open_transformed}'

        # Overlay text on the frame
        y0, dy = 30, 30
        for i, line in enumerate([line1, line2, line3, line4]):
            y = y0 + i * dy
            cv2.putText(latest_camera_image, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return latest_camera_image

    def destroy_node(self):
        # Clean up the processing thread if necessary
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
    main(args=sys.argv)  # Pass command-line arguments to rclpy
