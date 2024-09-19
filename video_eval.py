import os
import pathlib
import click
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from tqdm import tqdm
import dill
import scipy.spatial.transform as st
import av
import cv2  # Import OpenCV for image processing

# Use the specified imports for helper functions
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import (
    normalize, mat_to_rot6d, rot6d_to_mat,
    pose_to_mat, mat_to_pose
)

# Import the shared data processing methods
from data_processing import process_data_directories

@click.command()
@click.argument('data_dirs', nargs=-1)
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint (.ckpt file)')
@click.option('--output', '-o', default=None, help='Output file to save errors (optional)')
@click.option('--video_output', '-vo', default='output_video.mp4', help='Output video file path')
@click.option('--display', is_flag=True, help='Display frames live during processing')
def main(data_dirs, checkpoint, output, video_output, display):
    # Load the checkpoint
    ckpt_path = checkpoint
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("Model name:", cfg.policy.obs_encoder.model_name)
    print("Dataset path used during training:", cfg.task.dataset.dataset_path)
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    print('Observation pose representation:', obs_pose_rep)
    print('Action pose representation:', action_pose_repr)

    # Load the policy
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16  # Adjust as needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)

    # Process data directories to get episodes_info
    episodes_info = process_data_directories(data_dirs)
    if not episodes_info:
        print("No valid episodes found.")
        return

    shape_meta = cfg.task.shape_meta

    # Assuming single robot and single camera
    robot_name = 'robot0'
    camera_name = 'camera0'

    total_pos_error = 0
    total_rot_error = 0
    total_gripper_error = 0
    total_steps = 0
    pos_errors = []
    rot_errors = []
    gripper_errors = []

    # Initialize video writer
    video_writer = None

    # Iterate over each episode
    for episode_idx, episode_info in enumerate(tqdm(episodes_info, desc='Episodes')):
        episode_data = episode_info['episode_data']
        num_steps = episode_data[f'{robot_name}_eef_pos'].shape[0]
        video_path = episode_info['video_path']

        # Open video file
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(stream)]
        container.close()

        # Ensure that the number of frames matches the number of steps
        if len(frames) < num_steps:
            print(f"Episode {episode_idx} has fewer frames than steps.")
            num_steps = len(frames)

        # Get the starting pose of the episode
        start_pos = episode_data[f'{robot_name}_eef_pos'][0]
        start_rotvec = episode_data[f'{robot_name}_eef_rot_axis_angle'][0]
        start_pose = np.concatenate([start_pos, start_rotvec], axis=-1)
        start_pose_mat = pose_to_mat(start_pose)

        # Reset policy state if needed
        policy.reset()

        # Prepare to write video frames
        if video_writer is None:
            # Get frame dimensions
            frame_height, frame_width, _ = frames[0].shape
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_output, fourcc, 30.0, (frame_width, frame_height))

        for step_idx in range(num_steps):
            # Construct observation dictionary
            obs_dict_np = construct_observation_dict(
                episode_data, frames, step_idx, shape_meta, robot_name, camera_name, start_pose_mat
            )
            # Convert observations to tensors and move to device
            obs_dict = dict_apply(
                obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
            )
            # Run the policy to get action prediction
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
                raw_action = result['action_pred'][0].detach().cpu().numpy()
                raw_action_t0 = raw_action[0]  # First action step
                # Convert policy output to action format
                action_pred = process_policy_output(
                    raw_action_t0, action_pose_repr, episode_data, step_idx, robot_name
                )
            # Get ground truth action from dataset
            action_gt = get_ground_truth_action(
                episode_data, step_idx, num_steps, robot_name
            )
            # Compute errors
            pos_error = np.linalg.norm(action_pred[:3] - action_gt[:3])
            rot_error = rotation_error(action_pred[3:6], action_gt[3:6])
            gripper_error = gripper_state_error(action_pred[6], action_gt[6])

            total_pos_error += pos_error
            total_rot_error += rot_error
            total_gripper_error += gripper_error
            total_steps += 1
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)
            gripper_errors.append(gripper_error)

            # Print errors for the current frame
            print(f'Episode {episode_idx}, Step {step_idx}, Pos Error: {pos_error:.4f}, Rot Error: {rot_error:.4f}, Gripper Error: {gripper_error}')

            # Visualize and save frame with overlay
            frame = frames[step_idx].copy()
            # Overlay text showing the prediction errors
            line1 = f'Episode: {episode_idx}, Step: {step_idx}'
            line2 = f'Pos Error: {pos_error:.4f}'
            line3 = f'Rot Error: {rot_error:.4f}'
            line4 = f'Gripper Error: {gripper_error}'
            # Position the text lines
            y0, dy = 30, 30
            for i, line in enumerate([line1, line2, line3, line4]):
                y = y0 + i*dy
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Write the frame to the video
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Display the frame live if the display option is enabled
            if display:
                cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting due to user input.")
                    break  # Exit the loop if 'q' is pressed

        else:
            continue  # Continue if the inner loop wasn't broken
        break  # Break the outer loop if 'q' was pressed in inner loop

    # Release the video writer
    if video_writer is not None:
        video_writer.release()
        print(f'Video saved to {video_output}')

    # Destroy all OpenCV windows if display was enabled
    if display:
        cv2.destroyAllWindows()

    # Compute average errors
    if total_steps > 0:
        avg_pos_error = total_pos_error / total_steps
        avg_rot_error = total_rot_error / total_steps
        avg_gripper_error = total_gripper_error / total_steps
    else:
        avg_pos_error = avg_rot_error = avg_gripper_error = 0

    print(f'Average positional error over {total_steps} steps: {avg_pos_error}')
    print(f'Average rotational error over {total_steps} steps: {avg_rot_error}')
    print(f'Average gripper error over {total_steps} steps: {avg_gripper_error}')

    # Optionally save errors to output file
    if output is not None:
        np.savez(output, pos_errors=np.array(pos_errors), rot_errors=np.array(rot_errors), gripper_errors=np.array(gripper_errors))
        print(f'Errors saved to {output}')

def construct_observation_dict(episode_data, frames, step_idx, shape_meta, robot_name, camera_name, start_pose_mat):
    """
    Constructs the observation dictionary from the episode data at the given step index.
    """
    obs_dict = {}
    for key in shape_meta.obs:
        meta = shape_meta.obs[key]
        horizon = meta.horizon
        # Handle horizon
        if horizon == 1:
            # Get the data at step_idx
            if key == f'{camera_name}_rgb':
                # Get the image frame
                img = frames[step_idx]
                # Preprocess image if necessary (e.g., resize, normalize)
                img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
                # Convert to (Channels, Height, Width)
                img = np.transpose(img, (2, 0, 1))
                obs_dict[key] = img
            elif key in episode_data:
                data = episode_data[key][step_idx]
                # Check if the key contains '_rot_axis_angle'
                if '_rot_axis_angle' in key:
                    # Convert axis-angle to 6D rotation representation
                    rotvec = data
                    rot_mat = st.Rotation.from_rotvec(rotvec).as_matrix()
                    rot6d = mat_to_rot6d(rot_mat)
                    obs_dict[key] = rot6d
                else:
                    obs_dict[key] = data
            elif key == f'{robot_name}_eef_pos_wrt_start' or key == f'{robot_name}_eef_rot_axis_angle_wrt_start':
                # Compute pose relative to start pose
                curr_pos = episode_data[f'{robot_name}_eef_pos'][step_idx]
                curr_rotvec = episode_data[f'{robot_name}_eef_rot_axis_angle'][step_idx]
                curr_pose = np.concatenate([curr_pos, curr_rotvec], axis=-1)
                curr_pose_mat = pose_to_mat(curr_pose)
                rel_pose_mat = convert_pose_mat_rep(
                    curr_pose_mat,
                    base_pose_mat=start_pose_mat,
                    pose_rep='relative',
                    backward=False
                )
                rel_pose = mat_to_pose(rel_pose_mat)
                if key == f'{robot_name}_eef_pos_wrt_start':
                    obs_dict[key] = rel_pose[:3]
                else:  # key == f'{robot_name}_eef_rot_axis_angle_wrt_start'
                    # Convert rotation to 6D representation
                    rotvec = rel_pose[3:]
                    rot_mat = st.Rotation.from_rotvec(rotvec).as_matrix()
                    rot6d = mat_to_rot6d(rot_mat)
                    obs_dict[key] = rot6d
            else:
                # Handle other cases as needed
                pass
        else:
            # Handle horizons greater than 1
            start_idx = max(0, step_idx - horizon + 1)
            end_idx = step_idx + 1
            if key == f'{camera_name}_rgb':
                # Get the image frames
                imgs = frames[start_idx:end_idx]
                # Pad if necessary
                if len(imgs) < horizon:
                    padding = [imgs[0]] * (horizon - len(imgs))
                    imgs = padding + imgs
                # Process images: Normalize and transpose
                imgs = np.stack([
                    np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
                    for img in imgs
                ], axis=0)
                obs_dict[key] = imgs
            elif key in episode_data:
                data = episode_data[key][start_idx:end_idx]
                # Pad if necessary
                if data.shape[0] < horizon:
                    padding = np.repeat(data[0:1], horizon - data.shape[0], axis=0)
                    data = np.concatenate([padding, data], axis=0)
                # Check if the key contains '_rot_axis_angle'
                if '_rot_axis_angle' in key:
                    # Convert axis-angle to 6D rotation representation
                    rotvec = data
                    rot_mats = st.Rotation.from_rotvec(rotvec).as_matrix()
                    rot6d = mat_to_rot6d(rot_mats)
                    obs_dict[key] = rot6d
                else:
                    obs_dict[key] = data
            elif key == f'{robot_name}_eef_pos_wrt_start' or key == f'{robot_name}_eef_rot_axis_angle_wrt_start':
                # Compute pose relative to start pose
                curr_pos = episode_data[f'{robot_name}_eef_pos'][start_idx:end_idx]
                curr_rotvec = episode_data[f'{robot_name}_eef_rot_axis_angle'][start_idx:end_idx]
                # Pad if necessary
                if curr_pos.shape[0] < horizon:
                    padding_pos = np.repeat(curr_pos[0:1], horizon - curr_pos.shape[0], axis=0)
                    curr_pos = np.concatenate([padding_pos, curr_pos], axis=0)
                    padding_rotvec = np.repeat(curr_rotvec[0:1], horizon - curr_rotvec.shape[0], axis=0)
                    curr_rotvec = np.concatenate([padding_rotvec, curr_rotvec], axis=0)
                curr_pose = np.concatenate([curr_pos, curr_rotvec], axis=-1)
                curr_pose_mat = pose_to_mat(curr_pose)
                # Repeat start_pose_mat to match the horizon length
                base_pose_mat = np.repeat(start_pose_mat[np.newaxis, :, :], horizon, axis=0)
                rel_pose_mat = convert_pose_mat_rep(
                    curr_pose_mat,
                    base_pose_mat=base_pose_mat,
                    pose_rep='relative',
                    backward=False
                )
                rel_pose = mat_to_pose(rel_pose_mat)
                if key == f'{robot_name}_eef_pos_wrt_start':
                    obs_dict[key] = rel_pose[..., :3]
                else:  # key == f'{robot_name}_eef_rot_axis_angle_wrt_start'
                    # Convert rotation to 6D representation
                    rotvec = rel_pose[..., 3:]
                    rot_mats = st.Rotation.from_rotvec(rotvec).as_matrix()
                    rot6d = mat_to_rot6d(rot_mats)
                    obs_dict[key] = rot6d
            else:
                # Handle other cases as needed
                pass
    return obs_dict

def process_policy_output(raw_action_t, action_pose_repr, episode_data, step_idx, robot_name):
    """
    Converts the raw policy output to the desired action format.
    """
    pos = raw_action_t[:3]
    rot6d = raw_action_t[3:9]
    gripper = raw_action_t[9]

    # Convert 6D rotation back to rotation matrix
    rot_mat = rot6d_to_mat(rot6d)
    # Convert rotation matrix to rotation vector (axis-angle)
    rotvec = st.Rotation.from_matrix(rot_mat).as_rotvec()

    # Depending on the action_pose_repr, convert to absolute pose
    if action_pose_repr == 'delta':
        # Apply delta to current pose
        curr_pos = episode_data[f'{robot_name}_eef_pos'][step_idx]
        curr_rotvec = episode_data[f'{robot_name}_eef_rot_axis_angle'][step_idx]
        new_pos = curr_pos + pos
        new_rotvec = curr_rotvec + rotvec
    elif action_pose_repr == 'absolute':
        new_pos = pos
        new_rotvec = rotvec
    else:
        # Handle other representations if needed
        new_pos = pos
        new_rotvec = rotvec

    action_pred = np.concatenate([new_pos, new_rotvec, [gripper]], axis=0)
    return action_pred

def get_ground_truth_action(episode_data, step_idx, num_steps, robot_name):
    """
    Computes the ground truth action from the dataset.
    """
    if step_idx < num_steps - 1:
        # Next pose
        next_pos = episode_data[f'{robot_name}_eef_pos'][step_idx + 1]
        next_rotvec = episode_data[f'{robot_name}_eef_rot_axis_angle'][step_idx + 1]
        next_gripper = episode_data[f'{robot_name}_gripper_open'][step_idx + 1]
        # Current pose
        curr_pos = episode_data[f'{robot_name}_eef_pos'][step_idx]
        curr_rotvec = episode_data[f'{robot_name}_eef_rot_axis_angle'][step_idx]
        curr_gripper = episode_data[f'{robot_name}_gripper_open'][step_idx]
        # Ensure gripper values are scalars
        next_gripper = float(next_gripper)
        curr_gripper = float(curr_gripper)
        # Compute ground truth action as the delta
        delta_pos = next_pos - curr_pos
        delta_rotvec = next_rotvec - curr_rotvec
        delta_gripper = next_gripper - curr_gripper
        delta_gripper = np.array([delta_gripper], dtype=np.float32)  # Ensure it's a 1D array
        action_gt = np.concatenate([delta_pos, delta_rotvec, delta_gripper], axis=0)
    else:
        # For the last step, assume zero action
        action_gt = np.zeros(7, dtype=np.float32)
    return action_gt

def rotation_error(rotvec_pred, rotvec_gt):
    """
    Computes the angular difference between two rotations represented as rotation vectors.
    Returns the error in degrees.
    """
    rot_pred = st.Rotation.from_rotvec(rotvec_pred)
    rot_gt = st.Rotation.from_rotvec(rotvec_gt)
    # Compute relative rotation
    rel_rot = rot_gt.inv() * rot_pred  # Note: Changed order for correct relative rotation
    # Compute angle of relative rotation
    angle = rel_rot.magnitude()  # In radians
    return np.degrees(angle)  # Convert to degrees

def gripper_state_error(gripper_pred, gripper_gt):
    """
    Computes a binary error for the gripper state.
    Returns 0 if the predicted gripper state matches the ground truth, else 1.
    """
    pred_state = gripper_pred > 0.5  # Assuming gripper open state is represented by values > 0.5
    gt_state = gripper_gt > 0.5
    return int(pred_state == gt_state)

if __name__ == '__main__':
    main()
