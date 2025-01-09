import os
import json
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
from unstack.data_processing import process_data_directories


def mse(input, target):
    return ((input-target)**2).mean()

@click.command()
@click.argument('episode_dirs', nargs=-1)
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint (.ckpt file)')
@click.option('--output', '-o', default=None, help='Output file to save errors (optional)')
@click.option('--video_output', '-vo', default='output_video.mp4', help='Output video file path')
@click.option('--display', is_flag=True, help='Display frames live during processing')
def main(episode_dirs, checkpoint, output, video_output, display):
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
    policy.num_inference_steps = 2  # Adjust as needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)

    # Process data directories to get episodes_info
    episode_infos = process_data_directories(episode_dirs)
    if not episode_infos:
        print("No valid episodes found.")
        return

    shape_meta = cfg.task.shape_meta

    actions_gt, actions_pred = [], []
    pos_errors, rot_errors, gripper_errors = [], [], []

    # Initialize video writer
    video_writer = None

    # Iterate over each episode
    for episode_idx, episode_info in enumerate(tqdm(episode_infos, desc='Episodes')):
        episode_data = episode_info['episode_data']
        num_steps = episode_data['robot0_eef_pos'].shape[0]
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
        start_pos = episode_data['robot0_eef_pos'][0]
        start_rotvec = episode_data['robot0_eef_rot_axis_angle'][0]
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
                episode_data, frames, step_idx, shape_meta, start_pose_mat
            )
            # Convert observations to tensors, add batch dimensions and move to device
            obs_dict = dict_apply(
                obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
            )

            # Run the policy to get action prediction
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
                action_pred = result['action_pred'][0].detach().cpu().numpy()[0] # First action step
            action_gt = get_gt_action(episode_data, step_idx)

            pos_mse = mse(action_pred[:3], action_gt[:3])
            rot_mse = mse(action_pred[3:9], action_gt[3:9])
            gripper_mse = mse(action_pred[9], action_gt[9])

            actions_gt.append(action_gt)
            actions_pred.append(action_pred)

            # Compute errors
            pos_error = np.sqrt(pos_mse)
            rot_error = rotation_error(action_pred[3:9], action_gt[3:9])
            gripper_error = gripper_mse

            pos_errors.append(pos_error)
            rot_errors.append(rot_error)
            gripper_errors.append(gripper_error)

            # Print errors for the current frame
            print(f'Episode {episode_idx}, Step {step_idx}, Pos: {pos_error:.4f}m, Rot: {rot_error:.4f}°, Gripper: {gripper_error} | P {pos_mse:.4f} R {rot_mse:.4f} G {gripper_mse:.4f}')

            # Visualize and save frame with overlay
            frame = frames[step_idx].copy()
            # Overlay text showing the prediction errors
            line1 = f'Episode: {episode_idx}, Step: {step_idx}'
            line2 = f'Pos Error: {pos_error:.4f}m'
            line3 = f'Rot Error: {rot_error:.4f}°'
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
    pos_errors, rot_errors, gripper_errors = np.array(pos_errors), np.array(rot_errors), np.array(gripper_errors)

    with open("actions.json", "w", encoding="utf8") as f:
        json.dump(np.array([actions_gt, actions_pred], dtype=np.float32).tolist(), f)

    # Release the video writer
    if video_writer is not None:
        video_writer.release()
        print(f'Video saved to {video_output}')

    # Destroy all OpenCV windows if display was enabled
    if display:
        cv2.destroyAllWindows()
    
    print(f'Average positional error: {pos_errors.mean():.4f}±{pos_errors.std():.4f}')
    print(f'Average rotational error: {rot_errors.mean():.4f}±{rot_errors.std():.4f}')
    print(f'Average gripper error {gripper_errors.mean():.4f}±{gripper_errors.std():.4f}')

def construct_observation_dict(episode_data, frames, step_idx, shape_meta, start_pose_mat):
    """
    Constructs the observation dictionary from the episode data at the given step index.
    """
    obs_dict = {}
    for key in shape_meta.obs:
        meta = shape_meta.obs[key]
        horizon = meta.horizon

        # Handle horizons greater than 1
        start_idx = step_idx
        end_idx = step_idx + horizon
        if "rgb" in key:
            # move channel last to channel first
            # T,H,W,C -> T,C,H,W
            # convert uint8 image to float32
            imgs = np.moveaxis(frames[start_idx:end_idx], -1, 1).astype(np.float32) / 255.

            # solve padding
            if imgs.shape[0] < horizon:
                padding = np.repeat(imgs[:1], horizon - imgs.shape[0], axis=0)
                imgs = np.concatenate([padding, imgs], axis=0)

            obs_dict[key] = imgs
            # obs_dict[key] = np.random.uniform(0,1,imgs.shape) # test with random noise as images
        elif key in episode_data:
            data = episode_data[key][start_idx:end_idx]
            # Pad if necessary
            if data.shape[0] < horizon:
                padding = np.repeat(data[:1], horizon - data.shape[0], axis=0)
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
        elif key == 'robot0_eef_pos_wrt_start' or key == 'robot0_eef_rot_axis_angle_wrt_start':
            curr_pos = episode_data['robot0_eef_pos'][start_idx:end_idx]
            curr_rotvec = episode_data['robot0_eef_rot_axis_angle'][start_idx:end_idx]

            curr_pose = np.concatenate([curr_pos, curr_rotvec], axis=-1)
            curr_pose_mat = pose_to_mat(curr_pose)

            rel_pose_mat = convert_pose_mat_rep(
                curr_pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False
            )
            rel_pose = mat_to_pose(rel_pose_mat)

            # Pad if necessary
            if rel_pose.shape[0] < horizon:
                padding = np.repeat(rel_pose[:1], horizon - rel_pose.shape[0], axis=0)
                rel_pose = np.concatenate([padding, rel_pose], axis=0)

            if key == 'robot0_eef_pos_wrt_start':
                obs_dict[key] = rel_pose[..., :3]
            else:  # key == 'robot0_eef_rot_axis_angle_wrt_start'
                # Convert rotation to 6D representation
                rotvec = rel_pose[..., 3:]
                rot_mats = st.Rotation.from_rotvec(rotvec).as_matrix()
                rot6d = mat_to_rot6d(rot_mats)
                obs_dict[key] = rot6d
    return obs_dict

def get_gt_action(episode_data, step_idx):
    """
    Computes the ground truth action from the dataset.
    """
    try:
        pos = episode_data["robot0_eef_pos"][step_idx+1]
        rot_vec = episode_data["robot0_eef_rot_axis_angle"][step_idx+1]
        gripper_open = episode_data["robot0_gripper_open"][step_idx+1]
    except:
        pos = episode_data["robot0_eef_pos"][-1]
        rot_vec = episode_data["robot0_eef_rot_axis_angle"][-1]
        gripper_open = episode_data["robot0_gripper_open"][-1]

    return np.concatenate([pos, mat_to_rot6d(st.Rotation.from_rotvec(rot_vec).as_matrix()), gripper_open], axis=0)

def rotation_error(rotvec_pred, rotvec_gt):
    """
    Computes the angular difference between two rotations represented as rotation vectors.
    Returns the error in degrees.
    """
    rot_pred = st.Rotation.from_matrix(rot6d_to_mat(rotvec_pred))
    rot_gt = st.Rotation.from_matrix(rot6d_to_mat(rotvec_gt))
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
