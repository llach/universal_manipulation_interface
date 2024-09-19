import os
import pathlib
import numpy as np
import json
import scipy.spatial.transform as st

def get_closest_idx(stamp, stamps):
    return np.argmin(np.abs(stamps - stamp))

def process_data_directory(path):
    required_files = [
        "rgb.mp4",
        "rgb_stamps.json",
        "logged_stamps.json",
        "gripper_poses.json"
    ]
    # Check necessary files
    if not all((path / fn).is_file() for fn in required_files):
        print(f"{path} is missing data")
        return None

    # Read JSON files
    with open(path / "rgb_stamps.json", "r") as f:
        rgb_stamps = np.array(json.load(f))

    with open(path / "logged_stamps.json", "r") as f:
        logged_stamps = json.load(f)

    with open(path / "gripper_poses.json", "r") as f:
        gripper_poses = json.load(f)

    # Process timestamps
    if not logged_stamps:
        print(f"{path} doesn't have enough stamps")
        return None

    # Extract events timestamps
    grasping_end = next((ls[0] for ls in logged_stamps if ls[1] == "placing"), None)
    gripper_close = next((ls[0] for ls in logged_stamps if ls[1] == "gripper"), None)

    if grasping_end is None or gripper_close is None:
        print(f"{path} is missing required events")
        return None

    # Cut off placing frames
    rgb_stamps = rgb_stamps[rgb_stamps < grasping_end]
    if not rgb_stamps.size or rgb_stamps[-1] <= gripper_close:
        print(f"{path} doesn't have enough frames after gripper close")
        return None

    gripper_stamps = np.array([g[0] for g in gripper_poses])
    gripper_data = [gripper_poses[get_closest_idx(t_rgb, gripper_stamps)] for t_rgb in rgb_stamps]

    robot_name = 'robot0'
    episode_data = dict()

    eef_pos = np.array([g[2] for g in gripper_data])
    eef_rot = np.array([g[3] for g in gripper_data])

    episode_data[f'{robot_name}_eef_pos'] = eef_pos.astype(np.float32)
    episode_data[f'{robot_name}_eef_rot_axis_angle'] = st.Rotation.from_quat(eef_rot).as_rotvec().astype(np.float32)
    gripper_open = (np.array([g[0] for g in gripper_data]) < gripper_close).astype(np.uint8)
    episode_data[f'{robot_name}_gripper_open'] = np.expand_dims(gripper_open, axis=-1)

    episode_info = {
        'episode_data': episode_data,
        'video_path': path / "rgb.mp4",
        'frame_start': 0,
        'frame_end': len(rgb_stamps),
        'rgb_stamps': rgb_stamps
    }

    return episode_info

def process_data_directories(data_dirs):
    episodes_info = []
    for dir_path in data_dirs:
        path = pathlib.Path(dir_path)
        episode_info = process_data_directory(path)
        if episode_info is not None:
            episodes_info.append(episode_info)
    return episodes_info
