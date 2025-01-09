import os
import pathlib
import numpy as np
import json
import scipy.spatial.transform as st
import numpy as np
from scipy.spatial.transform import Slerp, Rotation


def interpolate_gripper_positions_and_rotations(
    rgb_stamps, gripper_stamps, gripper_positions, gripper_rotations
):
    interpolated_positions = []
    interpolated_rotations = []

    # Convert gripper_rotations (quaternions) to Rotation objects
    gripper_rotations_obj = Rotation.from_quat(gripper_rotations)

    for rgb_time in rgb_stamps:
        # Find the index of the gripper timestamp just before the rgb_time
        before_index = np.where(gripper_stamps < rgb_time)[0]
        if len(before_index) == 0:
            before_index = 0
        else:
            before_index = before_index[-1]

        # Find the index of the gripper timestamp just after the rgb_time
        after_index = np.where(gripper_stamps > rgb_time)[0]
        if len(after_index) == 0:
            after_index = len(gripper_stamps) - 1
        else:
            after_index = after_index[0]

        # Get the timestamps, positions, and rotations before and after
        time_before = gripper_stamps[before_index]
        time_after = gripper_stamps[after_index]
        position_before = gripper_positions[before_index]
        position_after = gripper_positions[after_index]
        rotation_before = gripper_rotations_obj[before_index]
        rotation_after = gripper_rotations_obj[after_index]

        # Perform linear interpolation for positions
        if time_after == time_before:  # Avoid division by zero
            interpolated_position = position_before
        else:
            interpolated_position = position_before + (
                (rgb_time - time_before) / (time_after - time_before)
            ) * (position_after - position_before)

        # Perform spherical linear interpolation (slerp) for rotations
        if time_after == time_before:
            interpolated_rotation = rotation_before
        else:
            slerp = Slerp([time_before, time_after], Rotation.from_quat([
                gripper_rotations[before_index], gripper_rotations[after_index]
            ]))
            interpolated_rotation = slerp([rgb_time])[0]

        interpolated_positions.append(interpolated_position)
        interpolated_rotations.append(interpolated_rotation.as_quat())  # Convert back to quaternion

    return np.array(interpolated_positions), np.array(interpolated_rotations)


def get_closest_idx(stamp, stamps):
    return np.argmin(np.abs(stamps - stamp))

def get_episode_info(path):
    required_files = [
        "rgb.mp4",
        "depth.mp4",
        "misc.json",
        "rgb_stamps.json",
        "gripper_poses.json"
    ]

    # check whether all necessary files exist
    if (not np.all([path.joinpath(fn).is_file() for fn in required_files])):
        print(path, "is missing data")
        exit(1)

    with open(path.joinpath("rgb_stamps.json"), "r") as f:
        rgb_stamps = json.load(f)

    with open(path.joinpath("misc.json"), "r") as f:
        misc = json.load(f)
        gripper_close_time = misc["gripper_close_time"]
    
    with open(path.joinpath("gripper_poses.json"), "r") as f:
        raw = json.load(f)

        ts = np.array([d[0] for d in raw])
        eef_pos = np.array([d[2] for d in raw])
        eef_rot = np.array([d[3] for d in raw])

        #TODO rot is not relative to start
        eef_pos -= eef_pos[0,:]

        eef_pos, eef_rot = interpolate_gripper_positions_and_rotations(rgb_stamps, ts, eef_pos, eef_rot)

    robot_name = 'robot0'
    episode_data = dict()
    
    episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
    episode_data[robot_name + '_eef_rot_axis_angle'] = st.Rotation.from_quat(eef_rot).as_rotvec().astype(np.float32)
    episode_data[robot_name + '_gripper_open'] = np.expand_dims(np.array(rgb_stamps)<gripper_close_time, axis=-1).astype(np.uint8)

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
        episode_info = get_episode_info(path)
        if episode_info is not None:
            episodes_info.append(episode_info)
    return episodes_info
