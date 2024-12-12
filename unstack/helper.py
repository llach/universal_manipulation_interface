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
