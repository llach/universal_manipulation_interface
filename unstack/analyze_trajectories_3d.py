import os
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

in_path = f"{os.environ['HOME']}/unstack_data/"

import numpy as np

def interpolate_gripper_positions(rgb_stamps, gripper_stamps, gripper_positions):
    interpolated_positions = []

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

        # Get the timestamps and positions before and after
        time_before = gripper_stamps[before_index]
        time_after = gripper_stamps[after_index]
        print(before_index, after_index)
        position_before = gripper_positions[before_index]
        position_after = gripper_positions[after_index]

        # Perform linear interpolation
        if time_after == time_before:  # Avoid division by zero
            interpolated_position = position_before
        else:
            interpolated_position = position_before + (
                (rgb_time - time_before) / (time_after - time_before)
            ) * (position_after - position_before)

        interpolated_positions.append(interpolated_position)

    return np.array(interpolated_positions)


# shapes: Nepisodes x Ntimesteps of episode x ...
timestamps  = [] # ... 1
positions   = [] # ... 3
quaternions = [] # ... 4
rgb_timestamps = [] # ... 1

# index of logged timestamps: [gripper close, grasping end, gripper open]
stamps = []

for i, ipath in enumerate(os.listdir(in_path)):
        path = pathlib.Path(os.path.join(in_path, ipath)).absolute() 
        if path.is_file(): continue
        print(path)

        gripper_poses = path.joinpath("gripper_poses.json")
        if not gripper_poses.is_file():
            print(f"[{i}]: no gripper poses found!")
            continue

        logged_stamps = path.joinpath("logged_stamps.json")
        if not logged_stamps.is_file():
            print(f"[{i}]: no logged stamps found!")
            continue

        rgb_stamps_path = path.joinpath("rgb_stamps.json")
        if not rgb_stamps_path.is_file():
            print(f"[{i}]: no rgb stamps found!")
            continue

        with open(rgb_stamps_path, "r") as f:
            rgb_timestamps.append(json.load(f))

        with open(logged_stamps, "r") as f:
            stps = json.load(f)
            if stps == []:
                print(f"[{i}]: no logged stamps found!")
                continue

            gripper_close = stps[0][0]
            grasping_end = stps[1][0]
            gripper_open = stps[2][0]
        
        with open(gripper_poses, "r") as f:
            raw = json.load(f)

            ts = np.array([d[0] for d in raw])
            pos = np.array([d[2] for d in raw])
            quats = np.array([d[3] for d in raw])

            interpolated_positions = interpolate_gripper_positions(rgb_timestamps[i], ts, pos)

            # get index of timestamp
            gripper_close = np.argmax(ts > gripper_close)
            grasping_end = np.argmax(ts > grasping_end)
            gripper_open = np.argmax(ts > gripper_open)

            # time relative to start
            ts = ts - ts[0] 

            

            # TODO slice to discard placing traj

        if np.sum(pos) == 0:
            print(f"[{i}]: gripper poses invalid!")
            continue

        timestamps.append(ts)
        positions.append(pos)
        quaternions.append(quats)
        stamps.append([
            gripper_close,
            grasping_end,
            gripper_open
        ])

        # only analyze I valid trajectories 
        if len(timestamps) == 4: break

Tmax = np.max([np.max(t) for t in timestamps]) # longest trajectory

# iterate over episodes
for i, (ts, pos, quats, stps) in enumerate(zip(timestamps, positions, quaternions, stamps)):
    tmax = np.max(ts) # duration of trajectory i
    t_norm = ts/tmax

    # Create a 3D plot
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111, projection='3d')


    # Create line segments for 3D
    segments = np.array([pos[i:i+2] for i in range(len(pos)-1)])

    # Normalize the values for color mapping
    norm = Normalize(vmin=0, vmax=1)

    # Plot each segment with the corresponding color
    for i, (start, end) in enumerate(segments):
        color = plt.cm.viridis(norm(t_norm[i]))  # Use the colormap
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, linewidth=2)

    # Scatter the points specified in stps in different colors with labels for the legend
    scatter_colors = ['blue', 'green', 'orange']  # Colors for the scatter points
    labels = ['close', 'end', 'open']  # Labels for the legend
    for idx, color, label in zip(stps, scatter_colors, labels):
        ax.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2], color=color, s=50, label=label+ f" [{idx}] ")

    # Add a legend
    ax.legend()

    # Adjust the axes limits
    ax.set_xlim(pos[:, 0].min(), pos[:, 0].max())
    ax.set_ylim(pos[:, 1].min(), pos[:, 1].max())
    ax.set_zlim(pos[:, 2].min(), pos[:, 2].max())

    # Add a colorbar
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array(t_norm)

    plt.colorbar(mappable, ax=ax, shrink=0.5)

    fig.tight_layout()
    plt.show()
    break