import os
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from helper import interpolate_gripper_positions

in_path = f"{os.environ['HOME']}/repos/unstack_cloud/"


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
        print("Processing ", path)

        gripper_poses = path.joinpath("gripper_poses.json")
        if not gripper_poses.is_file():
            print(f"[{i}]: no gripper poses found!")
            continue

        misc_data = path.joinpath("misc.json")
        if not misc_data.is_file():
            print(f"[{i}]: no misc data found!")
            continue

        rgb_stamps_path = path.joinpath("rgb_stamps.json")
        if not rgb_stamps_path.is_file():
            print(f"[{i}]: no rgb stamps found!")
            continue

        with open(rgb_stamps_path, "r") as f:
            rgb_timestamps.append(json.load(f))

        with open(misc_data, "r") as f:
            misc = json.load(f)
            gripper_close = misc["gripper_close_time"]
        
        with open(gripper_poses, "r") as f:
            raw = json.load(f)

            ts = np.array([d[0] for d in raw])
            pos = np.array([d[2] for d in raw])
            quats = np.array([d[3] for d in raw])

            pos -= pos[0,:]

            interpolated_positions = interpolate_gripper_positions(rgb_timestamps[-1], ts, pos)

            # get index of timestamp
            gripper_close = np.argmax(ts > gripper_close)

            # time relative to start
            ts = ts - ts[0] 


        if np.sum(pos) == 0:
            print(f"[{i}]: gripper poses invalid!")
            continue

        timestamps.append(ts)
        positions.append(pos)
        quaternions.append(quats)
        stamps.append([
            gripper_close
        ])

        # only analyze I valid trajectories 
        # if len(timestamps) == 10: break

Tmax = np.max([np.max(t) for t in timestamps]) # longest trajectory

# Create a 3D plot
fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111, projection='3d')

# iterate over episodes
for i, (ts, pos, quats, stps) in enumerate(zip(timestamps, positions, quaternions, stamps)):
    tmax = np.max(ts) # duration of trajectory i
    t_norm = ts/tmax

    # Create line segments for 3D
    segments = np.array([pos[i:i+2] for i in range(len(pos)-1)])

    # Normalize the values for color mapping
    norm = Normalize(vmin=0, vmax=1)

    # Plot each segment with the corresponding color
    # for i, (start, end) in enumerate(segments):
    #     color = plt.cm.viridis(norm(t_norm[i]))  # Use the colormap
    #     ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, linewidth=2)

    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=[plt.cm.viridis(norm(t_norm[i])) for i in range(len(t_norm))] )

    # Scatter the points specified in stps in different colors with labels for the legend
    scatter_colors = ['blue']#, 'green', 'orange']  # Colors for the scatter points
    labels = ['close']#, 'end', 'open']  # Labels for the legend
    for idx, color, label in zip(stps, scatter_colors, labels):
        ax.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2], color=color, s=50, label=label+ f" [{idx}] ")

    # Add a legend
    if i == 0: ax.legend()

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