# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()


datefmt = "%Y.%m.%d_%H_%M_%S"

def get_closest_idx(stamp, stamps):
    return np.argmin(np.abs(stamps-stamp))

# %%
@click.command()
@click.argument('in_path', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,299')
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-n', '--num_workers', type=int, default=None)
def main(in_path, output, out_res, compression_level, num_workers):
    in_path = in_path[0]

    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
        
    out_res = tuple(int(x) for x in out_res.split(','))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # dump lowdim data to replay buffer
    # generate argumnet for videos
    n_cameras = 1
    buffer_start = 0
    all_videos = set()
    vid_args = list()

    for ipath in os.listdir(in_path):
        path = pathlib.Path(os.path.join(in_path, ipath)).absolute() 
        print(path)
        if path.is_file(): continue

        with open(path.joinpath("rgb_stamps.json"), "r") as f:
            rgb_stamps = np.array(json.load(f))

        with open(path.joinpath("logged_stamps.json"), "r") as f:
            logged_stamps = json.load(f)

        with open(path.joinpath("gripper_poses.json"), "r") as f:
            gripper_poses = json.load(f)

        # select timestamp where placing begins and where gripper closes
        grasping_end = [ls[0] for ls in logged_stamps if ls[1] == "placing"][0]
        gripper_close = [ls[0] for ls in logged_stamps if ls[1] == "gripper"][0]

        # cut off placing frames
        rgb_stamps = rgb_stamps[rgb_stamps<grasping_end]
        assert rgb_stamps[-1] > gripper_close, "rgb_stamps[-1] > gripper_close" # gripper has to be closed before episode end

        gripper_stamps = np.array([g[0] for g in gripper_poses])
        # for g in gripper_stamps: print(g) # TODO make sure data is fresh
        gripper_data = [gripper_poses[get_closest_idx(t_rgb, gripper_stamps)] for t_rgb in rgb_stamps]

        # for r, g in zip(rgb_stamps, [g[0] for g in gripper_data]):
            # print(r,g)

        robot_name = 'robot0'
        episode_data = dict()
        
        eef_pos = np.array([g[2] for g in gripper_data])
        eef_rot = np.array([g[3] for g in gripper_data])

        episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
        episode_data[robot_name + '_eef_rot_axis_angle'] = st.Rotation.from_quat(eef_rot).as_rotvec().astype(np.float32)
        episode_data[robot_name + '_gripper_open'] = np.expand_dims(np.array([g[0] for g in gripper_data])<gripper_close, axis=-1).astype(np.uint8)

        out_replay_buffer.add_episode(data=episode_data, compressors=None)
        
        videos_dict = defaultdict(list)
            
        # aggregate video gen aguments
        n_frames = None
        video_path = path.joinpath("rgb.mp4")
        assert video_path.is_file()
        
        n_frames = len(rgb_stamps)
        
        videos_dict[str(video_path)].append({
            'camera_idx': 0,
            'frame_start': 0,
            'frame_end': n_frames,
            'buffer_start': buffer_start
        })
        buffer_start += n_frames
        
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")

    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
    # dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    def video_to_zarr(replay_buffer, mp4_path, tasks):
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        curr_task_idx = 0
        
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')
                        
                    # compress image
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False
                    
    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")
    print(f"n_steps {out_replay_buffer.n_steps} | n_episodes {out_replay_buffer.n_episodes}")

# %%
if __name__ == "__main__":
    main()
