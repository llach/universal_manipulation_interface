import os
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer

dataset_path = "cup_in_the_wild.zarr.zip"
zarr.open_group(dataset_path, "r+w")

# with zarr.ZipStore(dataset_path, mode='r') as zip_store:
#     zarr.open_group()
    # replay_buffer = ReplayBuffer.copy_from_store(
    #     src_store=zip_store, 
    #     store=zarr.MemoryStore()
    # )