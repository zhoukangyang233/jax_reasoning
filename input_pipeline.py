# credit: https://github.com/sapientinc/HRM
import pydantic
import numpy as np

from typing import List, Optional

import os
import json

import numpy as np
import pydantic
import random
from functools import partial

import jax
import jax.numpy as jnp
import torch
from torch.utils.data import Dataset, get_worker_info
from utils.logging_util import log_for_0, log_for_all

IGNORE_LABEL_ID = -100 # TODO{zhh}: check how this is used?

class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    
    total_groups: int
    mean_puzzle_examples: float

    sets: List[str]

class PuzzleDataset(Dataset):
    FIELD_MMAP_MODES = {
        "inputs": "r",
        "labels": "r",
        # Keep indices in memory
        "puzzle_identifiers": None,
        "puzzle_indices": None,
        "group_indices": None
    }

    def __init__(self, config, split: str):
        self._data = {}
        self.config = config
        self.split = split
        self.metadata: PuzzleDatasetMetadata = self._load_metadata()
        
        assert len(self.metadata.sets) == 1, "Currently only single set supported."
        log_for_0(f'Loading dataset...')
        for set_name in self.metadata.sets:
            # Load subset into memory
            self._data = {
                field_name: np.load(os.path.join(config.dataset_path, split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in self.FIELD_MMAP_MODES.items()
            }
        log_for_0(f'Dataset loaded.')

        if len(self) != self.metadata.total_groups:
            log_for_0(f"\033[31mWARNING: Dataset size {len(self)} does not match metadata {self.metadata.total_groups}.\033[0m")
            
    def __getitem__(self, index):
        t = self._data['inputs'][index], self._data['labels'][index], np.array(self._data['puzzle_identifiers'][index])
        return tuple(torch.from_numpy(x.astype('int32')) for x in t)

    def __len__(self):
        return len(self._data['inputs'])
    
    def __str__(self):
        md = '\n\t\t'.join([f"{k}: {v}" for k, v in self.metadata.model_dump().items()])
        return f"{self.__class__.__name__}\n\t- split: {self.split}\n\t- size: {len(self)}\n\t- metadata:\n\t\t{md}"

class SudokuDataset(PuzzleDataset):
    def _load_metadata(self):
        return PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,

            vocab_size=11,
            seq_len=81,
            num_puzzle_identifiers=1,

            total_groups={
                'train': 1000,
                'test': 1000,
            }[self.split],
            mean_puzzle_examples=1.0,

            sets=["all"]
        )

class SudokuFullDataset(PuzzleDataset):
    def _load_metadata(self):
        return PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,

            vocab_size=11,
            seq_len=81,
            num_puzzle_identifiers=1,

            total_groups={
                'train': 3831994,
                'test': 422786,
            }[self.split],
            mean_puzzle_examples=1.0,

            sets=["all"]
        )

class TestFolderDataset(PuzzleDataset):
    FIELD_MMAP_MODES = {
        "inputs": "r",
        # "labels": "r", # NO labels
        # Keep indices in memory
        "puzzle_identifiers": None,
        
        # NOTE{zhh}: these two are currently not used
        # "puzzle_indices": None,
        # "group_indices": None
    }
    
    def __init__(self, root):
        config = lambda: None
        config.dataset_path = root
        self.root = root
        super().__init__(config, split='')
        self._data['labels'] = np.zeros_like(self._data['inputs']) # dummy labels

    def _load_metadata(self):
        with open(os.path.join(self.root, "dataset.json"), 'r') as f:
            metadata = json.load(f)
        return PuzzleDatasetMetadata(**metadata)

DATASET_CONFIG_TO_CLS = {
    "sudoku": SudokuDataset,
}

# def merge_epochs(dataset, num_merge_epochs=-1):
#     if num_merge_epochs <= 1:
#         return dataset
#     else:
#         class MergedDataset(Dataset):
#             def __init__(self, dataset, num_merge_epochs):
#                 self.dataset = dataset
#                 self.num_merge_epochs = num_merge_epochs

#             def __len__(self):
#                 return len(self.dataset) // self.num_merge_epochs

#             def __getitem__(self, idx):
#                 # Get the merged item
#                 return self.dataset[idx * self.num_merge_epochs:(idx + 1) * self.num_merge_epochs]
        
#         return MergedDataset(dataset, num_merge_epochs)

def create_split(
    dataset_cfg,
    batch_size,
    split,
    # num_merge_epochs=-1
):
    """Creates a split from the ImageNet dataset using Torchvision Datasets.

    Args:
      dataset_cfg: Configurations for the dataset.
      batch_size: Batch size for the dataloader.
      split: 'train' or 'val'.
    Returns:
      it: A PyTorch Dataloader.
      steps_per_epoch: Number of steps to loop through the DataLoader.
    """
    rank = jax.process_index()
    dataset_cls = DATASET_CONFIG_TO_CLS.get(dataset_cfg.dataset_cls, lambda *args, **kwargs: exec('raise ValueError(f"Unknown dataset class {dataset_cfg.dataset_cls}.")'))
    if split == 'train':
        ds = dataset_cls(config=dataset_cfg, split=split)
        log_for_0(f'\n{ds}')
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=jax.process_count(),
            rank=rank,
            shuffle=True,
        )
        it = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            drop_last=False,
            worker_init_fn=partial(worker_init_fn, rank=rank),
            sampler=sampler,
            num_workers=dataset_cfg.num_workers,
            prefetch_factor=(
                dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
            ),
            pin_memory=dataset_cfg.pin_memory,
            persistent_workers=True if dataset_cfg.num_workers > 0 else False,
        )
        steps_per_epoch = len(it)
    elif split == 'test':
        ds = dataset_cls(config=dataset_cfg, split=split)
        log_for_0(ds)
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=jax.process_count(),
            rank=rank,
            shuffle=False,  # don't shuffle for test
        )
        it = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            drop_last=False,  # don't drop for test
            worker_init_fn=partial(worker_init_fn, rank=rank),
            sampler=sampler,
            num_workers=dataset_cfg.num_workers,
            prefetch_factor=(
                dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
            ),
            pin_memory=dataset_cfg.pin_memory,
            persistent_workers=True if dataset_cfg.num_workers > 0 else False,
        )
        steps_per_epoch = len(it)
    else:
        raise ValueError(f"Unknown split {split}.")
    log_for_all(f'Dataset is loaded')
    return it, steps_per_epoch, ds.metadata


def create_split_from_folder(
    root,
    batch_size,
):
    rank = jax.process_index()
    ds = TestFolderDataset(root=root)
    log_for_0(ds)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=jax.process_count(),
        rank=rank,
        shuffle=False,  # don't shuffle for test
    )
    it = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=False,  # don't drop for test
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )
    steps_per_epoch = len(it)
    log_for_all(f'Dataset is loaded')
    return it, steps_per_epoch, ds.metadata

#### DataLoader ####
def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def prepare_batch_data(batch, batch_size=None, dataset_metdata=None):
    # image, label = batch
    inputs, labels, puzzle_identifiers = batch
    assert inputs.shape[0] == labels.shape[0] == puzzle_identifiers.shape[0]
    
    metadata = dataset_metdata
    if metadata.ignore_label_id is not None:
        labels[labels == metadata.ignore_label_id] = IGNORE_LABEL_ID
    
    batch = {"inputs": inputs, "labels": labels, "puzzle_identifiers": puzzle_identifiers, 'zhh_is_pad': labels * 0}
    # pad the batch if smaller than batch_size
    if batch_size is not None and batch_size > puzzle_identifiers.shape[0]:
        pad_values = {
            # "inputs": metadata.pad_id,
            "labels": IGNORE_LABEL_ID,
            "puzzle_identifiers": metadata.blank_identifier_id,
            'zhh_is_pad': 1,
        }
        pad_size = batch_size - puzzle_identifiers.shape[0]
        inputs = batch["inputs"]
        batch = {k: torch.cat([v, torch.full((pad_size, ) + v.shape[1:], pad_values[k], dtype=v.dtype)], dim=0) for k, v in batch.items() if k in pad_values}
        
        # NOTE{zhh}: this is a hack, pad inputs with last input, avoid bad inputs influence halt time
        assert inputs.ndim == 2, f"inputs should be 2D, got {inputs.shape}"
        batch["inputs"] = torch.cat([inputs, inputs[-1:].repeat(pad_size, 1)], dim=0)

    LDC = jax.local_device_count()
    return {k: (v.reshape((LDC, -1) + v.shape[1:])).numpy() for k, v in batch.items()}

##### Augmentations #####

# Global list mapping each dihedral transform id to its inverse.
# Index corresponds to the original tid, and the value is its inverse.
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr
    
def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])

if __name__ == "__main__":
    config = lambda: None
    config.dataset_path = "/kmh-nfs-ssd-us-mount/data/sudoku-extreme-full"
    # Test
    ds = SudokuDataset(config, "train")
    print(ds)
    for i in range(3):
        inp, out, pid = ds[i]
        print(inp.shape, out.shape, pid.shape)
        print(inp)
        print(out)
        print(pid)