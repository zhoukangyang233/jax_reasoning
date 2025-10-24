# credit: https://github.com/sapientinc/HRM
from typing import List, Optional

import copy
import itertools
import json
import os
import random
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import torch

from torch.utils.data import Dataset
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
    augmentations_per_puzzle: Optional[int] = 0

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
        self.augmentations_per_puzzle = int(getattr(config, "augmentations_per_puzzle", 0))
        self.metadata: PuzzleDatasetMetadata = self._load_metadata()
        # Ensure metadata.augmentations_per_puzzle is set correctly
        self.metadata.augmentations_per_puzzle = self.augmentations_per_puzzle
        
        assert len(self.metadata.sets) == 1, "Currently only single set supported."
        log_for_0(f'Loading dataset...')
        for set_name in self.metadata.sets:
            # Load subset into memory
            self._data = {
                field_name: np.load(os.path.join(config.dataset_path, split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in self.FIELD_MMAP_MODES.items()
            }
        log_for_0(f'Dataset loaded.')

        if self.augmentations_per_puzzle < 0:
            raise ValueError(f"augmentations_per_puzzle must be >= 0, got {self.augmentations_per_puzzle}")
        self._base_len = len(self._data['inputs'])
        self._length = self._base_len * (self.augmentations_per_puzzle + 1)

        if self.augmentations_per_puzzle:
            self.metadata.total_groups = self._length
            log_for_0(
                f"Applying {self.augmentations_per_puzzle} augmentations per puzzle; effective dataset size {self._length}."
            )
        elif self._length != self.metadata.total_groups:
            log_for_0(
                f"\033[31mWARNING: Dataset size {self._length} does not match metadata {self.metadata.total_groups}.\033[0m"
            )
            
    def __getitem__(self, index):
        base_index = index % self._base_len
        augmentation_index = index // self._base_len

        inputs = np.array(self._data['inputs'][base_index], copy=True)
        labels = np.array(self._data['labels'][base_index], copy=True)
        puzzle_identifier = np.array(self._data['puzzle_identifiers'][base_index])

        aug_slot = -1
        if augmentation_index > 0 and self.augmentations_per_puzzle:
            inputs, labels = self._augment_example(
                inputs,
                labels,
                augmentation_index - 1,
                base_index,
            )
            aug_slot = augmentation_index - 1
        elif self.augmentations_per_puzzle > 0:
            aug_slot = -1

        tensors = (
            torch.from_numpy(inputs.astype(np.int32, copy=False)),
            torch.from_numpy(labels.astype(np.int32, copy=False)),
            torch.from_numpy(np.asarray(puzzle_identifier, dtype=np.int32)),
            torch.tensor(base_index, dtype=torch.int32),
            torch.tensor(aug_slot, dtype=torch.int32),
        )
        return tensors

    def __len__(self):
        return self._length

    def _augment_example(self, inputs, labels, aug_index, base_index):
        return inputs, labels

    def _next_augmentation_rng(self, base_index: int, aug_index: int):
        if self.augmentations_per_puzzle <= 0:
            raise ValueError("No augmentations configured for this dataset instance.")

        base_index = int(base_index)
        aug_index = int(aug_index)
        base_seed = (base_index + 1) * 1000 + aug_index
        seed = base_seed + 11114514
        return np.random.default_rng(seed)
    
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

    def _augment_example(self, inputs, labels, aug_index, base_index):
        rng = self._next_augmentation_rng(base_index, aug_index)
        if aug_index == 0 and base_index == 0:
            log_for_all(f"Augmenting example with random number: {aug_index}, {base_index}, {rng.integers(0, 1_000_000_000)}")
            #print("Augmenting example with random number: ", aug_index, base_index, rng.integers(0, 1_000_000_000))
        board_inputs = inputs.reshape(9, 9)
        board_labels = labels.reshape(9, 9)

        transform_id = int(rng.integers(0, 8))
        board_inputs = dihedral_transform(board_inputs, transform_id)
        board_labels = dihedral_transform(board_labels, transform_id)

        row_groups = np.arange(9).reshape(3, 3)
        for band in range(3):
            perm_choice = PERMUTATIONS_3[int(rng.integers(0, 6))]
            row_groups[band] = row_groups[band][perm_choice]
        band_perm_choice = PERMUTATIONS_3[int(rng.integers(0, 6))]
        row_groups = row_groups[band_perm_choice]
        row_indices = row_groups.reshape(-1)

        col_groups = np.arange(9).reshape(3, 3)
        for stack in range(3):
            perm_choice = PERMUTATIONS_3[int(rng.integers(0, 6))]
            col_groups[stack] = col_groups[stack][perm_choice]
        stack_perm_choice = PERMUTATIONS_3[int(rng.integers(0, 6))]
        col_groups = col_groups[stack_perm_choice]
        col_indices = col_groups.reshape(-1)

        board_inputs = board_inputs[row_indices][:, col_indices]
        board_labels = board_labels[row_indices][:, col_indices]

        vocab_size = self.metadata.vocab_size
        mapping = np.arange(vocab_size, dtype=board_labels.dtype)
        digit_tokens = np.unique(board_labels)
        if self.metadata.pad_id is not None:
            digit_tokens = digit_tokens[digit_tokens != self.metadata.pad_id]
        if (
            self.metadata.ignore_label_id is not None
            and self.metadata.ignore_label_id >= 0
            and self.metadata.ignore_label_id < vocab_size
        ):
            digit_tokens = digit_tokens[digit_tokens != self.metadata.ignore_label_id]
        if digit_tokens.size:
            mapping[digit_tokens] = rng.permutation(digit_tokens)

        board_inputs = mapping[board_inputs]
        board_labels = mapping[board_labels]

        return board_inputs.reshape(-1), board_labels.reshape(-1)
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
    *,
    dataset_overrides=None,
    shuffle=None,
    # num_merge_epochs=-1
):
    """Creates a split from the dataset using Torchvision Datasets.

    Args:
      dataset_cfg: Configurations for the dataset.
      batch_size: Batch size for the dataloader.
      split: 'train' or 'test'.
      dataset_overrides: Optional dict of overrides applied to the dataset config
        before instantiation (e.g. different augmentation settings).
      shuffle: Optional explicit shuffle flag for the DistributedSampler.
    Returns:
      it: A PyTorch Dataloader.
      steps_per_epoch: Number of steps to loop through the DataLoader.
    """
    rank = jax.process_index()
    dataset_cls = DATASET_CONFIG_TO_CLS.get(dataset_cfg.dataset_cls, lambda *args, **kwargs: exec('raise ValueError(f"Unknown dataset class {dataset_cfg.dataset_cls}.")'))
    dataset_cfg_for_split = dataset_cfg
    actual_split = split
    local_overrides = dataset_overrides or {}
    if split == 'test':
        dataset_cfg_for_split = copy.deepcopy(dataset_cfg)
        if hasattr(dataset_cfg_for_split, "augmentations_per_puzzle"):
            dataset_cfg_for_split.augmentations_per_puzzle = 0  # disable augmentation for evaluation
        actual_split = 'test'
    elif split == 'train':
        if local_overrides:
            dataset_cfg_for_split = copy.deepcopy(dataset_cfg)
        actual_split = 'train'
    else:
        raise ValueError(f"Unknown split {split}.")

    if local_overrides:
        for field, value in local_overrides.items():
            setattr(dataset_cfg_for_split, field, value)
        log_for_0(f"Applied dataset overrides for split '{split}': {local_overrides}")

    if shuffle is None:
        shuffle = (split == 'train') and not local_overrides

    if split == 'train':
        ds = dataset_cls(config=dataset_cfg_for_split, split=actual_split)
        log_for_0(f'\n{ds}')
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=jax.process_count(),
            rank=rank,
            shuffle=shuffle,
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
        ds = dataset_cls(config=dataset_cfg_for_split, split=actual_split)
        log_for_0(ds)
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=jax.process_count(),
            rank=rank,
            shuffle=shuffle,  # don't shuffle for test / eval
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
    # print the dataset sizes
    log_for_all(f'Dataset split "{split}" has {len(ds)} examples, {steps_per_epoch} steps per epoch with batch size {batch_size}.')
    return it, steps_per_epoch, ds.metadata, ds


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
    return it, steps_per_epoch, ds.metadata, ds

#### DataLoader ####
def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def prepare_batch_data(batch, batch_size=None, dataset_metdata=None):
    try:
        inputs, labels, puzzle_identifiers, base_indices, augmentation_indices = batch
    except ValueError as exc:
        raise ValueError(
            "Dataset must provide (inputs, labels, puzzle_identifiers, base_indices, augmentation_indices)."
        ) from exc

    assert (
        inputs.shape[0]
        == labels.shape[0]
        == puzzle_identifiers.shape[0]
        == base_indices.shape[0]
        == augmentation_indices.shape[0]
    ), "Mismatched batch dimensions"
    
    metadata = dataset_metdata
    if metadata.ignore_label_id is not None:
        labels[labels == metadata.ignore_label_id] = IGNORE_LABEL_ID
    
    batch = {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers,
        "base_puzzle_indices": base_indices,
        "augmentation_indices": augmentation_indices,
        'zhh_is_pad': labels * 0
    }
    # pad the batch if smaller than batch_size
    if batch_size is not None and batch_size > puzzle_identifiers.shape[0]:
        pad_values = {
            # "inputs": metadata.pad_id,
            "labels": IGNORE_LABEL_ID,
            "puzzle_identifiers": metadata.blank_identifier_id,
            "base_puzzle_indices": -1,
            "augmentation_indices": -1,
            'zhh_is_pad': 1,
        }
        pad_size = batch_size - puzzle_identifiers.shape[0]
        inputs = batch["inputs"]
        # NOTE{zhh}: this is a hack, pad inputs with last input, avoid bad inputs influence halt time
        assert inputs.ndim == 2, f"inputs should be 2D, got {inputs.shape}"
        batch["inputs"] = torch.cat([inputs, inputs[-1:].repeat(pad_size, 1)], dim=0)
        for key, pad_value in pad_values.items():
            value = batch[key]
            pad_tensor = torch.full((pad_size, ) + value.shape[1:], pad_value, dtype=value.dtype)
            batch[key] = torch.cat([value, pad_tensor], dim=0)

    LDC = jax.local_device_count()
    return {k: (v.reshape((LDC, -1) + v.shape[1:])).numpy() for k, v in batch.items()}

##### Augmentations #####

# Global list mapping each dihedral transform id to its inverse.
# Index corresponds to the original tid, and the value is its inverse.
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]
PERMUTATIONS_3 = np.array(list(itertools.permutations(range(3))), dtype=np.int32)

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
    config.augmentations_per_puzzle = 2
    config.dataset_path = "/kmh-nfs-ssd-us-mount/data/sudoku-extreme-1k"
    # Test
    ds = SudokuDataset(config, "train")
    print(ds)
    for t in range(1):
        inp, out, pid, base_idx, aug_idx = ds[t]
        print(inp.shape, out.shape, pid.shape, base_idx.shape, aug_idx.shape)
        for i in range(9):
            for j in range(9):
                print(f"{inp[i*9 + j]}", end=' ')
            print()
        #print(inp)
        #print(out)
        #print(pid)
        #print(base_idx, aug_idx)
    # Test for augmentation
    for t in range(1):
        inp, out, pid, base_idx, aug_idx = ds[1000 + t]
        print("Augmented:")
        for i in range(9):
            for j in range(9):
                print(f"{out[i*9 + j]}", end=' ')
            print()
