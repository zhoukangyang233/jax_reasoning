from datasets import load_dataset

# Download the SakanaAI/Sudoku-Bench dataset from Hugging Face
dataset = load_dataset("SakanaAI/Sudoku-Bench", 'challenge_100')

def filter_fn(example):
    return len(example['initial_board']) == 81

def map_fn(example):
    # Convert list of lists to numpy array
    import numpy as np
    example['inputs'] = np.array(list(map(int, example['initial_board'].replace('.', '0')))) + 1
    example['labels'] = np.array(list(map(int, example['solution']))) + 1
    example['puzzle_identifiers'] = np.array(0)
    assert example['inputs'].shape == (81,), (example['inputs'].shape, example)
    assert example['labels'].shape == (81,), (example['labels'].shape, example)
    assert example['puzzle_identifiers'].shape == (), example['puzzle_identifiers'].shape
    return example

# Example: print the first sample from the train split
# print(dataset["test"][0])

# merge all data into a np array
dataset = dataset.filter(filter_fn)
print('after filter dataset size', len(dataset['test']))
dataset = dataset.map(map_fn)
print('map done')

print(dataset['test'])

import numpy as np
inputs = np.stack(dataset['test']['inputs'])
labels = np.stack(dataset['test']['labels'])
puzzle_identifiers = np.stack(dataset['test']['puzzle_identifiers'])
if puzzle_identifiers.ndim > 1:
    puzzle_identifiers = puzzle_identifiers[:, 0]
assert puzzle_identifiers.ndim == 1, puzzle_identifiers.shape

for name, arr in zip(['all__inputs', 'labels', 'all__puzzle_identifiers'], [inputs, labels, puzzle_identifiers]):
    print(name, arr.shape, arr.dtype, np.min(arr), np.max(arr))
    np.save(f"{name}.npy", arr)