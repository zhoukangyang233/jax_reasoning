import json
import numpy as np
from pathlib import Path
import shutil

def main(sudoku_jsonl, dest='./all_datas/sudoku'):
    output_dir = Path(sudoku_jsonl).name.replace('.jsonl', '')
    output_dir = Path(dest) / output_dir
    if output_dir.exists():
        print(f'Overwritting existing directory {output_dir}...')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_folder = output_dir / 'test_folder'
    test_folder.mkdir(parents=True, exist_ok=True)

    inputs = []
    labels = []

    with open(sudoku_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            puzzle = data['input']
            solution = data['output']
            
            inputs.append(np.array([int(x) if x != '.' else 0 for x in puzzle if x != '\n']) + 1)
            labels.append(np.array([int(x) for x in solution if x != '\n']) + 1)
            
    inputs = np.stack(inputs)
    labels = np.stack(labels)
    assert inputs.shape == labels.shape, (inputs.shape, labels.shape)
    identifiers = np.zeros((inputs.shape[0],), dtype=np.int32)

    np.save(test_folder / 'all__inputs.npy', inputs)
    print('inputs num hints:', np.sum(inputs > 1, axis=1).mean())
    np.save(output_dir / 'labels.npy', labels)
    np.save(test_folder / 'all__puzzle_identifiers.npy', identifiers)
    
    dataset_cfg = {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": 11, "seq_len": 81, "num_puzzle_identifiers": 1, "total_groups": 422786, "mean_puzzle_examples": 1.0, "sets": ["all"]}
    with open(test_folder / 'dataset.json', 'w') as f:
        json.dump(dataset_cfg, f)

    shutil.move(sudoku_jsonl, output_dir / Path(sudoku_jsonl).name)
    print(f'Test folder: {test_folder}')

if __name__ == "__main__":
    main('sudoku_9_22_25_test.jsonl')  # Replace with your actual file path