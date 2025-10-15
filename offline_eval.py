"""
Offline evaluation for general datasets
"""
# example: python offline_eval.py --pred path/to/pred.npy --ans /kmh-nfs-ssd-us-mount/data/sudoku-extreme-full/test/all__labels.npy --cfg /kmh-nfs-ssd-us-mount/data/sudoku-extreme-full/test/dataset.json

import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ans", type=str, required=True, help="Path to the GT npy file")
parser.add_argument("--pred", type=str, required=True, help="Path to the prediction npy file")
parser.add_argument("--cfg", type=str, required=True, help="Path to the dataset.json file")

def assert_suffix(file, suffix):
    assert os.path.exists(file), f"{file} does not exist"
    assert file.endswith(suffix), f"{file} should be a {suffix} file"

args = parser.parse_args()
ans_file, pred_file, cfg_file = args.ans, args.pred, args.cfg

assert_suffix(ans_file, ".npy")
assert_suffix(pred_file, ".npy")
assert_suffix(cfg_file, ".json")

with open(cfg_file, "r") as f:
    cfg = json.load(f)

ans_data = np.load(ans_file)
pred_data = np.load(pred_file)
assert ans_data.shape == pred_data.shape, f"shape mismatch. GT shape: {ans_data.shape}, pred shape: {pred_data.shape}"
if ans_data.dtype != pred_data.dtype:
    print(f"[WARNING] dtype mismatch. GT dtype: {ans_data.dtype}, pred dtype: {pred_data.dtype}. Converting pred to {ans_data.dtype}")
    pred_data = pred_data.astype(ans_data.dtype)

valid_data = ans_data != cfg["ignore_label_id"]
all_correct = (ans_data == pred_data) & valid_data
acc = all_correct.all(axis=-1).mean()
print(f"Overall accuracy: {acc*100:.2f}%")


################
# train set
# ython offline_eval.py --pred /kmh-nfs-ssd-us-mount/staging/siri/sudoku/launch_20251014_045811_git91667fd_32382eca/logs/log1_20251014_045837_VMkmh-tpuvm-v4-8-8_98299690/inference_results.npy --ans /kmh-nfs-ssd-us-mount/data/sudoku-extreme-full/train/all__labels.npy --cfg /kmh-nfs-ssd-us-mount/data/sudoku-extreme-full/train/dataset.json
# [WARNING] dtype mismatch. GT dtype: uint8, pred dtype: int32. Converting pred to uint8
# Overall accuracy: 87.41%

# validation set
# python offline_eval.py --pred /kmh-nfs-ssd-us-mount/staging/siri/sudoku/launch_20251014_030345_git91667fd_f51c9424/logs/log1_20251014_030409_VMkmh-tpuvm-v4-8-8_3d19e712/inference_results.npy --ans /kmh-nfs-ssd-us-mount/data/sudoku-extreme-full/test/all__labels.npy --cfg /kmh-nfs-ssd-us-mount/data/sudoku-extreme-full/test/dataset.json
# [WARNING] dtype mismatch. GT dtype: uint8, pred dtype: int32. Converting pred to uint8
# Overall accuracy: 85.30%


# web dataset, ckpt older
# python offline_eval.py --pred /kmh-nfs-ssd-us-mount/staging/siri/sudoku/launch_20251014_044949_git91667fd_6ed9b3c9/logs/log1_20251014_045014_VMkmh-tpuvm-v4-8-8_9e0c4402/inference_results.npy --ans /kmh-nfs-ssd-us-mount/code/siri/jax_reasoning/other_data/labels.npy --cfg /kmh-nfs-ssd-us-mount/code/siri/jax_reasoning/other_data/test_folder/dataset.json
# [WARNING] dtype mismatch. GT dtype: int64, pred dtype: int32. Converting pred to int64
# Overall accuracy: 28.57%

# test set
# python offline_eval.py --pred /kmh-nfs-ssd-us-mount/staging/siri/sudoku/launch_20251014_220415_git31d4bb9_8bdca47e/logs/log1_20251014_220441_VMkmh-tpuvm-v4-8-8_a3386d3d/inference_results.npy --ans /kmh-nfs-ssd-us-mount/code/siri/jax_reasoning/all_datas/sudoku/sudoku_9_22_25_test/labels.npy --cfg /kmh-nfs-ssd-us-mount/code/siri/jax_reasoning/all_datas/sudoku/sudoku_9_22_25_test/test_folder/dataset.json
# [WARNING] dtype mismatch. GT dtype: int64, pred dtype: int32. Converting pred to int64
# Overall accuracy: 98.99%

# train on 1k, valid on hard
# python offline_eval.py --pred /kmh-nfs-ssd-us-mount/staging/siri/sudoku/launch_20251015_044013_git32736aa_aaa0556f/logs/log1_20251015_044039_VMkmh-tpuvm-v4-16-spot-kangyang-1_415d757d/inference_results.npy --ans /kmh-nfs-ssd-us-mount/data/sudoku-extreme-1k/test_hard/all__labels.npy --cfg /kmh-nfs-ssd-us-mount/data/sudoku-extreme-1k/test_hard/dataset.json

# train on 1k, valid on hard
# python offline_eval.py --pred /kmh-nfs-ssd-us-mount/staging/siri/sudoku/launch_20251015_044013_git32736aa_aaa0556f/logs/log1_20251015_044039_VMkmh-tpuvm-v4-16-spot-kangyang-1_415d757d/inference_results.npy --ans /kmh-nfs-ssd-us-mount/data/sudoku-extreme-1k/test_hard/all__labels.npy --cfg /kmh-nfs-ssd-us-mount/data/sudoku-extreme-1k/test_hard/dataset.json

# sudoku bench
# python offline_eval.py --pred /kmh-nfs-ssd-us-mount/staging/siri/sudoku/launch_20251015_045023_git32736aa_ad03acc8/logs/log1_20251015_045049_VMkmh-tpuvm-v4-16-spot-kangyang-1_36c62af9/inference_results.npy --ans /kmh-nfs-ssd-us-mount/data/sudoku-extreme-1k/test_sudoku_bench/all__labels.npy --cfg /kmh-nfs-ssd-us-mount/data/sudoku-extreme-1k/test_sudoku_bench/dataset.json