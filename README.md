# Reasoning Models in JAX
<img src="sudoku_example.png" width="200px" style="display: block; margin-left: auto; margin-right: auto; margin-bottom: 20px; margin-top: 20px" />

In this repo, we aim to implement [HRM](https://arxiv.org/abs/2506.21734) and [TRM](https://arxiv.org/abs/2510.04871v1) in JAX. We choose [sudoku-extreme-1k](https://huggingface.co/datasets/sapientinc/sudoku-extreme-1k) as the dataset.

## Configurations

- local debug:
    ```bash
    bash debug.sh
    ```

- remote training:
    ```bash
    python main.py --workdir=/path/to/your/workdir --mode=remote_run --config=configs/load_config.py:remote_run_config
    ```

- local training:
    ```bash
    python main.py --workdir=/path/to/your/workdir --mode=remote_run --config=configs/load_config.py:remote_run_config --config.just_evaluate=true --config.load_from=/path/to/your/checkpoint
    ```

- **the final testing**: use any formated Sudoku dataset, and generate output use your checkpoint
    ```bash
    python main.py --workdir=/path/to/your/workdir --mode=remote_run --config=configs/load_config.py:remote_run_config --config.run_inference_folder=true --config.dataset.dataset_path=/path/to/any/dataset
    ```

    This will generate a file named `inference_results.npy` inside your workdir. Then use
    ```bash
    python offline_eval.py --pred /path/to/your/inference_results.npy --ans /path/to/dataset/labels.npy --cfg /path/to/dataset.json
    ```

    Notice that in this setting, `/path/to/any/dataset` may not even have labels inside it. Thus there is no possibility of data leaking.

    The data is locally at `/kmh-nfs-ssd-us-mount/data/sudoku-extreme-1k/`.

## Runs

<!-- - [HRM Init run](https://wandb.ai/evazhu-massachusetts-institute-of-technology/TRM/runs/lr5w1tti) (test pass@1: $88.1\%$) -->