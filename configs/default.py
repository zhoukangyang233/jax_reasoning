import ml_collections

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.dataset_cls = "sudoku"
    dataset.dataset_path = "/kmh-nfs-ssd-us-mount/data/sudoku-extreme-full"
    
    dataset.seq_len = 81
    dataset.vocab_size = 11
    dataset.num_puzzle_identifiers = 1
    
    dataset.num_workers = 4
    dataset.prefetch_factor = 2
    dataset.pin_memory = False
    dataset.cache = True
    
    config.model = model = ml_collections.ConfigDict()
    model.name = "HRM_default"
    
    config.training = training = ml_collections.ConfigDict()
    training.seed = 42
    training.batch_size = 768
    training.eval_batch_size = 768
    training.epochs = 100_000
    training.log_per_step = 100
    training.num_vis = 4
    training.eval_interval = 10_000
    training.checkpoint_interval = 10_000
    training.loss_fn = 'stablemax_cross_entropy'
    
    training.optimizer = 'adamw'  # adamw, adam_atan2
    training.learning_rate = 1e-4
    training.lr_schedule = 'cos'
    training.lr_min_ratio = 1.0
    training.warmup_steps = 2_000
    training.weight_decay = 0.1

    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.on_use = True
    wandb.notes = ''
    
    # pretrained model
    config.run_inference_folder = False
    config.just_evaluate = False
    config.load_from = ''

    return config
