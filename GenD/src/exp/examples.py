from copy import deepcopy

from .. import config as C
from ..config import Config

experiments = {
    "example-training": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.Linear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(ce_labels=1.0),
            run_dir="runs/example",
            val_files=[],
            batch_size=2,
            mini_batch_size=2,
            max_epochs=1,
            wandb=False,
            devices=4,
        )
    ],
    "example-test": [
        Config(
            run_dir="runs/test",
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=4,
        )
    ],
    "GenD_CLIP--CDFv2-example": [
        Config(
            run_dir="runs/test",
            checkpoint="yermandy/GenD_CLIP_L_14",
            max_epochs=1,
            wandb=False,
            devices=4,
        )
    ],
    "GenD_PE--CDFv2-example": [
        Config(
            run_dir="runs/test",
            checkpoint="yermandy/GenD_PE_L",
            max_epochs=1,
            wandb=False,
            devices=4,
        )
    ],
    "GenD_DINO--CDFv2-example": [
        Config(
            run_dir="runs/test",
            checkpoint="yermandy/GenD_DINOv3_L",
            max_epochs=1,
            wandb=False,
            devices=4,
        )
    ],
}


_LOCAL_MAIN_CONFIG = Config(
    backbone=C.Backbone.CLIP_L_14,
    head=C.Head.NLinear,
    unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
    loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.1),
    run_dir="runs/local-main",
    val_files=[],
    batch_size=128,
    mini_batch_size=2,
    max_epochs=30,
    lr_scheduler="cyclic",
    num_epochs_in_cycle=10,
    monitor_metric="train/loss",
    monitor_metric_mode="min",
    early_stopping_patience=-1,
    compute_train_metrics=False,
    wandb=False,
    devices=4,
)

experiments["local-main"] = [deepcopy(_LOCAL_MAIN_CONFIG)]
for seed in range(5):
    config = deepcopy(_LOCAL_MAIN_CONFIG)
    config.seed = seed
    experiments[f"local-main-seed{seed}"] = [config]


# Official HF weights (FF++ c23, paper Tab. 2) evaluated with the dfd-hr protocol:
# CDF-v2, FSh, DFD, DFDC, DFDCP from DeepfakeBench JSON (/data/hjr/dfbench) + WDF from the folder test set,
# <= 32 frames per video. See src/utils/dfbench.py. Test files are collected only when the experiment runs.
def _use_dfdhr_test_protocol(config: Config) -> Config:
    from ..utils.dfbench import dfdhr_test_files

    config.tst_files = dfdhr_test_files()
    return config


def _gend_checkpoint(repo_id: str) -> str:
    """Use a local download at <GEND_WEIGHTS_ROOT>/<repo_id> (default weights/yermandy/...) if present,
    otherwise the Hugging Face Hub id. The path keeps 'yermandy/' so run.py routes it to GenDHF."""
    import os

    local = os.path.join(os.environ.get("GEND_WEIGHTS_ROOT", "weights"), repo_id)
    return local if os.path.isfile(os.path.join(local, "config.json")) else repo_id


for _name, _ckpt in {
    "GenD_CLIP--HF-local-test": _gend_checkpoint("yermandy/GenD_CLIP_L_14"),
    "GenD_PE--HF-local-test": _gend_checkpoint("yermandy/GenD_PE_L"),
    "GenD_DINO--HF-local-test": _gend_checkpoint("yermandy/GenD_DINOv3_L"),
}.items():
    experiments[_name] = [
        Config(
            run_dir="runs/test",
            checkpoint=_ckpt,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
        ),
        _use_dfdhr_test_protocol,
    ]
