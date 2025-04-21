import argparse
import os

import yaml
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import Runner, set_random_seed


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config_path", type=str, default="./config/base_yolov8.yaml")

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    set_random_seed(hparams["seed"])

    os.makedirs(hparams["output_root_dir"], exist_ok=True)

    cfg = Config.fromfile(hparams["model"]["mm_config_path"])
    cfg.work_dir = os.path.join(
        hparams["output_root_dir"],
        hparams["experiment_name"],
    )

    cfg.load_from = hparams["model"]["checkpoints"]

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
