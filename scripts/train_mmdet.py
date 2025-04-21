import argparse
import os

import mmyolo
import yaml
from mmengine.config import Config
from mmengine.registry import MODELS, build_from_cfg, init_default_scope
from mmengine.runner import Runner, set_random_seed
from mmyolo.utils import register_all_modules

register_all_modules()

init_default_scope("mmyolo")


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--config_path", type=str, default="./config/base_centernet.yaml"
    )

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

    if hparams["model"]["checkpoints"] is not None and os.path.isfile(
        hparams["model"]["checkpoints"]
    ):
        cfg.load_from = hparams["model"]["checkpoints"]

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
