import argparse
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf


def load_cfg(cfg_path: Path):
    cfg = OmegaConf.load(cfg_path)
    extends = OmegaConf.select(cfg, "extends")
    if extends:
        base = OmegaConf.load(cfg_path.parent / str(extends))
        cfg = OmegaConf.merge(base, cfg)
        OmegaConf.set_struct(cfg, False)
        if "extends" in cfg:
            del cfg["extends"]
    return cfg


def main(cfg_path: Path):
    cfg = load_cfg(cfg_path)
    app = instantiate(cfg)
    app.run()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg_path = args.cfg
    main(Path(cfg_path))
