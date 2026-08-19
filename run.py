import argparse
from pathlib import Path
from typing import Optional

from hydra.utils import instantiate
from omegaconf import OmegaConf

from pipeline_jetson.components.edge.log_util import setup_edge_logging


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


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid --display: {value!r}")


def main(cfg_path: Path, display: Optional[bool] = None):
    setup_edge_logging()
    cfg = load_cfg(cfg_path)
    if display is not None:
        cfg.display = display
    app = instantiate(cfg)
    app.run()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--display", type=_parse_bool, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(Path(args.cfg), display=args.display)
