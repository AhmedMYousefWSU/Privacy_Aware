import os.path as osp

import yaml

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


def load_config(path=None):
    path = path or osp.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        if not osp.isabs(v):
            cfg["paths"][k] = osp.join(PROJECT_ROOT, v)
    return cfg


def work(cfg, *parts):
    return osp.join(cfg["paths"]["work_dir"], *parts)
