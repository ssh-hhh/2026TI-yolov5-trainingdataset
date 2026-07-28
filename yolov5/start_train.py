# -*- coding: utf-8 -*-
"""一键训练 — 参数在 train_config.yaml 里改"""
import sys, os, subprocess, yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("train_config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cmd = [sys.executable, "train.py"]
for k, v in cfg.items():
    cmd += [f"--{k}", str(v)]

print("=" * 50)
print(f"  钢珠检测 | {cfg['weights']} | {cfg['img']}px | batch={cfg['batch']}")
print(f"  epochs={cfg['epochs']} | GPU {cfg['device']} | -> runs/train/{cfg['name']}/")
print("=" * 50)

subprocess.run(cmd)
