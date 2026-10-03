#!/usr/bin/env python3
"""tests/fixtures/fake_p0c_runner.py — launch_p0c.sh 控制流测试专用桩。

解析 --seeds/--tag,向结果账本追加一行 stub 行后退出 0。
绝不执行真实训练;账本目录经 P0C_RESULTS_DIR 沙箱化(轮 20),与真实
runner 的同一旋钮对齐;未设时回落生产相对路径(干跑兼容)。
"""
import json
import os
import sys

seed = tag = None
flags = []
argv = sys.argv[1:]
for i, a in enumerate(argv):
    if a == "--seeds" and i + 1 < len(argv):
        seed = int(argv[i + 1])
    elif a == "--tag" and i + 1 < len(argv):
        tag = argv[i + 1]
    elif a in ("--transformer_only", "--mid_eval", "--eval_depths",
               "--skip_transformer"):
        flags.append(a)

path = os.path.join(os.environ.get("P0C_RESULTS_DIR",
                                   os.path.join("benchmarks", "results")),
                    "reasoning_depth.jsonl")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"mode": "fixed_sweep", "seed": seed, "tag": tag,
                        "stub": True, "flags": flags,
                        "env_unbuffered": os.environ.get("PYTHONUNBUFFERED")
                        }) + "\n")
