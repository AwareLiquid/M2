#!/usr/bin/env python3
"""distill_inject — 蒸馏门检索式注入(AMM-015;ExpeL 模式 store→检索→组装)

用法: .venv/bin/python scripts/distill_inject.py --variable "<current_variable>" [--top 3]
从 docs/loop/DISTILL.md 的 ### 轮 条目中,按 current_variable 关键词
(≥2 字词元)命中打分,输出最近 top N 相关条目供每轮三查注入;
零命中时打印提示(读回环退化为尾部 2 条,由协议步骤 1 兜底)。
"""
import argparse
import os
import re
import sys


def repo_root():
    root = os.environ.get("M2_LOOP_ROOT")
    if root:
        return os.path.abspath(root)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def entries(distill_path):
    src = open(distill_path, encoding="utf-8").read()
    return re.findall(r"(?m)^### (轮 .+?)$(.*?)(?=^### |\Z)", src, re.S)


def keywords(variable):
    toks = [t for t in re.split(r"[^\w]+", variable) if len(t) >= 2]
    return toks or ([variable] if variable else [])


def main():
    ap = argparse.ArgumentParser(description="蒸馏门检索式注入(AMM-015)")
    ap.add_argument("--variable", required=True, help="GOALS current_variable 原文")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    path = os.path.join(repo_root(), "docs", "loop", "DISTILL.md")
    kws = keywords(args.variable)
    scored = []
    for title, body in entries(path):
        s = sum((title + body).count(k) for k in kws)
        if s > 0:
            scored.append((s, title, body))
    scored.sort(key=lambda x: int(re.search(r"轮 (\d+)", x[1]).group(1)), reverse=True)
    if not scored:
        print("distill_inject: 零命中(读回环退化为尾部 2 条,步骤 1 兜底)")
        return 0
    for _, title, body in scored[: args.top]:
        print(f"### {title}{body.rstrip()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
