"""loop_closer 合轮收尾器测试(轮 98 工程硬化)。

背景事故:轮 92 带红合轮(裸管道吞 pytest RC)+ 轮 97 带红合轮
(`;` 链报告失败但不中止)——本工具的契约=任何一门红 ⇒ 中止且不提交。
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLOSER = REPO / "scripts" / "loop_closer.sh"


def run_closer(root: Path, *args):
    env = dict(os.environ, M2_LOOP_ROOT=str(root))
    return subprocess.run(
        ["bash", str(CLOSER), *args], env=env,
        capture_output=True, text=True, timeout=120,
    )


def test_aborts_when_gate_check_fails(tmp_path):
    """gate --check-round 红(空账本无判单)⇒ 非零退出且不产生提交。"""
    (tmp_path / "docs" / "loop").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    r = run_closer(tmp_path, "99", "test msg",
                   "--direction", "d", "--evidence", "e")
    assert r.returncode != 0
    assert "ABORT" in r.stderr + r.stdout
    assert "合轮 round=" not in r.stdout + r.stderr


def test_refuses_missing_args():
    """缺参 ⇒ 非零退出(set -u 防御)。"""
    r = subprocess.run(["bash", str(CLOSER)], capture_output=True, text=True)
    assert r.returncode != 0


def test_closer_stages_whole_tree_not_whitelist():
    """轮 105 账实不符结类(轮 106 修复):closer 的 git add 窄白名单
    (docs/loop 四件)漏 scripts/tests 新工具文件(轮 98"消息有账、库里
    无物"实证),也将漏终评判读轮必入库的 benchmarks/verdicts+ROADMAP。
    修复=add -A 全仓(运行时伪迹已由 .gitignore 划界:benchmarks/results/
    runs/ data/ .loop-lock)。守护=脚本禁再退回窄白名单。"""
    src = CLOSER.read_text(encoding="utf-8")
    assert "git add -A" in src, "closer 必须 add -A 全仓(轮 105 账实结类)"
    assert "git add docs/loop/DISTILL.md" not in src, \
        "closer 退回窄白名单=账实不符坑复发(轮 98/105)"


if __name__ == "__main__":
    sys.exit(subprocess.run([sys.executable, "-m", "pytest", __file__]).returncode)
