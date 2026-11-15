"""distill_inject 注入管道测试(AMM-015;ExpeL 模式 store→检索→组装)。"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INJECT = REPO / "scripts" / "distill_inject.py"

FIXTURE = """# DISTILL

### 轮 1(2026-10-01;无关条目)
- 有效经验: 与研究变量无关的条目。

### 轮 2(2026-10-02;变量相关)
- 现状: p0c-prime-depth 判读在途。
- 有效经验: 单 seed 不可裁决,走 negative_diagnoses。

### 轮 3(2026-10-02;尾部无关)
- 有效经验: 排版卫生。
"""


def run(root: Path, *args):
    env = dict(os.environ, M2_LOOP_ROOT=str(root))
    return subprocess.run(
        [sys.executable, str(INJECT), *args], env=env,
        capture_output=True, text=True, timeout=60,
    )


def test_retrieves_variable_relevant_entries(tmp_path):
    (tmp_path / "docs" / "loop").mkdir(parents=True)
    (tmp_path / "docs" / "loop" / "DISTILL.md").write_text(FIXTURE, encoding="utf-8")
    r = run(tmp_path, "--variable", "p0c-prime-depth")
    assert r.returncode == 0
    assert "轮 2" in r.stdout
    assert "单 seed 不可裁决" in r.stdout
    assert "排版卫生" not in r.stdout, "未命中条目不得注入"


def test_zero_hit_falls_back(tmp_path):
    (tmp_path / "docs" / "loop").mkdir(parents=True)
    (tmp_path / "docs" / "loop" / "DISTILL.md").write_text(FIXTURE, encoding="utf-8")
    r = run(tmp_path, "--variable", "zzz-unrelated-var")
    assert r.returncode == 0
    assert "零命中" in r.stdout
