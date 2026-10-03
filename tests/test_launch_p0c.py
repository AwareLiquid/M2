"""tests/test_launch_p0c.py — launch_p0c.sh 控制流门(2026-09-28 事故工程硬化)

不跑真实训练(桩进程模拟 runner 写账本行),钉住 launcher 三重卫生中
可自动验证的两条:
  1. 幂等重入:结果账本已有 (tag, seed) 行 ⇒ 重跑跳过该 seed;
  2. 双开防护:pidfile 指向活进程 ⇒ 拒绝点火(退出码 1)。
caffeinate 包装与 MPS 真实训练属人工验收项,不在本文件。

轮 20 隔离修复:本文件曾把 canonical jsonl 当沙箱(桩行临时落入真实
账本+purge 读改写窗口与真实 fsync 追加存在吞行竞争)。现全程经
P0C_RESULTS_DIR 指向 tmp_path 沙箱,规范账本零接触,并由
test_canonical_ledger_never_touched 钉死。
"""

import json
import os
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAUNCHER = os.path.join(ROOT, "scripts", "launch_p0c.sh")
STUB = os.path.join(ROOT, "tests", "fixtures", "fake_p0c_runner.py")
CANONICAL = os.path.join(ROOT, "benchmarks", "results",
                         "reasoning_depth.jsonl")
TAG = "t_launch_p0c_test"


def _sandbox_results(tmp_path):
    return str(tmp_path / "reasoning_depth.jsonl")


def _lines(tmp_path):
    path = _sandbox_results(tmp_path)
    if not os.path.exists(path):
        return []
    return [ln for ln in open(path, encoding="utf-8").read().splitlines()
            if ln.strip()]


def _env(tmp_path):
    return {**os.environ,
            "P0C_PYTHON": STUB,
            "P0C_TAG": TAG,
            "P0C_RESULTS_DIR": str(tmp_path),
            "P0C_LOG": str(tmp_path / "launcher.log"),
            "P0C_PIDFILE": str(tmp_path / "launcher.pid")}


def _tag_rows(tmp_path):
    return [json.loads(ln) for ln in _lines(tmp_path)
            if json.loads(ln).get("tag") == TAG]


def test_rerun_skips_completed_seeds(tmp_path):
    """首跑写 2 行;重跑全跳过,不再新增(中断后重跑=免费续跑)。"""
    env = _env(tmp_path)
    r1 = subprocess.run([LAUNCHER, "0", "1"], env=env,
                        capture_output=True, text=True, cwd=ROOT)
    assert r1.returncode == 0, r1.stderr
    rows = _tag_rows(tmp_path)
    assert sorted(r["seed"] for r in rows) == [0, 1]
    assert all(r.get("stub") for r in rows)
    # pidfile 收尾清理
    assert not os.path.exists(env["P0C_PIDFILE"])
    log1 = open(env["P0C_LOG"], encoding="utf-8").read()
    assert log1.count("start") == 2 and "rc=0" in log1

    r2 = subprocess.run([LAUNCHER, "0", "1"], env=env,
                        capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0, r2.stderr
    assert len(_tag_rows(tmp_path)) == 2, "重跑必须跳过已有账本行的 seed"
    log2 = open(env["P0C_LOG"], encoding="utf-8").read()
    assert log2.count("跳过") == 2


def test_refuses_double_launch(tmp_path):
    """活 pid 在 ⇒ 拒绝点火(2026-09-28 之后不再允许双开踩踏)。"""
    env = _env(tmp_path)
    pidfile = env["P0C_PIDFILE"]
    os.makedirs(os.path.dirname(pidfile), exist_ok=True)
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))  # pytest 进程本身=活进程
    try:
        r = subprocess.run([LAUNCHER, "0"], env=env,
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 1
        assert "不双开" in (r.stdout + r.stderr)
        assert _tag_rows(tmp_path) == [], "拒绝点火时不得写入任何行"
    finally:
        os.remove(pidfile)


def test_refuses_double_launch_via_other_pidfile(tmp_path):
    """轮 14 事故补丁:换一个 P0C_PIDFILE 不得绕过双开防护——同目录任一
    *.pid 指向活进程 ⇒ 拒绝点火(2026-10-01 两次真实绕过;T1 单机=全局
    单训练)。"""
    env = _env(tmp_path)
    other = tmp_path / "other_run.pid"
    other.write_text(str(os.getpid()))  # pytest 进程本身=活进程
    try:
        r = subprocess.run([LAUNCHER, "0"], env=env,
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 1
        assert "不双开" in (r.stdout + r.stderr)
        assert _tag_rows(tmp_path) == []
    finally:
        other.unlink()


def test_runner_gets_unbuffered_stdout(tmp_path):
    """轮 14 工程硬化:launcher 必须 PYTHONUNBUFFERED=1 发射 runner——
    stdout 块缓冲使 log 步数滞后真实进度 ~30min,判读被假报零进度误导
    (轮 10/13 实测)。桩把环境探针写进行,断言 launcher 传到了。"""
    env = _env(tmp_path)
    r = subprocess.run([LAUNCHER, "0"], env=env,
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert _tag_rows(tmp_path)[0]["env_unbuffered"] == "1"


def test_control_mode_launches_transformer_only(tmp_path):
    """P0C_MODE=control(对照先行):命令行必须带 --transformer_only、
    不带 --eval_depths,且幂等/双开纪律与 sweep 模式一致。"""
    env = {**_env(tmp_path), "P0C_MODE": "control"}
    r = subprocess.run([LAUNCHER, "0"], env=env,
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    rows = _tag_rows(tmp_path)
    assert len(rows) == 1 and rows[0].get("stub")
    assert "--transformer_only" in rows[0]["flags"]
    assert "--eval_depths" not in rows[0]["flags"]
    assert "--mid_eval" in rows[0]["flags"], "对照腿默认带中途可见性 checkpoint"
    # 幂等:重跑跳过
    r2 = subprocess.run([LAUNCHER, "0"], env=env,
                        capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0, r2.stderr
    assert len(_tag_rows(tmp_path)) == 1


def test_sweep_mode_flags_and_skip_transformer(tmp_path):
    """sweep 模式默认带 --eval_depths;P0C_SKIP_TRANSFORMER=1(phase 2
    全量重发省预算旋钮)追加 --skip_transformer,且不误带对照腿开关。"""
    env = {**_env(tmp_path), "P0C_SKIP_TRANSFORMER": "1"}
    r = subprocess.run([LAUNCHER, "0"], env=env,
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    flags = _tag_rows(tmp_path)[0]["flags"]
    assert "--eval_depths" in flags and "--skip_transformer" in flags
    assert "--transformer_only" not in flags


@pytest.mark.skipif(not os.path.exists(CANONICAL),
                    reason="canonical 账本尚不存在(全新工作区)")
def test_canonical_ledger_never_touched(tmp_path):
    """轮 20 隔离修复回归门:launcher 测试全程沙箱,规范账本零接触——
    修复前本文件曾把 canonical 当沙箱(mtime 被测试套件反复推走,
    2026-10-01 23:06/23:19 两度实证),purge 读改写窗口撞真实 fsync
    追加=吞行风险(2026-10-02 上午 d=8 终评行将落入同一文件)。"""
    before = os.stat(CANONICAL)
    env = _env(tmp_path)
    r = subprocess.run([LAUNCHER, "0", "1"], env=env,
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert len(_tag_rows(tmp_path)) == 2, "沙箱账本必须收到桩行(隔离生效)"
    after = os.stat(CANONICAL)
    assert (before.st_size, before.st_mtime_ns) == \
        (after.st_size, after.st_mtime_ns), \
        "launcher 测试不得触碰规范账本(大小或 mtime 变动=隔离破缺)"
