"""tests/test_loop.py -- loop system: goal_check router, marathon_guard, direction_gate.

The loop scripts (scripts/goal_check, scripts/marathon_guard,
scripts/direction_gate) operate on the repo root by default but accept an
M2_LOOP_ROOT override; every test fabricates a minimal loop workspace in a
tmp dir and drives the real scripts end-to-end through subprocess.
"""
import json
import os
import re
import subprocess
import sys
import time

import pytest

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))

GOAL_CHECK = os.path.join(SCRIPTS, "goal_check")
MARATHON_GUARD = os.path.join(SCRIPTS, "marathon_guard")
DIRECTION_GATE = os.path.join(SCRIPTS, "direction_gate")

GOALS_TEMPLATE = """# GOALS.md — 程序计数器

```yaml
state: RUNNING
mode: {mode}
current_action: test
current_variable: test-var(单变量锚点)
```

```yaml
goal_queue:
{queue}
```
"""


def make_root(tmp_path, queue, mode="ON"):
    root = tmp_path / "repo"
    loop = root / "docs" / "loop"
    loop.mkdir(parents=True)
    (loop / "GOALS.md").write_text(GOALS_TEMPLATE.format(mode=mode, queue=queue), encoding="utf-8")
    return root


def run(script, root, *args):
    env = dict(os.environ, M2_LOOP_ROOT=str(root))
    return subprocess.run(
        [sys.executable, script, *args], env=env,
        capture_output=True, text=True, timeout=60,
    )


def read_goals(root):
    return (root / "docs" / "loop" / "GOALS.md").read_text(encoding="utf-8")


def entry(id_, check_cmd, status="todo"):
    return (
        f"  - id: {id_}\n"
        f"    goal: {id_} goal\n"
        f"    done_condition: {id_} done\n"
        f"    check_cmd: {check_cmd}\n"
        f"    status: {status}\n"
    )


class TestGoalCheckRouter:
    def test_not_achieved_keeps_queue(self, tmp_path):
        """队首未达 ⇒ 队列条目不因队首未达而被弹出(深位达成的弹出
        行为由 test_deep_achieved_pops_while_head_stays 专门覆盖)。"""
        root = make_root(tmp_path, entry("a", "false") + entry("b", "false"))
        r = run(GOAL_CHECK, root)
        assert r.returncode == 1
        assert "NOT-Achieved" in r.stdout
        assert "- id: a" in read_goals(root) and "- id: b" in read_goals(root)

    def test_achieved_pops_first_and_promotes(self, tmp_path):
        root = make_root(tmp_path, entry("a", "true") + entry("b", "false"))
        r = run(GOAL_CHECK, root)
        assert r.returncode == 0
        assert "ACHIEVED" in r.stdout and "b" in r.stdout
        body = read_goals(root)
        assert "- id: a" not in body
        assert "- id: b" in body
        assert "```yaml" in body and "state: RUNNING" in body

    def test_achieved_pops_only_first_entry_multiline_queue(self, tmp_path):
        """回归:pop_first 曾在 ≥3 条队列上隔条删除(吞掉第 3 条)。

        用真实 GOALS.md 格式(多行折行 goal 字段)验证只摘除队首,
        其余 3 条逐字节保留。
        """
        multiline = (
            "  - id: a-first\n"
            "    goal: 长字段折行测试——γ 全局头配额\n"
            "      + full MHA 修复注意力后做 fixed-depth sweep,\n"
            "      多 seed 纪律(双峰任务报 grok 率)。\n"
            "    done_condition: 判决文件存在且含\n"
            "      h_supported 字段(预注册格式)。\n"
            "    check_cmd: true\n"
            "    status: todo\n"
        )
        queue = (multiline
                 + entry("b-second", "false")
                 + entry("c-third", "false")
                 + entry("d-fourth", "false"))
        root = make_root(tmp_path, queue)
        r = run(GOAL_CHECK, root)
        assert r.returncode == 0
        body = read_goals(root)
        ids = re.findall(r"(?m)^\s*- id: (\S+)", body)
        assert ids == ["b-second", "c-third", "d-fourth"], f"弹出吞错了条目: {ids}"
        assert "check_cmd: false" in body and "status: todo" in body

    def test_achieved_on_single_entry_empties_queue(self, tmp_path):
        root = make_root(tmp_path, entry("only", "true"))
        r = run(GOAL_CHECK, root)
        assert r.returncode == 0
        assert "- id:" not in read_goals(root)

    def test_deep_achieved_pops_while_head_stays(self, tmp_path):
        """AMM-004/005 蒸馏门核心行为:每轮全队列检测——深位条目达成即
        当场弹出,不再等它冒到队首;队首未达 ⇒ 仍 NOT-Achieved(exit 1)。
        """
        root = make_root(tmp_path, entry("head", "false")
                         + entry("done-mid", "true")
                         + entry("tail", "false"))
        r = run(GOAL_CHECK, root)
        assert r.returncode == 1
        assert "NOT-Achieved" in r.stdout
        assert "深位达成同轮弹出" in r.stdout and "done-mid" in r.stdout
        ids = re.findall(r"(?m)^\s*- id: (\S+)", read_goals(root))
        assert ids == ["head", "tail"]

    def test_all_achieved_pops_all(self, tmp_path):
        root = make_root(tmp_path, entry("a", "true") + entry("b", "true"))
        r = run(GOAL_CHECK, root)
        assert r.returncode == 0
        assert "ACHIEVED" in r.stdout
        assert "- id:" not in read_goals(root)

    def test_deep_pop_preserves_folded_goal_multiline(self, tmp_path):
        """pop_ids 回归(承 pop_first 隔条删除 bug 纪律):深位弹出时,
        相邻条目的多行折行 goal/done_condition 字段逐字节保留。"""
        head_multiline = (
            "  - id: head-first\n"
            "    goal: 折行字段——γ 全局头配额\n"
            "      + full MHA 修复注意力,\n"
            "      多 seed 纪律。\n"
            "    done_condition: 判决文件存在且含\n"
            "      h_supported 字段。\n"
            "    check_cmd: false\n"
            "    status: doing\n"
        )
        queue = (head_multiline
                 + entry("deep-done", "true")
                 + entry("c-third", "false"))
        root = make_root(tmp_path, queue)
        r = run(GOAL_CHECK, root)
        assert r.returncode == 1
        body = read_goals(root)
        assert "goal: 折行字段——γ 全局头配额\n      + full MHA 修复注意力,\n      多 seed 纪律。" in body
        assert "h_supported 字段" in body
        ids = re.findall(r"(?m)^\s*- id: (\S+)", body)
        assert ids == ["head-first", "c-third"]

    def test_current_variable_echoed_on_route(self, tmp_path):
        """蒸馏锚点回显:路由输出必须带 current_variable(例外轮纪律的
        可见化,防'每轮合规但合力不指向同一变量')。"""
        root = make_root(tmp_path, entry("a", "false"))
        r = run(GOAL_CHECK, root)
        assert r.returncode == 1
        assert "current_variable" in r.stdout and "test-var" in r.stdout

    def test_queue_empty(self, tmp_path):
        root = make_root(tmp_path, "")
        r = run(GOAL_CHECK, root)
        assert r.returncode == 2
        assert "QUEUE-EMPTY" in r.stdout

    def test_mode_off_refuses(self, tmp_path):
        root = make_root(tmp_path, entry("a", "true"), mode="OFF")
        r = run(GOAL_CHECK, root)
        assert r.returncode == 5
        assert "MODE-OFF" in r.stdout

    def test_heartbeat_touches_lock(self, tmp_path):
        root = make_root(tmp_path, entry("a", "false"))
        assert not (root / ".loop-lock").exists()
        run(GOAL_CHECK, root)
        assert (root / ".loop-lock").exists()


class TestGoalCheckAudit:
    def test_healthy_queue(self, tmp_path):
        root = make_root(tmp_path, entry("a", "true") + entry("b", "false"))
        r = run(GOAL_CHECK, root, "--audit")
        assert r.returncode == 0
        assert "2 条目全部健康" in r.stdout

    def test_duplicate_id_detected(self, tmp_path):
        root = make_root(tmp_path, entry("a", "true") + entry("a", "false"))
        r = run(GOAL_CHECK, root, "--audit")
        assert r.returncode == 1
        assert "重复条目 id" in r.stdout

    def test_missing_check_cmd_detected(self, tmp_path):
        queue = "  - id: a\n    goal: a\n    status: todo\n"
        root = make_root(tmp_path, queue)
        r = run(GOAL_CHECK, root, "--audit")
        assert r.returncode == 1
        assert "check_cmd 缺失/空" in r.stdout

    def test_missing_queue_block_detected(self, tmp_path):
        root = tmp_path / "repo"
        loop = root / "docs" / "loop"
        loop.mkdir(parents=True)
        (loop / "GOALS.md").write_text("# no queue here\n", encoding="utf-8")
        r = run(GOAL_CHECK, root, "--audit")
        assert r.returncode == 1
        assert "未找到 goal_queue" in r.stdout

    def test_audit_requires_current_variable(self, tmp_path):
        """蒸馏门锚点进 audit 数数锚:current_variable 缺失=审计不过
        (AMM-005:变量纪律不靠自觉)。"""
        root = tmp_path / "repo"
        loop = root / "docs" / "loop"
        loop.mkdir(parents=True)
        (loop / "GOALS.md").write_text(
            GOALS_TEMPLATE.format(mode="ON", queue=entry("a", "true"))
            .replace("current_variable: test-var(单变量锚点)\n", ""),
            encoding="utf-8")
        r = run(GOAL_CHECK, root, "--audit")
        assert r.returncode == 1
        assert "current_variable 缺失" in r.stdout


class TestMarathonGuard:
    def test_no_lock_is_clear(self, tmp_path):
        root = make_root(tmp_path, "")
        r = run(MARATHON_GUARD, root)
        assert r.returncode == 0

    def test_fresh_lock_is_busy(self, tmp_path):
        root = make_root(tmp_path, "")
        (root / ".loop-lock").write_text(str(int(time.time())))
        r = run(MARATHON_GUARD, root)
        assert r.returncode == 1
        assert "BUSY" in r.stdout

    def test_expired_lock_is_clear(self, tmp_path):
        root = make_root(tmp_path, "")
        lock = root / ".loop-lock"
        lock.write_text("0")
        old = time.time() - 200 * 60
        os.utime(lock, (old, old))
        r = run(MARATHON_GUARD, root)
        assert r.returncode == 0
        assert "自过期" in r.stdout


class TestDirectionGate:
    def test_add_then_check_round_passes(self, tmp_path):
        root = make_root(tmp_path, "")
        r = run(DIRECTION_GATE, root, "--add", "--round", "3",
                "--direction", "四轴耦合: reasoning", "--evidence", "tests 全绿")
        assert r.returncode == 0
        r = run(DIRECTION_GATE, root, "--check-round", "3")
        assert r.returncode == 0
        lines = (root / "docs" / "loop" / "direction-gate.jsonl").read_text().strip().splitlines()
        assert json.loads(lines[-1])["round"] == 3

    def test_check_round_mismatch_fails(self, tmp_path):
        root = make_root(tmp_path, "")
        run(DIRECTION_GATE, root, "--add", "--round", "2",
            "--direction", "d", "--evidence", "e")
        r = run(DIRECTION_GATE, root, "--check-round", "3")
        assert r.returncode == 1
        assert "轮号" in r.stdout

    def test_check_round_on_empty_ledger_fails(self, tmp_path):
        root = make_root(tmp_path, "")
        r = run(DIRECTION_GATE, root, "--check-round", "1")
        assert r.returncode == 1

    def test_add_requires_all_fields(self, tmp_path):
        root = make_root(tmp_path, "")
        r = run(DIRECTION_GATE, root, "--add", "--round", "1",
                "--direction", "", "--evidence", "e")
        assert r.returncode != 0

    def test_drift_streak_warns(self, tmp_path):
        root = make_root(tmp_path, "")
        for i in range(2):
            run(DIRECTION_GATE, root, "--add", "--round", str(i + 1),
                "--direction", "DRIFT", "--evidence", "e")
        r = run(DIRECTION_GATE, root, "--add", "--round", "3",
                "--direction", "DRIFT", "--evidence", "e")
        assert r.returncode == 0
        assert "BLOCKED-HUMAN" in r.stdout


@pytest.mark.parametrize("script", [GOAL_CHECK, MARATHON_GUARD, DIRECTION_GATE])
def test_scripts_are_executable(script):
    assert os.access(script, os.X_OK), f"{script} 缺少可执行位"
