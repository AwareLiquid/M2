#!/bin/bash
# loop_closer — 合轮收尾器(轮 98 工程硬化;坑账家族=假门禁/假进度三击后立项)
# 语义=GOAL-PROMPT step 3 的机械化执行,纪律文本仍以宪法为准:
#   判单 --add/--check-round → pytest → audit → 三绿方提交 → 推送 → 远端实测回显 → touch 锁
# 任何一门 RC≠0 ⇒ 立即中止,不提交不合轮(轮 92/97 带红合轮事故的结类修复)。
# 用法: scripts/loop_closer.sh <round> <commit-msg> --direction <四轴耦合> --evidence <证据>
set -u
ROUND="$1"; MSG="$2"; shift 2
ROOT="${M2_LOOP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT" || exit 2

fail() { echo "ABORT: $1(不提交不合轮)"; exit 1; }

./scripts/direction_gate --add --round "$ROUND" "$@" >/dev/null || fail "gate --add"
./scripts/direction_gate --check-round "$ROUND" >/dev/null || fail "gate --check-round"
.venv/bin/python -m pytest -q > /tmp/loop_closer_pytest.txt 2>&1 || fail "pytest(见 /tmp/loop_closer_pytest.txt)"
./scripts/goal_check --audit >/dev/null 2>&1 || fail "goal_check --audit"

git add -A || fail "git add"
# 轮 106 账实结类(轮 105 登记):窄白名单(docs/loop 四件)漏 scripts/tests
# 新工具文件(轮 98 实证),也将漏终评判读轮必入库的 benchmarks/verdicts+
# docs/ROADMAP_M2.md;运行时伪迹已由 .gitignore 划界,全仓 add 是唯一稳态。
git commit -q -m "$MSG" || fail "git commit"
git push fork main:loop-rounds-7-18 2>&1 | tail -1
REMOTE=$(git ls-remote fork refs/heads/loop-rounds-7-18 2>/dev/null | cut -c1-7)
LOCAL=$(git rev-parse --short=7 HEAD)
[ -n "$REMOTE" ] && [ "$REMOTE" != "$LOCAL" ] && echo "WARN: 远端($REMOTE)≠本地($LOCAL),欠账按远端实测口径记账"
touch .loop-lock && echo "合轮 round=$ROUND local=$LOCAL remote=${REMOTE:-未出数}"
