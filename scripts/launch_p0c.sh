#!/usr/bin/env bash
# launch_p0c.sh — P0-C′ 训练腿发射器(2026-10-01 轮 6 工程硬化;轮 7 加对照先行 mode)
#
# 背景:2026-09-28 事故——18.5h 训练随机器关机全损,无 checkpoint、无增量
# 落盘、无点火排程,循环静默停摆 3.5 天。本脚本三重卫生:
#   1. caffeinate 防睡眠:机器睡眠/关机=进程死,事故根因之一;
#   2. 每 seed 一次调用:预注册 protocol.runner 本意(单点死亡只损失一条腿,
#      不再是整个 3-seed sweep);
#   3. 幂等重入:canonical jsonl 已有 (tag, seed) 行即跳过该 seed——任何
#      中断后直接重跑本脚本即可续跑缺失的腿。
#   依赖:reasoning_depth.py 增量落盘(每个 depth-run/transformer eval 完成
#   即写 partial 台账;mid-eval checkpoint 亦只写 partial,死亡最多丢一个
#   进行中的臂)。
#
# P0C_MODE(轮 7,对照先行=用户 2026-10-01 批准的执行改订):
#   sweep   默认 = 预注册原协议: MT-LNN fixed-depth sweep × seeds(tag p0c_prime)
#   control        = 只训 transformer 对照臂(--transformer_only,任务可达性
#                    佐证、不入判据;tag p0c_prime_v2_control)
#
# 测试/干跑旋钮(默认=预注册协议原样):
#   P0C_DEVICE=cpu P0C_STEPS=2 P0C_DEPTHS="1 2" P0C_TAG=p0c_smoke_test \
#     ./scripts/launch_p0c.sh 0
#   对照先行腿: P0C_MODE=control ./scripts/launch_p0c.sh 0
set -uo pipefail
cd "$(dirname "$0")/.."

# 轮 14 工程硬化:python stdout 块缓冲使 log 步数滞后真实进度可达 ~30min,
# 两轮判读被假报零进度误导(轮 13 实测 90s 短窗撞缓冲边界);unbuffered
# 让 log 即时反映在途状态,步速计量不再依赖长窗采样。
export PYTHONUNBUFFERED=1

PY=${P0C_PYTHON:-.venv/bin/python}
DEVICE=${P0C_DEVICE:-mps}
STEPS=${P0C_STEPS:-30000}
DEPTHS=${P0C_DEPTHS:-1 2 4 8}
MID_EVAL=${P0C_MID_EVAL:-10000 20000}
MODE=${P0C_MODE:-sweep}
RESULTS=benchmarks/results/reasoning_depth.jsonl

case $MODE in
  sweep)
    TAG=${P0C_TAG:-p0c_prime}
    PIDFILE=${P0C_PIDFILE:-benchmarks/results/p0c_prime_run.pid}
    LOG=${P0C_LOG:-benchmarks/results/p0c_prime_run.log}
    ;;
  control)
    TAG=${P0C_TAG:-p0c_prime_v2_control}
    PIDFILE=${P0C_PIDFILE:-benchmarks/results/p0c_control_run.pid}
    LOG=${P0C_LOG:-benchmarks/results/p0c_control_run.log}
    ;;
  *)
    echo "P0C_MODE=$MODE 无效(sweep|control)" >&2
    exit 2
    ;;
esac

# 双开防护:活 pid 不双开(死 pid 视为陈旧,直接接管)。
# 轮 14 工程硬化:防护覆盖 PIDFILE 同目录全部 *.pid——2026-10-01 事故:换一个
# P0C_PIDFILE 即绕过单文件检查,同一 GPU 上重复发射(T1 单机=全局单训练)。
for pf in "$PIDFILE" "$(dirname "$PIDFILE")"/*.pid; do
  [[ -f $pf ]] || continue
  oldpid=$(cat "$pf" 2>/dev/null || true)
  if [[ -n $oldpid ]] && kill -0 "$oldpid" 2>/dev/null; then
    if [[ $pf == "$PIDFILE" ]]; then
      echo "已有活训练进程 pid=$oldpid,不双开。查进度: tail -f $LOG" >&2
    else
      echo "$pf 指向活进程 pid=$oldpid(异 pidfile 双开),不双开。查进度: tail -f $LOG" >&2
    fi
    exit 1
  fi
done

seeds=("$@")
[[ ${#seeds[@]} -eq 0 ]] && seeds=(0 1 2)

seed_done() {  # seed_done S TAG: canonical jsonl 已有 (TAG, S) 行 ⇒ 退出码 0
  # 纯 bash 行级 AND:先按 "seed": S,(json.dumps 定宽格式)筛行,再按 tag 过滤。
  # 不依赖 python——launcher 的控制流(跳过/双开)可被桩进程完整测试。
  [[ -f $RESULTS ]] || return 1
  grep -E "\"seed\": $1(,|})" "$RESULTS" 2>/dev/null \
    | grep -qF "\"tag\": \"$2\""
}

build_cmd() {  # build_cmd SEED → 全局数组 CMD(两种 mode 共用主干,协议同 v1)
  CMD=("$PY" -m benchmarks.reasoning_depth
       --task pointer_chase --difficulty 16 --n_values 16
       --mode fixed --seeds "$1" --steps "$STEPS" --mix
       --full_mha --n_global_heads 2 --beta2 0.999 --clip 0
       --device "$DEVICE" --tag "$TAG")
  if [[ $MODE == control ]]; then
    CMD+=(--transformer_only)
  else
    CMD+=(--eval_depths $DEPTHS)
    # phase 2 全量重发用:任务可达性佐证已由 control 腿承担时省预算
    [[ ${P0C_SKIP_TRANSFORMER:-} == 1 ]] && CMD+=(--skip_transformer)
  fi
  [[ -n $MID_EVAL ]] && CMD+=(--mid_eval $MID_EVAL)
}

CAFFEINATE=$(command -v caffeinate 2>/dev/null || true)
for seed in "${seeds[@]}"; do
  if seed_done "$seed" "$TAG"; then
    echo "=== $TAG seed $seed 已有 canonical 行,跳过 ($(date '+%F %T')) ===" | tee -a "$LOG"
    continue
  fi
  echo "=== $MODE $TAG seed $seed start $(date '+%F %T') ===" | tee -a "$LOG"
  build_cmd "$seed"
  if [[ -n $CAFFEINATE ]]; then
    "$CAFFEINATE" -is "${CMD[@]}" >> "$LOG" 2>&1 &
  else
    "${CMD[@]}" >> "$LOG" 2>&1 &
  fi
  pid=$!
  echo "$pid" > "$PIDFILE"
  echo "pid=$pid 已记录($PIDFILE),等待该 seed 完成..." >&2
  wait "$pid"
  rc=$?
  echo "=== $TAG seed $seed end rc=$rc $(date '+%F %T') ===" | tee -a "$LOG"
done
rm -f "$PIDFILE"
echo "全部 seed 处理完毕。" >&2
