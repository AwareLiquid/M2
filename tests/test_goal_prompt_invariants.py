"""tests/test_goal_prompt_invariants.py -- load-bearing iron laws of the canonical prompt.

Every iron law in docs/loop/GOAL-PROMPT-M2.md's ```` ```text ```` fence is a
sentence whose deletion has caused a real incident somewhere (source: Physic
AMM-018/AMM-019/AMM-036 lineage; M2 轮 1-2). Any edit to the canonical prompt
must keep all of them -- this test is the mechanical gate (先例: v4.0 重写删掉
"中途不停"导致轮 93 停止). Change a law deliberately => change this test in
the same commit via an AMENDMENTS proposal, never silently.

AMM-003 (2026-10-01): drive model switched to pull-based Go rounds; the
marathon law "绝不主动结束回合" was reshaped (not deleted) into
"绝不中途弃轮" — rounds end by design (commit = round closure), but the
in-round protocol must complete. This is the only fragment ever changed,
via proposal, with this test updated in the same commit.
"""
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANONICAL = os.path.join(ROOT, "docs", "loop", "GOAL-PROMPT-M2.md")
ARCHIVED = os.path.join(ROOT, "docs", "loop", "GOAL-PROMPT-M2-v1-ARCHIVED.md")


def text_fence():
    src = open(CANONICAL, encoding="utf-8").read()
    blocks = re.findall(r"```text\n(.*?)```", src, re.S)
    assert blocks, "canonical prompt 缺少 ```text 围栏(点火提取将失败)"
    return blocks[0]


def test_single_text_fence():
    src = open(CANONICAL, encoding="utf-8").read()
    assert len(re.findall(r"```text", src)) == 1, "canonical 只允许一个 text 围栏"


def test_fence_is_substantial():
    assert len(text_fence().strip().splitlines()) >= 15, "prompt 被过度削减"


# 承重铁律:每条的删除都对应一次真实事故
IRON_LAWS = [
    # (承重句片段, 事故来源)
    ("marathon_guard", "双开防护(锁新鲜即 BUSY)"),
    ("goal_check", "心跳路由本体"),
    ("--audit", "数数锚:GOALS 被格式化器吞行(Physic 轮 136/138/142)"),
    ("显式", "禁把管道尾巴退出码当门禁(Physic set -e 事故)"),
    ("绝不中途弃轮", "Go 轮次合轮纪律(AMM-003 换形):提交即合轮、协议未走完不得停——原'绝不主动结束回合'(Physic 轮 409/410/AMM-036 挂起停摆)的拉式等价物"),
    ("连续 3 次空审计", "PARKED 上限:防空转(AMM-035)"),
    ("快照进", "上下文耗尽/挂起前状态落盘"),
    (".loop-lock", "PARKED 不删锁;锁=轮内互斥(AMM-003:合轮删锁)"),
    ("只读盘", "恢复不依赖会话记忆"),
    ("git show", "编辑 GOALS 后落盘验证"),
    ("预注册", "判负标准先行(ROADMAP P0 先例)"),
    ("负结果", "负结果与正结果同等记录"),
    ("grok 率", "双峰任务禁单 seed 声明(M2 第六轮复现危机)"),
    ("3 seeds", "机制对比多种子下限"),
    ("PPL 非主指标", "四轴对标纪律(ROADMAP §5)"),
    ("BLOCKED-HUMAN", "需人决策即停"),
    ("AMENDMENTS 提案", "机制改动不自改宪法"),
    # AMM-005 (2026-10-01): distillation gate
    ("全队列检测", "蒸馏门:每轮全量 check_cmd,深位达成即弹出(轮 4/5 已完成条目滞留队列的教训)"),
    ("经验蒸馏", "四栏入 DISTILL.md 往 current_variable 单变量收敛(6 轮 K=0 的弥散教训)"),
    # AMM-009 (2026-10-01): 大道至简版,蒸馏门全量进宪法
    ("current_variable", "单变量锚点进铁律(AMM-009:读回环+四栏+单变量+问表全量入宪法)"),
    ("问表", "社区取经先填问表(AMM-008/009:现状/问题/目标/检索颗粒度)"),
]


@pytest.mark.parametrize("fragment,reason", IRON_LAWS)
def test_iron_law_present(fragment, reason):
    assert fragment in text_fence(), f"承重铁律丢失: {fragment!r}({reason})"


def test_archive_marked_deprecated():
    """废止副本必须带横幅,杜绝'canonical 停旧版'式静默失效(Physic AMM-037)。"""
    if not os.path.exists(ARCHIVED):
        pytest.skip("无 v1 归档")
    head = open(ARCHIVED, encoding="utf-8").read()[:400]
    assert "已废止" in head and "禁止用于点火" in head
