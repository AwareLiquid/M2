"""tests/test_distill_gate.py -- AMM-005 蒸馏门的账本纪律(仓库态测试)。

DISTILL.md 是经验蒸馏账本(追加式禁改历史),GOALS.md 的
current_variable 是单变量锚点。本测试守护两件不靠自觉的事:
  1. 账本存在且每个轮次条目四栏齐全(现状/问题/有效经验/变量判定);
  2. GOALS.md 声明了唯一的 current_variable(goal_check --audit 亦机械
     校验,此处为文档层双保险)。
蒸馏只在轮内发生(拉式,零定时任务)——无脚本可测其无,由 GOAL-PROMPT
铁律与不变式测试守护。
"""
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DISTILL = os.path.join(ROOT, "docs", "loop", "DISTILL.md")
GOALS = os.path.join(ROOT, "docs", "loop", "GOALS.md")

FOUR_FIELDS = ("现状", "问题", "有效经验", "变量判定")


def distill_entries():
    src = open(DISTILL, encoding="utf-8").read()
    blocks = re.findall(r"(?m)^### (轮 .+?)$(.*?)(?=^### |\Z)", src, re.S)
    assert blocks, "DISTILL.md 缺少 '### 轮 N' 条目"
    return blocks


def test_distill_ledger_four_fields_per_entry():
    for title, body in distill_entries():
        for field in FOUR_FIELDS:
            assert f"**{field}**" in body or f"{field}:" in body, \
                f"DISTILL 条目[{title}] 缺四栏之一: {field}"


def test_distill_append_only_no_revision_markers():
    src = open(DISTILL, encoding="utf-8").read()
    assert "~~" not in src and "已删除" not in src, \
        "蒸馏账本禁改历史:出现删改痕迹"


def test_goals_current_variable_declared_and_atomic():
    src = open(GOALS, encoding="utf-8").read()
    m = re.search(r"(?m)^current_variable:[ \t]*(\S.*?)\s*$", src)
    assert m, "GOALS.md 缺 current_variable(蒸馏门锚点,AMM-005)"
    assert len(re.findall(r"(?m)^current_variable:", src)) == 1, \
        "current_variable 只允许一个(单变量纪律)"


HORIZON = os.path.join(ROOT, "docs", "loop", "RSI-HORIZON.md")


def test_horizon_community_gate_protocol_present():
    """AMM-008 社区蒸馏门:问表四栏协议必须常驻 HORIZON(缺=机制退化)。"""
    src = open(HORIZON, encoding="utf-8").read()
    assert "社区蒸馏门协议" in src, "HORIZON 缺社区蒸馏门协议区(AMM-008)"
    for field in ("现状", "问题", "目标", "检索颗粒度"):
        assert field in src, f"问表缺栏: {field}(用户规格,AMM-008 依据)"
    for rule in ("独立来源", "拒绝也留痕"):
        assert rule in src, f"社区蒸馏门纪律缺条款: {rule}"


def test_horizon_entries_carry_query_card():
    """每条社区获取条目必须带问表+判定(无问表的检索=手工检索,不入账)。"""
    src = open(HORIZON, encoding="utf-8").read()
    blocks = re.split(r"(?m)^### #\d+", src)[1:]
    assert blocks, "社区获取条目缺席(AMM-008 落地轮应有首批)"
    for b in blocks:
        body = b.split("\n## ")[0]
        assert "**问表**" in body, "条目缺问表(检索前必填,AMM-008)"
        assert "**判定**" in body, "条目缺判定(四分法,拒绝也留痕)"


def test_goals_current_action_bounded():
    """程序计数器瘦身(AMM-009):current_action 块 ≤48 行——近 2 轮
    全文+更早轮单行;历史全文=git。块高反弹 ⇒ 本测试红,逼下一轮压缩。"""
    src = open(GOALS, encoding="utf-8").read()
    m = re.search(r"(?m)^current_action: >-\n(.*?)(?=^blocked_on:)", src, re.S)
    assert m, "current_action 块缺失"
    assert len(m.group(1).splitlines()) <= 48, \
        f"current_action 块 {len(m.group(1).splitlines())} 行 >48(AMM-009 瘦身纪律:压缩更早轮为单行)"
