# GOAL-PROMPT-M2 v4.1(2026-10-01,AMM-010 连续循环版;唯一 canonical 点火源)

> 蒸馏门全量进本文件(AMM-009:读回环=步骤 1,四栏入账=步骤 3,单变量
> +社区问表=铁律);AMM-010 单目标连续循环(goal 校验驱动轮次迭代);
> 立法史与语义保全链=AMENDMENTS.md(AMM-001..010);
> 旧版全文=git。承重句由 tests/test_goal_prompt_invariants.py 机械守护
> (21 条),修改本文件前先读该测试。细则读盘不背:GOALS.md 细则区
> (状态机/轮次阶梯/锁判读/训练腿跨轮/算力四档/结论分级)。

```text
/goal 本会话=一个连续循环(拉式点火:用户点火一次=一个循环,循环由
goal 校验驱动连续轮次,单目标达成或合法终止方停;禁 cron/定时/心跳
监听不变;轮次自包含,开局即查新状况,不依赖会话记忆)。
单目标=goal_queue 清空或推进至 BLOCKED-HUMAN(点火时用户可改指);
轮次=循环内的迭代单元。
开场 ./scripts/marathon_guard:exit 0=畅通;exit 1=锁在,按 GOALS.md
细则区"锁判读"证据链处置(死轮残留=删锁接管;疑真并行轮=停勿双开)。

每轮协议:
1. 三查(只读盘):GOALS 状态/队列 → 训练腿进程(pid 文件+ps)→ 增量
   (训练日志尾+git status/log,未推送提交清点);并读 DISTILL.md 尾部
   2 条——写下的经验必须被下一轮读到(蒸馏门读回环)。
2. ./scripts/goal_check:全队列检测目标达成(达成即弹出,含深位),按
   VERDICT 行动——0=队首已弹出,继续下一位;1=对队首迭代一步(训练
   在途≠阻塞);2=队列空,按 GOALS.md 阶梯取活;5=MODE-OFF 停。
3. 合轮收尾(显式判定退出码,禁把管道尾巴退出码当门禁):pytest 全绿
   + goal_check --audit 过 + direction_gate --add 本轮判单(四轴耦合)
   后 --check-round 本轮过 + DISTILL.md 追加四栏蒸馏条目(现状/问题/
   有效经验/变量判定;例外轮显式原因,连续 2 个例外轮 ⇒ BLOCKED-HUMAN)
  ⇒ 原子提交 main ⇒ 推 fork 分支+PR(AMM-004,禁直推远端 main)
  ⇒ 合轮(touch .loop-lock 刷新循环锁;训练在途≠阻塞,下一轮由
  goal 校验驱动开动,不靠定时)。

铁律(不变式测试守护,禁删改):
- 循环终止仅四因:①单目标达成;②手动停;③上下文真耗尽;④PARKED
  (连续 3 次空审计轮)——③④前必快照进 GOALS;PARKED 不删 .loop-lock;
  终止时删 .loop-lock(PARKED 除外)并报告。轮次=循环内迭代单元,
  走完步骤 1-3 即合轮开下一轮;轮内绝不中途弃轮。
- 恢复只读盘不读会话记忆;GOALS 编辑前 git diff 查噪声,提交后
  git show 验证落盘。
- 实验预注册判负标准先行(benchmarks/verdicts/<id>.json);负结果与
  正结果同等记录。
- 双峰任务报 grok 率;机制对比探针 ≥3 seeds,对外声明 ≥5 seeds;
  禁单 seed 声明。
- 只在四轴(数学/代码推理、长流式记忆、持续学习、端侧延迟/内存)
  head-to-head 对标;PPL 非主指标。
- 需人决策 ⇒ BLOCKED-HUMAN;机制改动只走 AMENDMENTS 提案,不自改宪法。
- 每轮全队列检测目标达成;经验蒸馏四栏入 DISTILL.md,往
  current_variable 单变量收敛(组合变量须原子声明);社区取经先填问表
  (现状/问题/目标/检索颗粒度,协议=RSI-HORIZON.md);蒸馏只在轮内,
  禁定时任务监听。
```
