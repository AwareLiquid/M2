> **已废止(2026-09-27,AMM-002):本文件仅为立法史存档,禁止用于点火。**
> 唯一 canonical=docs/loop/GOAL-PROMPT-M2.md(v2 瘦身版)。

# GOAL-PROMPT-M2 v1.0(2026-09-27,循环体系 bootstrap;移植自 AwareLiquid-Physic GOAL-PROMPT-v8/AMM-035 语义)

> 用法:重启马拉松时把下方 ```text 围栏全文粘贴给 agent CLI(zcode -p / claude -p 等);
> `scripts/ignite.sh` 自动提取本围栏注入 `docs/loop/agent-cmd.conf` 配置的命令。
> 修订纪律:机制改动只走 AMENDMENTS.md 提案,不自改本文件正文(AMM-001)。

```text
/goal 按 docs/loop/GOALS.md 的 goal_queue 持续自循环,本会话即马拉松。
心跳节奏:有活连跑,无活按冷却语义收束——每心跳运行 ./scripts/goal_check,
严格按其 VERDICT 与退出码行动;启动时先 ./scripts/marathon_guard(exit 1=已有
活马拉松,确认状态即结束)。**产出心跳**(队列迭代/实验发起/训练判读/写作
登记/硬化有行动)必须产出可验收成果:pytest 全绿+./scripts/goal_check --audit
过+./scripts/direction_gate --check-round 本轮轮号 过(一律显式退出码判定,
禁把管道尾巴退出码当门禁),原子提交 main 后立即进入下一心跳。挂起心跳
(阶梯⑤)的产出=阶梯空审计记录(写入 GOALS,可验收),同样过门禁提交。
**挂起心跳绝不结束回合**——回合终止即会话终止,循环停摆;挂起=提交完空
审计记录后回合内继续下一心跳(Physic 轮 410 复盘教训,v8.1 语义)。

## 会话结束(仅两因,与心跳节奏无关)
1. 手动停止;
2. 上下文真耗尽(自动压缩后仍无法维持工作记忆:快照进 GOALS 后结束,
   重开粘贴本 prompt 续跑,快照收束不删 .loop-lock)。
除此之外会话不结束;挂起不是停止,是冷却。

## 状态机
- RUNNING(默认):心跳循环中。
- PARKED(唯一非手动自停态):同一会话内阶梯⑤连续 3 次空审计(零产出
  心跳,任何产出心跳清零计数)⇒ PARKED:快照进 GOALS+结束会话,不删
  .loop-lock(100min 自过期);重入口=用户指令/新目标,一句话即重启。
- BLOCKED-HUMAN:需人决策的事项(预注册标准改动/算力档位变更/路线重排)。

## 心跳单元(训练腿跨心跳——M2 与探针型项目的关键差异)
- 队列目标按算力档分两类:T0/T1 内 ≤30min 可出判读的=当轮发起当轮判读;
  训练腿(小时级,本地 GPU/服务器/Kaggle)=发起后把条目 status 置 doing,
  后续心跳=查进度/取结果/判读/落盘登记,训练在途≠阻塞。
- 每个实验目标必须预注册判负标准先行(格式沿 ROADMAP P0 先例:判决文件
  benchmarks/verdicts/<id>.json 含 h_supported 字段);负结果与正结果同等
  记录登记。blocked_on 清单禁列在途训练项与 PR 项。

## 状态驱动恢复
会话级状态在 GOALS.md;任务级状态在队列条目 status 字段。恢复永远=读盘
(marathon_guard→goal_check),不依赖任何会话记忆或散文注记。编辑 GOALS
前先 git diff 查格式化器噪声,提交后 git show 验证落盘,数数锚
(./scripts/goal_check --audit)每轮必查。

## 心跳分支(细则以 goal_check 输出为准)
0 ACHIEVED=弹出晋升 / 1 NOT-Achieved=对队首迭代一步 / 2 QUEUE-EMPTY=按
自主工作阶梯取活(依序;每级产出都计入行动产出并重置挂起计数):
  ① 判读后续池迭代(RESULTS/ROADMAP/实验日志中明示的可迭代点;
     段内同族 ≤2,防同族空转);
  ② 停车场三问解停审计(AMENDMENTS 停车场逐条过"T1/T2 可动?可逆?
     无预注册门槛?"——三者皆否才可继续挂起,审计本身=产出,逐项记录入 GOALS);
  ③ 写作登记(RESULTS/ROADMAP/HANDOFF 回填,实验日志诚实记录含负结果);
  ④ 工程硬化(工具缺口/测试加固/门禁补强/RSI 夜账);
  ⑤ 以上四项全部为空(逐项记录审计依据入 GOALS)⇒ 挂起心跳:提交空审计
     记录,挂起计数 +1;计数满 3 ⇒ 转 PARKED(不再无限挂起)。

## 启动/收尾
启动:./scripts/marathon_guard,exit 1 ⇒ 已有马拉松(锁 <100min),确认状态
即结束;锁自过期后重启自然畅通;STALE-HINT=疑似死锁残留,人工证据链确认后
可接管删除 .loop-lock。收口=模式切换而非会话结束:mode OFF 用 GOALS.md
yaml 块的 mode 字段(=循环总开关,goal_check/ignite 均拒动),手动停止照旧。
RSI 入账时机:每累计约 10 个产出心跳、或转 PARKED、或上下文收束前,
按 docs/loop/RSI-INDEX.md 口径补账(夜账=追加式条目,机械维度脚本出草稿,
K/T 终判人裁)。

## RSI 体系(指标定义唯一源=docs/loop/RSI-INDEX.md)
要点:K=重大结论数(机制/定量/终局判定,含负结果);T=跨 seed/跨机独立复现
成功率(M2 最痛一课:grokking 复现危机使单 seed 结论全体作废——T 是本项目
最重要指标);D=预注册执行率+多 seed 纪律遵守率+坑复发数;EXP=证据轮占比,
<20% 连续 2 窗⇒冻结新方向只做实验与判读。"刷高可见集≠改进"——只有对未见
数据/新 seed 的一次性迁移才算数。

## 纪律
- 算力四档:T0=本地 CPU / T1=本地 GPU(RTX 5060 8GB) / T2=服务器 /
  T3=Kaggle。T2/T3 会话发起需用户预先授权;服务器在途训练=队列执行段,
  禁重复发起同类训练。
- 资源红线:本地 GPU 显存与 CPU 占用异常即中止,护机优先。
- 结论分级:[A]构造保证 / [B]本机实测 / [C]终局声明;[C] 须 T3 级证据
  或跨机复现;meta 带 exec_tier 与 seed 数。
- 多 seed 纪律:双峰/grokking 任务报 grok 率(n seeds 中解出几个),
  禁止单 seed 准确率声明;一切机制对比 ≥3 seeds。
- 对标纪律(ROADMAP §5):只在数学/代码推理、长流式记忆、持续学习、
  端侧延迟/内存四轴上做 head-to-head;禁"全面对标 70B"表述;
  PPL 非主指标(打平已证,无增量信息)。
- 每轮提交前向 docs/loop/direction-gate.jsonl 追加判单
  (./scripts/direction_gate --add --round N --direction … --evidence …)
  并 --check-round N;direction 必答与四轴目标的耦合;连续 2 条 DRIFT ⇒
  state: BLOCKED-HUMAN。
- 预注册判负标准先行;机制改动只走 AMENDMENTS 提案;需人决策 ⇒
  state: BLOCKED-HUMAN 停止;卡两轮或上下文真耗尽 ⇒ 快照进 GOALS 后重开续跑。

## 项目配置
分支:main 直接原子提交(无 PR 层,与 Physic 的 wave/loop+PR 模型不同)。
不提交 .pt/.pth(已 gitignore)、数据本体、benchmarks/results/(gitignore);
判决文件放 benchmarks/verdicts/(可跟踪)。
真相源:ROADMAP_M2/EXPERIMENT_PLAN/RESULTS_*=研究事实,GOALS.md=元状态,
RSI-INDEX.md=指数,AMENDMENTS.md=治理,HANDOFF.md=人读摘要(每轮追加一行)。
```

## 与 Physic 版的差异清单(AMM-001 附件)

| 项 | Physic(来源) | M2(本版) | 理由 |
|---|---|---|---|
| 队列达成验证 | check_cmd 退出码 | 同(不变) | 机械验证是体系核心 |
| 分支模型 | wave/loop 集成线+dir/<slug>+PR 合并人工 | main 直接提交 | M2 单人仓,无 PR 审查层 |
| 心跳粒度 | T1 探针 ≤30min 当轮闭环 | 训练腿跨心跳(status: doing) | M2 实验是小时级 |
| 退出码 | 0/1/2/3/4(含 MINING-FROZEN/DEBT-FIRST) | 0/1/2/5(EXP/欠账门后补,先人裁) | 门禁渐进硬化 |
| 判单载体 | direction-gate.jsonl(方向漂移) | 同,加四轴耦合声明 | 对齐 ROADMAP 对标纪律 |
| 价值出口 | 六门(蒸馏向) | 多 seed/预注册/四轴对标/PPL 非主指标 | M2 是训练型项目 |
