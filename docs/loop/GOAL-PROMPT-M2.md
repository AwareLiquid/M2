# GOAL-PROMPT-M2 v4.8(2026-10-02,AMM-017 资源=蒸馏约束;唯一 canonical 点火源)

> 架构(AMM-015/016 对齐领域共识"prompt 薄、机制厚、经验走注入管道",
> HORIZON #5 四源):本文件=点火火花塞(循环契约+三步协议+指针);
> 24 条铁律=docs/loop/IRON-LAWS.md(测试守护,每轮读回);操作细节
> (VERDICT 语义/合轮门禁链/蒸馏四栏格式/锁判读/阶梯/算力红线)=
> GOALS.md 细则区+脚本头注(AMM-016 自围栏下沉,机械执行=goal_check/
> loop_closer/direction_gate);AMM-017 资源=蒸馏约束非停车条件(只
> 推导本机 ≤30min 探针可执行方向,超预算愿望登记不停车);驱动=RSI
> 机器。立法史与版本一览=AMENDMENTS.md(AMM-001..017);旧版全文=git。

```text
/goal 本会话=一个连续循环(拉式点火:用户点火一次=一个循环,循环由
goal 校验驱动连续轮次,单目标=goal_queue 清空(队列由续向蒸馏续填:
方向由四栏蒸馏推导且须本机 ≤30min 探针可执行,人不在方向环;超预算
的推导愿望登记不入队不停车;蒸馏推导不出且队列空方停,点火时用户
可改指);禁 cron/定时/心跳监听;轮次自包含,开局即查新状况,不依赖
会话记忆。
开场 ./scripts/marathon_guard(判读=GOALS.md 细则区"锁判读")。

每轮协议(细则与语义=GOALS.md 细则区+脚本头注;铁律=IRON-LAWS.md):
1. 三查(只读盘):GOALS 状态/队列+铁律 → 训练腿进程 → 增量;读
   DISTILL 尾部 2 条+distill_inject 检索注入——经验必须被下一轮读到。
2. ./scripts/goal_check → 按 VERDICT 行动(达成弹出含深位;无可执行
   项走阶梯+续向蒸馏,只推导 ≤30min 可执行方向,超预算留痕跳过,
   不停;推导不出且队列空 ⇒ 单目标达成停)。
3. ./scripts/loop_closer.sh 合轮(门禁链+DISTILL 四栏+提交+推 fork+
   刷锁);红即 ABORT,不提交不合轮。

铁律=docs/loop/IRON-LAWS.md(每条一次真实事故,测试守护,每轮读回);
细则=GOALS.md;蒸馏=docs/loop/DISTILL.md;立法=AMENDMENTS.md。
```
