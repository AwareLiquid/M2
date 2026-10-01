# GOALS.md — 程序计数器(循环元状态,唯一真相源)

> 任何会话(人/cron/agent CLI)打开 M2 工作区:读这里 → 执行 current_action →
> 完成后推进状态并原子提交。规则:一次只有一个 current_action;达成条件
> 必须可机械验证(check_cmd 退出码,禁散文);研究内容不进本文件,
> 详情指针指向对应文档。更新本文件 = 推进程序计数器。
> 编辑前 git diff 查格式化器噪声;提交后 git show 验证落盘;
> 每轮提交前必跑 ./scripts/goal_check --audit(数数锚:条目数=check_cmd 数)。

## 循环细则(AMM-002 自 prompt 下沉,AMM-003 轮次化改订;agent 每轮开场读本区,不靠会话记忆)

- **驱动模型(AMM-003 拉式点火;AMM-004 PR 流;AMM-010 连续循环)**:
  用户点火一次=一个连续循环,循环内由 goal 校验驱动连续轮次(单目标
  =goal_queue 清空或推进至 BLOCKED-HUMAN,点火时可改指),达成或合法
  终止方停。**禁 cron/launchd/定时任务/心跳监听**不变,调度工具一律
  不创建(ignite.sh+agent-cmd.conf 已于轮 11 归档删除,git 可逆);
  循环锁=.loop-lock 每轮合轮 touch 刷新,循环终止删(PARKED 墓碑例外)。
- **锁判读(marathon_guard exit 1 时)**:最后提交晚于锁 mtime(锁后已
  提交=已合轮残留)或 锁龄≥30min+锁后零提交+零活进程(死轮残留)
  ⇒ 删锁接管;锁龄<30min+锁后零提交+有活进程迹象 ⇒ 疑真并行轮,
  停勿双开,报告用户。连续循环内:锁 mtime=每轮合轮 touch 刷新,活循环
  的锁恒新鲜;循环终止删锁;PARKED 例外保留(墓碑,100min 自过期)。
- **状态机**:RUNNING(默认)/ PARKED(唯一非手动自停态:连续 3 次空审计
  轮触发;处置=RSI 夜账补账+快照进本文件+提交合轮,不删 .loop-lock,
  100min 自过期;重入口=用户 Go 点火:读 PARKED 快照+复述停摆原因进
  本轮报告+置回 RUNNING)/ BLOCKED-HUMAN(需人决策)。
- **轮次单元(训练腿跨轮)**:≤30min 可出判读的=当轮发起当轮判读;
  训练腿(小时级)=发起后条目 status 置 doing,后续每轮=查进度/判读/
  落盘,训练在途≠阻塞。blocked_on 禁列在途训练项。长训练腿一律走
  scripts/launch_p0c.sh 式幂等发射器(逐 seed 判重+caffeinate 防睡+
  pidfile 双开拒+逐深度增量落盘;2026-09-28 事故教训:18.5h 随关机
  全损,jsonl 零落盘)。
- **轮次分支阶梯(QUEUE-EMPTY 时依序取活,产出清零空审计计数)**:
  ① 判读后续池迭代(RESULTS/ROADMAP 明示可迭代点;段内同族 ≤2);
  ② 停车场三问解停审计(AMENDMENTS 停车场逐条过"T1/T2 可动?可逆?
     无预注册门槛?",逐项记录依据);
  ③ 写作登记(RESULTS/ROADMAP/HANDOFF 回填);
  ④ 工程硬化(工具缺口/测试加固/**RSI 夜账到期检查**——累计 ≥10 产出
     轮次未入账即属此项有活,防夜账断喂式静默失效);
  ⑤ 全空 ⇒ **空审计**:四项逐项审计依据写进 current_action 后提交,
     合轮(连续循环内由 goal 校验驱动下一轮);连续 3 次(跨轮)⇒ PARKED。
- **算力四档**:T0 本地 CPU / T1 本地加速器(本机 MPS / RTX 5060 8GB,
  视所在机器) / T2 服务器 / T3 Kaggle;T2/T3 发起需用户预先授权;
  资源红线=护机优先,异常即中止。
- **结论分级**:[A]构造保证 / [B]本机实测 / [C]终局声明(须 T3 或跨机
  复现);meta 带 exec_tier 与 seed 数。
- **判单门**:每产出轮次提交前 direction_gate --add --round N
  --direction(四轴耦合) --evidence,再 --check-round N;连续 2 条
  DRIFT ⇒ BLOCKED-HUMAN。
- **蒸馏门(AMM-005;纪律正文=GOAL-PROMPT v4.0,本区只留 GOALS 特有)**:
  current_variable=当前唯一研究变量(单变量;组合变量须在其预注册原子
  声明);换变量须旧变量判读落盘(或显式弃置+原因)后留痕切换。
  例外轮计数(轮 12 补):只认全例外(主判定=例外);半例外(推进为主+
  机制为辅)不断连击;用户明示续作机制工作 ⇒ 重置连击(明示=人裁,
  留痕于当轮 DISTILL);阈值 2 与后果不动。
- **社区蒸馏门(AMM-008)**:问表四栏+颗粒度对齐纪律+触发条件(例行=
  十轮节拍③,定义=RSI-INDEX)+入账四分法=RSI-HORIZON.md 协议区,不在
  本区复述;采纳走 AMENDMENTS。
- **程序计数器瘦身(AMM-009)**:current_action 仅近 2 轮全文+更早轮
  单行索引(全文=git log,永不丢);块高 ≤48 行由测试守护。

```yaml
state: RUNNING            # 轮 17 验收轮:AMM-010 一致性 3 处残留修补完毕,体系一致;下轮起连续循环驱动;d=8 复核臂(pid 40268)在途,出数即终判
mode: ON                  # 循环总开关(OFF ⇒ goal_check 不动作)
current_goal: >-
  M2 循环 v2(AMM-010 单目标连续循环):goal 校验驱动的连续轮次循环,
  单目标=队列清空或推进至 BLOCKED-HUMAN(协议/阶梯/纪律唯一源=
  docs/loop/GOAL-PROMPT-M2.md,本文件不复述)。
current_variable: p0c-prime-depth(思考深度→能力;组合变量=γ配额K=2+full_mha 原子修复包,其余冻结;判据=verdicts/p0c_prime.prereg.v2.json,沿用 v1,no_posthoc_move)
current_action: >-
  [轮次索引:更早轮单行,全文=git log(AMM-009 瘦身,永不丢)]
  轮 1(09-27)循环 bootstrap+验收修复(pop_first 隔条删除 bug)。
  轮 2(09-27)AMM-002 prompt 瘦身 canonical 化+17 条不变式测试。
  轮 3(09-27)P0-C′ 发射:预注册 v1+fixed sweep(T1 MPS)。
  轮 4(09-27)episodic-stream 落地(DATA_FORMS §2)。
  轮 5(09-27)dpo-grpo-wiring 接线(rl 单步路径,默认关)。
  轮 6(10-01)AMM-003 拉式驱动+09-28 事故响应(fsync 增量+幂等发射)。
  轮 7(10-01)AMM-004 PR 流(fork+PR#1)+AMM-005 蒸馏门(goal_check
  全队列+current_variable+DISTILL)+P0-C′ 对照先行(phase_1 chance ⇒
  phase_3 复核发射;前段会话猝死,锁判读取证后接力收编)。
  轮 8(10-01)研究登记:RSI-HORIZON 对标 8 体系(已具备 5/缺口 2)。
  轮 9(10-01)AMM-008 社区蒸馏门(问表四栏)+社区先验(多跳结构性
  障碍+grokking 量级 10^5-10^6 步)喂 P0-C′ 判读链。
  轮 10(10-01)AMM-009 大道至简(GOAL-PROMPT v4.0 承重句 19→21+程序计数器
  瘦身+变动率一行仪表);phase_3 d=1 终评 chance 落盘,d=8 在途。
  轮 11(10-01)清理轮(审计处方打包,零承重句变动);例外连击(10+11)
  触发 BLOCKED-HUMAN,轮 12 用户问询=明示授权解除(半例外不计连击+
  明示续作重置连击,阈值/后果零改动)。
  轮 13(10-01)研究主线判读+事故响应:d=1 终评 M(1,0)=0.066=chance;
  原 d=8 腿与 20:01 并行腿死于 20:22 机器重启(块缓冲假进度;"loss 0.13@
  3600"降级 exploratory);幂等单臂重发 pid 40268,M(1,·) 取 18:31 冻结
  partial 行留痕;清 kill -0 0 永等 watcher;github 443 不可达 PR 推迟。
  轮 14(10-01)工程硬化轮:launcher 双开防护扩展同目录全部 *.pid(两次
  真实绕过事故闭环)+PYTHONUNBUFFERED=1(块缓冲假进度),+2 测试;d=8
  复核臂在途未动;上轮并行轮遗留锁=我方残留,证据链接管删除。
  轮 15(10-01)社区蒸馏门轮(用户点名=触发③):三路检索各 ≥2 独立源
  入 HORIZON #3——①SSM/线性递归状态追踪表达性下界(arXiv 2404.08819;
  M2 full_mha+全局头恰是文献解法方向);②looped transformer 算法任务
  正先验(ICLR 2024+Zhang,附怀疑方复核);③grokking 视界非常数(配方
  wd=0.01+beta2=0.999+clip=0 已在位)。喂终判分叉:d=8 chance ⇒
  budget_wall+架构下界双归因;加预算 10^5≈2 天/臂 10^6≈17 天/臂 ⇒ T2/T3
  议题=人裁;换任务候选=排序类。非例外轮。
  轮 16(10-01)机制修订轮(用户明示点火=修宪授权):AMM-010 单目标连续
  循环 ADOPTED——GOAL-PROMPT v4.0→v4.1 三处改订:①开篇"一次 Go 轮次"
  →"一个连续循环"(点火一次=一个循环,goal 校验驱动连续轮次);②步骤
  3 尾"删锁等点火"→"合轮 touch 刷新循环锁,下一轮由 goal 校验驱动";
  ③终止铁律换形"轮次终止四因①协议走完"→"循环终止四因①单目标达成"
  (单目标=goal_queue 清空或推进至 BLOCKED-HUMAN,点火时可改指);
  "绝不中途弃轮"语义保全(作用域=轮内)。21 承重句 fragment 零改动;
  空转闸(3 空审计⇒PARKED)=连续循环防呆,原样;禁外部调度不变。
  例外轮(机制修订,用户明示 ⇒ 连击重置)。
  轮 17(10-01)验收轮(用户点名):AMM-010 一致性全仓扫描——修 3 处残留
  (阶梯⑤"等待下次点火"/current_goal v1 引用/HANDOFF 驱动段补 AMM-010),
  AMENDMENTS 旧语义=立法史不算不一致;21 fragment 复验绿,交付 v4.1 全文。
blocked_on: >-
  服务器后台训练(nchain 完整 5 seeds/genreplay)=队列执行段在途,
  非人工阻塞(blocked_on 禁列在途训练项);无其他人工阻塞。
next_trigger_hint: goal_check ⇒ 路由(挂起/阶梯/状态机语义见本文件细则区)
pointer: docs/ROADMAP_M2.md; docs/EXPERIMENT_PLAN.md; docs/DATA_FORMS.md;
  docs/TRAINING.md; docs/loop/{GOAL-PROMPT-M2,AMENDMENTS,RSI-INDEX,DISTILL,RSI-HORIZON}.md
updated: 2026-10-01 (轮 17 验收:AMM-010 一致性 3 处修补,体系一致;v4.1 为现行宪法)
```

```yaml
goal_queue:
  - id: p0c-prime-depth-retest
    goal: P0-C′ 思考深度命题复测(ROADMAP §4 P0-C 结论 3 待办)——γ 全局头配额
      + full MHA 修复注意力后的 MT-LNN 上做 fixed-depth sweep(深度 1/2/4/8
      各训全新模型),任务用更难配置(更多跳数/更大图,pointer_chase d16/
      parity 线);预注册判负标准先行落盘,结果登记 ROADMAP P0 实验日志;
      多 seed 纪律(双峰任务报 grok 率,禁单 seed 声明)。
    done_condition: 判决文件 benchmarks/verdicts/p0c_prime.json 存在且含
      h_supported 字段(预注册格式),ROADMAP 已登记判决条目。
    check_cmd: python3 -c "import json; d=json.load(open('benchmarks/verdicts/p0c_prime.json')); assert 'h_supported' in d"
    status: doing
  - id: 2b-120k-leg
    goal: 2B 训练下一腿——60K 步(val PPL 2.556)后继续至 120K 步并登记
      val PPL;附上下文平坦 PPL 判读(128:2.72/256:2.72/512:2.68,长上下文
      无增益,判读数据侧 vs 架构侧归因并给出下一腿决策)。
    done_condition: docs/RESULTS_2B_120K.md 存在且登记 val PPL + 上下文
      PPL 判读 + 下一腿决策。
    check_cmd: test -f docs/RESULTS_2B_120K.md
    status: todo
```
