# M2 Roadmap — 2B 推理引擎:小模型对标 70B 的路线图
# M2 Roadmap — A 2B Reasoning Engine That Competes With 70B on Chosen Axes

> 版本 v1.0 · 2026-07-28 · 战略路线图(实验日志见 `BRAIN_INSPIRED_ROADMAP.md`,基准数据见 `BENCHMARKS.md` / `RESULTS.md`)
>
> **核心命题 / Thesis**:知识外置(RAG),本体只做推理与记忆控制。
> 2B 本体 = 纯推理引擎 + 记忆控制器;在数学/代码推理、长流式记忆、持续学习、端侧延迟四个轴上对标 70B base 模型。
> **不承诺**"全面对标 70B"——知识容量随参数缩放是物理规律,任何对外材料不得写全面超越。

---

## 1. 现状诚实评估(2026-07 实测)/ Honest Baseline

M1 本质:**带生物模块封装的线性递归模型(SSM 家族)**,与 Mamba/RWKV/Griffin 同能力类别。

| 维度 Dimension | 实测 Measured | 定性 Verdict |
|---|---|---|
| PPL(语言建模)| 与同参数 transformer 统计打平(10-seed)| 及格线,非卖点 |
| 推理内存 Inference memory | O(1) 恒定,1M token 处 **8063×** 优势 | 真实差异化 ✓ |
| 跨窗口记忆 Cross-window recall | 0.56 vs transformer 0.00 | 真实差异化 ✓ |
| 不规则采样 Irregular sampling | 退化 +7.7% vs LSTM/GRU +31~33% | 真实差异化 ✓ |
| 训练稳定性 Training stability | ~4× 更稳(loss variance)| 工程优势 ✓ |
| Hebbian 模块 | **实测惰性(inert)**,PPL 贡献 < seed 噪声 | 待改造或砍除 |
| GWT / 预测编码 / 睡眠 | 部分有数据(见实验日志复测五~十二:EWC ✓、经验回放 ✓、生成式做梦重放 ✓ 且 PC 的梦更好)| 贡献部分量化,需补齐 ablation |

**一句话**:M1 现在赢在"效率与记忆形态",不赢在"思考能力"。M2 的全部工作就是补思考能力。

**In one line**: M1 wins on efficiency and memory form-factor today, not on reasoning. M2 exists to close the reasoning gap.

---

## 2. 为什么 2B 有机会在特定轴上打 70B / Why a 2B Can Win on Chosen Axes

三个有确凿外部证据的突破口:

1. **窄域推理可越级(蒸馏 + RL)** — DeepSeek-R1-Distill-Qwen-1.5B 在 AIME 上超过 GPT-4o。路径:强教师推理轨迹蒸馏 → SFT → GRPO(可验证奖励)。
2. **递归深度 = 用计算换参数** — HRM(27M 参数)靠循环递归推理在 ARC-AGI 上打赢大模型;latent recurrent-depth 工作(Geiping et al. 2025)证明同一核心循环 N 次可替代更多层数。**M1 的液体核心天生递归——这是我们架构与该路线的天然契合点,目前未被利用。**
3. **知识外置** — 70B 的大部分参数在"背书"。2B + 检索 + 学习型记忆控制器可卸掉知识负担,把参数全部留给推理回路。

对标声明的纪律:只在【数学/代码推理、长流式记忆(RULER/流式基准)、持续学习(任务链)、端侧延迟/内存】四轴上做 head-to-head;每个对比注明对方是 base 还是 instruct。

---

## 3. 生物模块改造表 / Bio-Module Refit Plan

原则:**功能上像大脑,而不只是命名上像大脑。** 每个模块必须有独立 ablation,贡献 < seed 噪声的模块砍除。

| 模块 | 现状 | 改造方案(神经科学对应 + 有效 ML 技术) | 验收标准 |
|---|---|---|---|
| **Hebbian** | 惰性 | 改成 **fast weights / test-time learning**(DeltaNet、Titans 路线):推理时按 surprise 更新快速权重矩阵,实现"边用边学"。改不动就删,不留装饰品 | 在 recall 任务上 ablation ≥ 2σ 增益,否则删除 |
| **预测编码 PC** | 模块名 + 部分实验信号(做梦重放中 PC 的梦更好,5/5 种子) | 变成**真实训练信号**:逐层预测下一时刻潜变量,预测误差做辅助 loss;**误差大 = surprise = 记忆写入门控**(海马编码触发机制),直接驱动 RAG 写入决策 | 辅助 loss 使主任务收敛更快或更稳(多种子);surprise 门控写入优于均匀写入 |
| **GWT 瓶颈** | 未完全量化 | 真正的**稀疏全局工作区**:k-winner 竞争广播,仅赢家进入全局状态;天然形成可解释"注意焦点" | ablation 证明贡献;广播稀疏度-性能曲线 |
| **睡眠固化** | 概念 + 生成式重放实验 ✓ | **离线重放蒸馏**:空闲期把 episodic buffer(RAG 库高价值条目)蒸馏进权重 + EWC 防遗忘。互补学习系统:**权重=皮层(语义),向量库=海马(情景)** | 任务链持续学习基准上,"睡过"的模型显著优于未睡 |
| **5 时间尺度 τ** | 已有 | 保留,升级为 HRM 式**分层递归**:慢尺度规划、快尺度执行;推理时可变步数循环 + 自适应停机("难题多想几轮") | 思考步数 vs 准确率曲线单调上升 |

改造完成后,"大脑式思考"= 四个可验证机制:**迭代递归推理、surprise 驱动记忆、稀疏全局广播、睡眠期固化**。每个可独立出论文图。

---

## 4. 三阶段计划 / Three Phases

### P0 — 把递归推理证出来(现在,本地 RTX 5060 8GB 可做)

> **结局回填(2026-09-07)**:P0 已执行完毕并出结局 —— A/B 已实现,C 预注册判决判负,D 部分完成。以下勾选框与正文保留立项原文不动,结局以各勾选框下的注记回填(追加式)。

目标:整个 thesis 的最小证据 —— **"多想 = 更准"曲线**。

- [ ] **P0-A** 液体核心加 `thinking_steps` 参数:同一核心循环 N 次做潜空间迭代;训练时随机采样深度(Geiping 式 depth randomization),推理时可变;N=1 严格等价现状(向后兼容)
  - **结局（2026-09-07 回填）**：**已完成**（commit 8cee459;N=1 向后兼容 + Geiping 式深度随机化;落地名 `core_iterations`）
- [ ] **P0-B** 合成推理基准 `benchmarks/reasoning_depth.py`:深度敏感任务(多步算术链 / 奇偶校验 / 多跳推理),transformer 对照组,多种子
  - **结局（2026-09-07 回填）**：**已完成**（`benchmarks/reasoning_depth.py` 已在库,随 commit 8cee459 引入）
- [ ] **P0-C** 训练 + 出图:思考步数 ∈ {1,2,4,8} vs 准确率曲线;若曲线单调上升 → thesis 成立,进 P1
  - **结局（2026-09-07 回填）**：**已执行并判负**——预注册判决 h_supported=false（深度增益 0.0007 < 2σ 阈值 0.0048;诊断 budget_wall;判决任务更换依据 ADJ-001:pointer_chase d8 不可学习 → 换 parity d16,换任务后的检验仍待 GPU 会话）。thesis"多想 = 更准"在已测规模未立住,详见 `RESULTS.md` latent-recursion 裁决条目 + `kb/hypotheses/H001-stack-depth-monotonic.md` verdict log + `benchmarks/results/latent_recursion_verdict.json`
- [ ] **P0-D** 生物模块清理:Hebbian 改 fast-weights 或删;GWT / PC 补 ablation(沿用实验日志的 5-seed 纪律)
  - **结局（2026-09-07 回填）**：**部分完成**——Hebbian 实测惰性已证（`RESULTS.md` O1 module switch-matrix:5 个 bio 模块 PPL 中性、归档至 flags,GWTB/PC 同批覆盖）;删除/改造决断转向 `docs/RESEARCH_PLAN.md` 的观察名单触发机制（PR #33/#35）

> 下一步方向见 `docs/RESEARCH_PLAN.md`（PR #33/#35）;P1/P2/P3 是否重排由下一合并窗口决定（本 PR 不重排,只回填事实）。

**风险与止损**:若 8 步思考对准确率无增益(多种子),说明当前核心的递归不产生有效迭代计算 → 先修核心的状态更新算子(参考 TRM:递归时注入输入、状态残差连接),而不是加大规模。

### P1 — 蒸馏优先,不做预训练(数百美元云预算)

- 教师:开源强推理模型(Qwen3 系列等)生成推理轨迹;学生:350M~1B M1 架构
- SFT + logit 蒸馏(反向 KL),样本效率比预训练高一个数量级——穷人路线里唯一走得通的
- μP(maximal update parametrization)做缩放迁移:50M → 350M 先验证缩放曲线,**不盲跳 2B**
- 评测切换:弃 PPL,换 GSM8K / MATH-500 / ARC / RULER + 自有流式与持续学习基准

### P2 — 2B 混合架构(融资/算力到位后)

- **混合层配比**:纯 SSM 在精确检索上有已知短板(业界共识),液体核心为主 + 少量滑动窗口注意力层(Jamba/Zamba/Samba 证据),保近似 O(1) 同时补 recall
- 后训练:SFT → GRPO(可验证奖励:数学/代码)→ 长度控制
- 记忆控制器上线:surprise 门控读写 RAG,睡眠期蒸馏固化
- 端侧交付:ONNX / WebGPU 路径已验证(见 `benchmarks/export_o1_for_browser.py`)

---

## 4.5 P0 实验日志(诚实记录,含负结果)/ P0 Experiment Log

### 2026-07-28 · 第一轮:anytime 随机深度训练 → 两个负结果,均有明确解释

设置:2 层 MT-LNN(203K)vs ModernCausalTransformer(247K),3 seeds × 3000 步,
loss 仅在答案位;MT-LNN 训练时每步随机采样深度 1..8,评估深度 ∈ {1,2,4,8}。

| 任务 | 结果 | 诊断 |
|---|---|---|
| pointer_chase k=2/k=4(16 节点) | 双方 loss 均钉死在 ln(16)=2.77,acc=随机;**连 k=1 纯查表也学不会**(6000 步) | 不是深度问题、不是架构问题(transformer 同败)。每样本全新随机置换 → 无法记忆,必须学会上下文查表算法 → 处于 grokking 平台期。破法:更小图 / 更长训练 / 课程学习 |
| mod_chain k=8 | 双方部分学会(mt_lnn 0.26,transformer 0.29,随机 0.10);**mt_lnn 各深度完全同分** | **随机深度训练教会模型"无视迭代"**:深度不变解是最容易的最优化路径,反馈门保持 0。Geiping/HRM 均未用朴素随机深度——前者用截断反传+泊松采样,后者用逐迭代深监督 |

**决策**:P0 主声明改为 HRM 式 fixed-depth sweep——每个深度 d 训练一个全新模型
(参数量相同,权重绑定迭代),d 越大解得越好即命题成立。anytime(一套权重任意深度)
降级为 P0 之后的进阶目标,需要深监督/截断反传才有希望。
`benchmarks/reasoning_depth.py --mode fixed` 已实现。

### 2026-07-28 · 第二轮:fixed-depth sweep → 两个更硬的负结果

| 实验 | 结果 | 含义 |
|---|---|---|
| mod_chain k=8 fixed sweep(深度 1/2/4/8 各训一个全新模型,6000 步 × 3 seeds) | 0.289 / 0.297 / 0.293 / 0.293 —— **深度全平,差异在噪声内**;transformer 0.302 仍领先 | 即使固定深度训练+评估,只循环 LNN 子层也买不来能力 |
| pointer_chase 破平台探针(8 节点 k=2,12000 步) | **transformer 破平台:0.9932(基本解决,loss 0.07)**;MT-LNN depth1=0.2494、depth4=0.2498,loss 从 2000 步起钉死 2.0(已收敛,非训练不足) | 任务在此规模可学(transformer 证明);**MT-LNN 的混合栈没学会,且液体核心多迭代 4 次毫无帮助**。上下文关系查找的计算发生在注意力里;疑似 LNN 子层在阻碍注意力形成归纳头 |

**核心教训**:"只循环液体核心 ≠ 思考"。推理任务需要的组合查找由注意力承担,
把迭代范围限制在 LNN 子层是在错误的部件上加深度。

### 2026-07-28 · 第三轮:消融阶梯 → 两个嫌疑人都排除

同探针任务(8 节点 k=2,12000 步,transformer 对照 0.9932):

| 消融 | acc | 结论 |
|---|---|---|
| `--no_scan`(关液体递归,LNN→gated FFN) | 0.2889 | 液体递归**不是**主要阻碍(仅 +0.04) |
| `--n_layers 4`(注意力层翻倍) | 0.2488 | **不是**容量问题(零帮助) |

排除法指向 **MicrotubuleAttention 本身**。代码检查发现:GTP-cap 距离衰减偏置
`-γ_h·(i-j)` 对每头强制生效,4 头 γ init = [0.8, 0.2, 0.05, 0.0125](ALiBi 式
几何序列)——距离 25 处 3 个头的惩罚达 1.25~20 nats,只有 1 个头准全局。
归纳头需要两层注意力组合,可用远视头太少。另一嫌疑:GQA(n_kv_heads=2)。

### 2026-07-28 · 第四轮:GTP 衰减假说证实 + GQA 协同效应

同探针任务(8 节点 k=2,12000 步,seed 0):

| 变体 | loss@12k | acc | 结论 |
|---|---|---|---|
| 基线 MT-LNN | 2.0(钉死) | 0.249 | — |
| `--gamma_init 0.001`(全头全局) | 1.59(**仍在降**) | 0.360 | **GTP 衰减确认是阻断器** |
| `--full_mha`(仅关 GQA) | 2.02(钉死) | 0.248 | GQA 单独无罪 |
| γ + full MHA | **1.14(仍在降)** | **0.557** | **协同**:头能看远后 KV 多样性才起作用 |
| transformer 对照 | 0.07 | 0.993 | — |

**M2 架构原则 #1(P0 的第一个正面产出)**:生物启发的 GTP-cap 距离衰减把
4 头中 3 头的远视力干掉,而归纳头(上下文查表的基础回路)需要两层远视注意力
组合。修复不是删生物机制,而是 ALiBi 式**全局头配额**:每层至少 K 个头
γ≈0(真全局),其余头保留生物衰减。混合架构中"少量注意力层"必须是全局层。

### 2026-07-28 · 第五轮:结案 — γ+MHA 30k 步 **acc = 1.0000**

γ 全局化 + full MHA 的 MT-LNN 在 30k 步**完美解决**探针任务(1.0000),
反超 transformer 对照(0.9932)。**完全解释成立**:M1 的上下文关系推理
能力被 GTP 衰减 init(主因)+ GQA(协同)完全扼杀,修复后无残余劣势——
液体子层不拖后腿。(no_scan 对照收敛轨迹一致,确认液体核心无摩擦。)

**P0-C 结论**:
1. "只循环液体核心 = 思考"被证伪——组合查找的计算在注意力里
2. 真正的产出是**架构原则 #1(全局头配额)**,一行配置的修复,恢复推理满血
3. 思考深度命题需要在"注意力已修复"的模型上重新检验(P0-C′,已做,见下第七轮判决):
   修好注意力后,更难的任务(更多跳数/更大图)上迭代深度是否开始起作用
   → **判决 h_supported=false(2026-10-02,详见 §4.5 第七轮)**

### 2026-08-01 · 第六轮:复现危机 — 固定难度探针在 grokking 掷硬币区,单 seed 结论全体作废

| 数据点 | 配置(kv=2, 30k 步, 8 节点 k=2 固定难度) | acc |
|---|---|---|
| Kaggle seed 0 | g0(全衰减 init) | 1.0000 |
| **本地 seed 1 / 2(复现)** | 同上 | **0.1832 / 0.1652(≈随机)** |
| Kaggle seed 0 | g2(配额 2 全局头) | 0.179 |

**结论**:任务在 30k 步呈双峰(要么解出要么随机),单 seed 的"满分 vs 随机"
对比是 Bernoulli 抽样噪声。**第四、五轮的 γ/MHA 消融结论与 Kaggle 反例互相
不矛盾,但全部降级为"未裁决"**——它们测的很可能是"到达 grokking 的概率/时间"
而非能力有无。第 2 条(GTP 衰减=阻断器)相应降级为假设。

**新协议(裁决中)**:单环 + mix 课程任务(在 1/1 本地运行中于 28k 步可靠
grok 到 loss=0)+ 每配置 ≥3 seeds + per-k 评估。首批:Kaggle kernel
`m1-gqa-quota-replication-g0`(g0×3 seeds)+ 本地 g2。方法论教训入档:
**双峰任务上必须报告 grok 率(n seeds 中解出几个),禁止报告单 seed 准确率**。

### 2026-10-02 · 第七轮:P0-C′ 思考深度复测判决(γ+MHA 修复后,pointer_chase d16 fixed-depth)——**判负,诊断 budget_wall**

预注册 `benchmarks/verdicts/p0c_prime.prereg.v2.json`(v2 判据逐字沿用 v1,no_posthoc_move);判决文件 `benchmarks/verdicts/p0c_prime.json`(h_supported=false)。

| 读数(口径=mean over k∈{1..15},排 k=16 复制捷径桶) | M | grok(≥0.90) |
|---|---|---|
| d=1(seed 0,18:31 冻结 partial 判例) | 0.0660 = chance | 0/1 |
| d=8(seed 0,40268 复核臂 30000 步 rc=0) | **0.0656 = chance** | 0/1 |
| gain_s0 | **−0.0004**(判据 A 需 ≥0.02) | — |
| transformer 对照(30k 步) | 0.0632 = chance | 0/1 |

mid_eval 三读数一致 chance:mid@10000=0.0632 / mid@20000=0.0658 / final=0.0656。

**判决**:h_supported=false(A 失败;B 单 seed 结构性不可裁决;C 无 grok 可言)。**诊断 budget_wall**(prereg negative_diagnoses 双条件兑现:grok_rate(8)=0 ∧ 对照 grok 率=0)——pointer_chase d16/n16 在 30k 步预算对 MT-LNN(γ 配额 K=2+full_mha 修复后)与 transformer 对照均不可达,深度效应免谈;**非 thesis 反证**。约束:单 seed 探针级([B] 本机实测;grok_rate 分母 3 不满足,双峰任务纪律=n/1 透明计数)。

**后续分叉(另立预注册,不得现场改判据;待人裁)**:①加预算——T1 MPS 实测 0.67 step/s:10^5 步≈2 天/臂、10^6 步≈17 天/臂(RSI-HORIZON #3 sizing:grokking 视界文献先验 10^5-10^6 步,当前预算低一个量级以下);②换任务——排序类(轮 15 检索候选);③2b-120k-leg 三选另案。

### 2026-10-03 · 第八轮:p0c-sort 排序类深度探针判决(γ+MHA 修复后,bubble_trace fixed-depth d=1 vs d=8)——**判负,诊断 flat_iterations_ignored**

预注册 `benchmarks/verdicts/p0c_sort.prereg.json`(AMM-017 预算版:本机 ≤30min/发射,S=2200 等步数配对,3 seeds 分次发射;判据先行中程提交 0c7fbf1,no_posthoc_move);判决文件 `benchmarks/verdicts/p0c_sort.json`(h_supported=false)。

任务 bubble_trace(轮 350 新增,`gen_bubble_trace`):k=16 值从 {0..31} 池无放回抽取按随机排列呈现,执行 j~U{1..7} 轮左→右相邻比较交换(升序)后读争议区位置 p 的值;池化标签把占据者秩偏斜(海龟/兔子,实测 max mass 0.23)卷进超几何弥散——per-j best-constant floor 0.035..0.083(mean 0.055)实测钉死,边际捷径被设计杀死;秩→标签查找=O(1) 注意力(深度平坦),不混淆深度对照。

| 读数(口径=mean over j∈{1..7} per-j 桶,每桶 8×256) | M(1) | M(8) | gain |
|---|---|---|---|
| seed 0 / 1 / 2 | 0.6568 / 0.6652 / 0.6738 | 0.6512 / 0.6835 / 0.6469 | −0.0056 / +0.0183 / −0.0269 |
| **3-seed 均值** | **0.6653** | **0.6605** | **mean_gain=−0.0047**(σ_paired=0.0226) |

**判决**:h_supported=false(A/B/C 三判据全败:A mean_gain=−0.0047<0.02;B −0.0047<2σ=0.0452;C highj_gain=−0.0035<0)。**诊断 flat_iterations_ignored**(prereg negative_diagnoses 双条件兑现:M(1)、M(8) 均≫chance 线 0.09 且 |mean_gain|<0.02)——任务在该预算**可学**(双臂 0.66-0.75)但**迭代被无视**:2 层电路(每层注意力+core+FFN)满预算下即可逼近 j≤7 比较链动力学,与预注册 calibration_disclosure 的定 j 形态探索(d1≈d8≈0.82)机制一致;**非 thesis 反证亦非 budget_wall**。约束:[B] 本机实测 3 seeds 探针级;30k 步级终判=愿望登记挂账。

**后续分叉(另立预注册,愿望登记不停车,AMM-017)**:①升级 j/k 预算(更深比较链/更长数组,深度信号在"可学 j 前沿"之外找;30k 步级=愿望登记既有挂账);②换任务族(S5 词问题=NC¹ 完全分离任务已在代码库,理论shortcut最硬;looped 文献配方);③设计教训入档:定 j 配置的深度敏感性论证会输给万能逼近,深度信号要靠难度轴+课程混合显形,且标定必须镜像判读协议等步数双臂跑满 S(轮 350 截断 d1 预算假信号教训)。

### 2026-10-03 · 第九轮:p0c-sort-stack 深度旋钮对照判决(整块迭代含注意力 vs core,stack fixed-depth d=1 vs d=8)——**判负,诊断 budget_wall(深堆叠优化失败)**

预注册 `benchmarks/verdicts/p0c_sort_stack.prereg.json`(单变量=深度旋钮 core→stack,余逐字沿用第八轮;S=2000,3 seeds 分次发射;判据先行 fefc67b);判决文件 `benchmarks/verdicts/p0c_sort_stack.json`(h_supported=false)。

机制假设(第八轮判负的归因):core 迭代只重复 LNN 递归子层不重复注意力,而比较链子步=跨位置交换需注意力;stack 整块迭代 d=8 提供 8 次注意力深度 ≥ j=7 链所需子步数,若旋钮错配是唯一障碍,stack 应显形深度信号。

| 读数(口径同第八轮:mean over j∈{1..7}) | M(1) | M(8) | gain |
|---|---|---|---|
| seed 0 / 1 / 2 | 0.6327 / 0.6535 / 0.6396 | 0.0887 / 0.0584 / 0.0984 | −0.544 / −0.595 / −0.541 |
| **3-seed 均值** | **0.6419(可学)** | **0.0818(chance 带)** | **−0.5601**(σ=0.0304) |

**判决**:h_supported=false(A/B/C 全败)。**诊断 budget_wall(深堆叠优化失败)**:stack d8 在 S=2000 内完全不训练(loss 平台 ~3.2),同任务 stack d1 与 core 双臂均 0.64-0.75——「深核可训练、深堆叠不可训练」的不对称。判据字面偏离如实登记:seed 2 M(8)=0.0984 超出 budget_wall 条款 0.09 线 0.0084(判据未动,no_posthoc_move 遵守;诊断按实质登记,任何读法结论相同)。

**深度命题的探针级总结论(第七+八+九轮合成)**:γ+MHA 修复后,深度信号在本机 30min 探针预算内三个方向均未显形——①pointer_chase d16:预算墙(任务不可达);②bubble_trace core 旋钮:迭代被无视(任务可学,机制=core 不重复注意力);③bubble_trace stack 旋钮:深堆叠不训练(可训练性墙)。**非 thesis 反证**:三判负各自排除了一条路径,深度-能力命题收敛为「深度旋钮参数化与任务结构/预算的三重错配」;升级路径(30k 步级终判/深监督变体/S5 任务族)=愿望登记或另立预注册,不虚报不硬凑。

**后续分叉(愿望登记不停车,AMM-017)**:①深监督变体探针(HRM 式逐迭代 CE,库内 train_model 已有机制,直接攻"深堆叠不训练",≤30min 可推导);②30k 步级终判预算(愿望登记既有挂账);③S5 词问题任务族(NC¹ 分离,理论 shortcut 最硬)。

### 2026-10-03 · 第十轮:p0c-sort-stack-ds 深监督变体判决(stack d8+逐迭代 CE,阶段 1 可达性门)——**h_supported=false,诊断 direction_but_underpowered(部分救活)**

预注册 `benchmarks/verdicts/p0c_sort_stack_ds.prereg.json`(单变量=deep_supervision 开;对照臂复用在盘第九轮 d8 行不新发腿;仅 d8 单臂 S=2000;判据先行 9d897be);判决文件 `benchmarks/verdicts/p0c_sort_stack_ds.json`。

机制假设(第九轮判负归因):深堆叠缺可学习梯度通路;deep_supervision(HRM 式逐迭代 CE)给每个 stack 迭代直接监督。

| 读数(口径同前:mean over j∈{1..7}) | M(8)+ds(per seed) | 对照 M(8) 无 ds |
|---|---|---|
| seed 0 / 1 / 2 | 0.1423 / 0.1254 / 0.1070(逐 j 单调爬升,j7=0.15-0.24) | 0.0887 / 0.0584 / 0.0984(chance 带,loss 平台) |
| **3-seed 均值** | **0.1249**(3/3 > chance 线 0.09) | **0.0818** |

**判决**:h_supported=false(A 失败:0.1249<0.30;B/C 过:效应量 +0.0431≥0.02 且 3/3 seed 一致超 chance)。**诊断 direction_but_underpowered**(prereg 明文)——deep_supervision 使深堆叠从**完全不训练**变为**稳定部分训练**(梯度通路假设方向性兑现),但 S=2000 内量级远低于 d1 可学水平 0.64;剩余差距=预算/量级问题而非有无问题。

**阶段 2(另立预注册)**:深度对照 d1+ds vs d8+ds(对照臂复用本轮在盘行,只发 d1+ds 3 腿,短腿)——在"ds 都起飞"的新平面上裁决深度信号方向。

### 2026-10-03 · 第十一轮:p0c-sort-ds-depth 阶段 2 深度对照判决(d1+ds vs d8+ds)——**h_supported=false,诊断 depth_hurts(深度单调伤)**

预注册 `benchmarks/verdicts/p0c_sort_ds_depth.prereg.json`(单变量=深度 d8→d1,两臂均 ds 开;对照臂复用第十轮 d8+ds 在盘行;新发 d1+ds 3 短腿 S=2000;判据先行 3f88b4a);判决文件 `benchmarks/verdicts/p0c_sort_ds_depth.json`。

| 读数(口径同前:mean over j∈{1..7}) | M(1)+ds(per seed) | M(8)+ds(在盘) | gain(per seed) |
|---|---|---|---|
| seed 0 / 1 / 2 | 0.6332 / 0.6557 / 0.6375 | 0.1423 / 0.1254 / 0.1070 | −0.491 / −0.530 / −0.531 |
| **3-seed 均值** | **0.6421** | **0.1249** | **−0.5172**(σ=0.0229,3/3 全负) |

**判决**:h_supported=false(A/B/C 全败)。**诊断 depth_hurts**(prereg 明文)——ds 平面上深度单调伤:d1+ds 学到 0.6421(与 d1 无 ds 同水平=**ds 无害性成立**,预注册 honest_prediction 两结论之一),d8+ds 仅 0.1249;深度信号在本探针预算内不但不存在而且方向为负。honest_prediction(轮 358 预判)如实兑现。

**深度命题探针级链条收官(§4.5 第七-十一轮,五连负全路径)**:①pointer_chase d16(core)=budget_wall;②bubble_trace(core)=flat_iterations_ignored;③bubble_trace(stack)=深堆叠不训练;④stack+ds 阶段 1=direction_but_underpowered(部分救活);⑤ds 深度对照阶段 2=depth_hurts。**本机 30min 探针预算内深度增益全路径未显形且方向为负;非 thesis 终局反证**(升级预算 30k/10^5 步级=愿望登记挂账;S5 词问题 NC¹ 完全分离+looped 文献正先验=唯一剩余正先验角落,可推导性待四栏蒸馏裁决)。

### 2026-10-03 · 第十二轮:p0c-s5-depth S5 词问题深度探针判决(NC¹ 正先验角落)——**h_supported=false,诊断 budget_wall_s5(深度链全路径收官)**

预注册 `benchmarks/verdicts/p0c_s5_depth.prereg.json`(唯一变量=深度,stack+ds d={1,8} 同腿配对 ×3 seeds,S=2000,判据先行 5dd6e2f);判决文件 `benchmarks/verdicts/p0c_s5_depth.json`。

| 读数(chance=1/120≈0.0083,线 0.02) | M(1) per seed | M(8) per seed |
|---|---|---|
| seed 0 / 1 / 2 | 0.0076 / 0.0090 / 0.0072 | 0.0076 / 0.0080 / 0.0070 |
| **3-seed 均值** | **0.0079 = chance** | **0.0076 = chance** |

**判决**:h_supported=false(A/B/C 全败,mean_gain=−0.0004≈0)。**诊断 budget_wall_s5**(prereg 明文,honest_prediction 如实兑现)——NC¹ 完全分离任务在探针预算不可达:文献 looped 正先验的配方域(长训练+特定课程)远超本机 30min 预算。事故留痕:首发两腿 rc=1 崩=轮 350 make_generator vocab 回归(数据前修复 a618425+自测 vocab 覆盖守卫,canonical 零污染)。

**深度命题探针级总结论(§4.5 第七-十二轮,六连负全路径)**:pointer_chase 预算墙 → bubble_trace core 迭代无视 → stack 深堆叠不训练 → ds 部分救活 → ds 平面深度单调伤 → S5 正先验角落预算墙。**本机 30min 探针预算内,思考深度→能力命题在全部可推导路径(2 任务族×2 旋钮×2 监督制+理论最强任务)上均为负;非 thesis 终局反证**——探针预算(2k 步)与文献配方域(30k-10^5+ 步)差 1-2 个量级,升级预算=愿望登记挂账(30k 步级排序腿+10^5 步级 S5,凭证/资源到位由用户点火改指)。

### 2026-10-03 · 第十三轮:p0c-stream-len 长流式记忆长度外推探针判决(换轴后首探针)——**h_supported=false,诊断 task_unreachable**

预注册 `benchmarks/verdicts/p0c_stream_len.prereg.json`(换轴四栏推导落地:32 条键互异 KV 事实流式回忆,流长由填充密度变化[间隔 6/30/127],短训 T=256 长测 T∈{1024,4096};MT-LNN vs transformer 同 max_seq_len;判据先行 a4f4847);判决文件 `benchmarks/verdicts/p0c_stream_len.json`。

**判决**:h_supported=false(A 失败:**任务本身在 2000 步预算不可学**——两模型训练长度 acc:mtlnn 0.0625=chance / transformer 0.1563<0.8 门,外推问题未触及)。**诊断 task_unreachable**(prereg 明文)——与 ROADMAP 第一轮"连 k=1 纯查表 6000 步都学不会"同现象族:in-context 查表类任务在本架构规模需要更长训练。处置=换任务参数另立预注册(N_FACTS=8 易变体:保留回忆距离测试)或加步数愿望登记。

**事故留痕**:mtlnn 三腿 T=4096 评估 MPS OOM(global_coherence 稀疏分数 4GB 峰值)——自适应 batch(长 T 减至 4)+长度间清缓存修复,seed 0/1 重发 rc=0,canonical 零污染。延迟描述性报告:两模型延迟均超线性(mtlnn 4096/256 比 15.3,transformer 16.4)——**MT-LNN 的 O(1) 主张只覆盖液体子层,混合架构的注意力层仍是 O(T²)**(架构诚实边界再次自证)。

**长流式记忆轴现状**:探针基础设施已建(streaming_recall.py+自测黄金回放),任务参数量级是当前瓶颈(32 事实×2000 步太难);易变体(N_FACTS=8)已入队。

### 2026-10-04 · 第十四轮:p0c-stream-len-e8 易变体判决(N_FACTS=8)——**h_supported=false,诊断 still_unreachable(流式轴探针级预算墙,收官)**

预注册 `benchmarks/verdicts/p0c_stream_len_e8.prereg.json`(单变量=事实数 32→8,余逐字沿用;判据先行 2ac2ff1);判决文件 `benchmarks/verdicts/p0c_stream_len_e8.json`。

| 读数(chance=0.0625) | M(1) per seed | acc(256) 3-seed 均值 | R(1024/256 保持率) |
|---|---|---|---|
| mtlnn | 0.141 / 0.25 / 0.281 | **0.2239**(< 0.8 门) | 0.313 |
| transformer | 0.312 / 0.344 / 0.359 | **0.3386**(< 0.8 门) | 0.671 |

**判决**:h_supported=false(A 再次失败)。**诊断 still_unreachable**(prereg 明文)——8 事实(容量降 4×)在 2000 步仍不可学,与第一轮(k=1 查表 6000 步)+第十三轮(32 事实)构成三连证据:**in-context 查表类任务在本架构/本预算下学习速度是数量级级瓶颈,流式轴探针级预算墙成立**。加步数(8000 步 mtlnn 腿≈35min 超红线)=愿望登记挂账;流式轴段内同族迭代已达上限 2(32/8),进一步降事实数不再推导。

**端侧延迟描述性证据(两轮一致)**:同宽下 mtlnn 绝对延迟比 transformer 慢 ~8×,延迟长度比均超线性(mtlnn 15.4×/transformer 16.4×)——**MT-LNN 的 O(1) 主张只覆盖液体子层,混合架构含注意力层**;端侧延迟轴候选探针=纯液体核(attention_layers=())延迟平坦性,可推导。

### 2026-10-04 · 第十五轮:p0c-latency 端侧延迟探针判决(纯液体核 vs hybrid vs transformer,零训练 T 扫描)——**h_supported=true(循环首个正判决:O(1) 主张在液体子层独立成立)**

预注册 `benchmarks/verdicts/p0c_latency.prereg.json`(换轴第三轴首探针;零训练纯推理,单扫描 median-of-3 无训练随机性如实声明;判据先行 3f65436,amendment 1 数据前 113d81b);判决文件 `benchmarks/verdicts/p0c_latency.json`。

| 前向延迟 batch=1(s) | T=512 | T=2048 | T=8192 | T=16384 | G=L(16384)/L(512) |
|---|---|---|---|---|---|
| **纯液体核**(关注意力+关 global_coherence) | 0.0170 | 0.0457 | 0.1840 | **0.4314** | **25.4×(≈线性)** |
| hybrid(默认注意力) | 0.0223 | 0.1015 | 1.0169 | **OOM** | — |
| transformer | 0.0064 | 0.0133 | 0.2025 | 6.7345 | **1052.9×(二次)** |

**判决**:h_supported=**true**(A/B/C 全过:pure 4/4 点完成;25.4 ≤ 0.5×1052.9;25.4 ≤ 32 线性带)。**O(1) 端侧主张在液体子层独立成立**——16384 长流 0.43s/请求、272MB 状态;transformer 同扫描 1053× 增长已入二次段。事故留痕:首扫 pure_liquid 16384 OOM=use_global_coherence 默认开(global_coherence topk T×T 4GB)——infra_oom 预注册诊断兑现,数据前修复(113d81b)重跑,行留账本。

**架构诚实边界(三轴探针链的共同产出)**:①O(1) 只覆盖液体子层——hybrid/transformer 的注意力与 global_coherence 均为 T×T 二次,端侧部署须限制注意力层或换稀疏/线性注意力;②小 T 段纯液体核绝对延迟慢于 transformer ~3×(逐 token 循环开销),O(1) 优势只在长流兑现;③流式记忆轴的学习瓶颈(第十四轮)与延迟轴的正判决相互独立——能力与延迟是两个待分别论证的主张。



## 5. 评测纪律 / Evaluation Discipline

- **不再以 PPL 为主指标**(打平已证,无增量信息)
- 主指标:GSM8K、MATH-500、ARC、RULER(长上下文)、自有流式记忆与任务链持续学习基准
- 一切声明 ≥ 5 seeds,报均值 ± 标准差;负结果照记(沿用 `BRAIN_INSPIRED_ROADMAP.md` 的诚实记录传统)
- 对外表述:只说四轴对标,注明对手模型的具体版本与 base/instruct 状态

## 6. 关联文档 / Related Docs

- `BRAIN_INSPIRED_ROADMAP.md` — 类脑机制实验日志(复测一~十二,EWC/重放/做梦数据)
- `BENCHMARKS.md` / `RESULTS.md` — 当前诚实基准
- `ABLATIONS.md` — 消融记录
- `docs/PRODUCT_LINES.md` — 产品线定位

---

### English Summary

**Thesis**: knowledge lives in RAG; the 2B model is a pure reasoning engine + learned memory controller. We target parity with 70B *base* models on four axes only: math/code reasoning, long-stream memory, continual learning, and edge latency/memory — never "overall parity."

**Why possible**: (1) narrow-domain reasoning can leapfrog via distillation + RL (R1-Distill-1.5B > GPT-4o on AIME); (2) recurrent depth trades compute for parameters (HRM 27M on ARC-AGI; latent recurrent-depth scaling) — and M1's liquid core is *natively recurrent*, an unused structural advantage; (3) offloading knowledge to retrieval frees parameters for reasoning circuitry.

**Bio-modules become verifiable mechanisms**: iterative latent reasoning (variable thinking steps), surprise-gated memory writes (predictive-coding error as the hippocampal write trigger), sparse global broadcast (k-winner GWT), and sleep-phase consolidation (offline replay distillation + EWC). Any module whose ablation gain is below seed noise gets cut.

**Phases**: P0 (now, local 8GB GPU) — prove the "think longer → more accurate" curve on depth-sensitive synthetic tasks; P1 (small cloud budget) — distillation-first at 350M–1B with μP scaling checks, no from-scratch pretraining; P2 (funded) — 2B hybrid (liquid core + sparse sliding-window attention), GRPO post-training, memory controller, edge delivery via ONNX/WebGPU.
