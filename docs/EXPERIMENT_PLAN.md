# M2 算力利用与实验计划(2026-09-09)

服务器:A100 40GB(36.140.149.133),租期至 2026-10-08。
研究方向:融合式预测编码液态核 + 生成式重放 + 选择性状态转移(COMPONENT_AUDIT 决策)。

## 资源分工

| 资源 | 负载 | 任务 |
|---|---|---|
| **GPU A100** | 78%(baseline 训练中) | MTLNNModel 机制对比(batch=128 修复后的真实分化) |
| **CPU 8 核** | ~2.6 核(nchain) | PCLiquidCore 系列(逐 t 循环,CPU-bound) |
| 磁盘 | 6.7G 可用 | GPU checkpoint 走 /tmp(用完即删),结果 json 落盘 |

## 队列 1(GPU):MTLNNModel 机制对比

- 目的:batch=128 修复后,medium 规模(17.7M)下各机制在 pointer_chase 的真实分化
  (此前 batch=8 时全部随机,batch=128 后 probe 已见分化,medium 需确认)
- 设计:13 experiment × 3 seeds(0,1,2)× medium × pointer_chase × 5000 步
- 预计:~9-10 小时;结果 → `/root/gpu_exp_results.log`
- 决策门:若 medium 上某机制 accuracy 稳定高于 baseline(3 seeds 全胜)→ 该机制候选有效

## 队列 2(CPU):PCLiquidCore 防遗忘

- nchain(5 seeds×5 tasks×40 epochs,运行中):
  seed0 初步:PC 遗忘 +1.02 vs GRU +0.66(与 M1 报告相反,5 seeds 后定论)
- genreplay(排队):PC-astro(无回放)/ PC-replay(真实)/ PC-genreplay(生成)对照
- 决策门:PC 在"最差任务 MSE"或"遗忘"上 ≥5 次配对优势(对齐 M1 的 E3 证据)才支持候选

## 队列 3(算力余量,排队):mod_chain / 更大规模

- GPU 队列 1 完成后:13 experiment × mod_chain(顺序任务对照)
- 若 2b(2.06B)能在 A100 40GB 训练(FP32 状态 33GB 需谨慎评估)→ 单独规划

## 实验纪律(继承审计)

- 同参数同预算;每机制独立记账;不互借结论
- 不自动启用"已判负"旧开关(snap/STE/Hebbian 正则等)
- 历史可复用结果不重训;新结果须多 seed + 配对

## 实验结果登记(2026-09-09 三队列完成)

### 队列 1:probe 机制对比(pointer_chase,5000 步,batch=128,3 seeds)

| experiment | seeds | avg | 稳定性 |
|---|---|---|---|
| hamiltonian_world_model | .98/.98/.98 | **0.980** | 唯一稳定学会 |
| fast_weight_memory | .78/.27/1.0 | 0.685 | 高方差 |
| baseline | .71/.28/1.0 | 0.661 | 高方差 |
| latent_core | .37/.37/.99 | 0.578 | 高方差 |
| world_model | .89/.38/.41 | 0.559 | 高方差 |
| top_down | .53/.99/.15 | 0.554 | 高方差 |
| competitive_workspace | .98/.42/.15 | 0.515 | 高方差 |
| selective_state | .19/.96/.15 | 0.432 | 高方差 |
| global_rhythm | .36/.15/.13 | 0.212 | 低-中 |
| predictive_coding | .16/.16/.17 | 0.166 | 稳定低 |
| global_coherence | .16/.15/.17 | 0.159 | 稳定低 |
| latent_stack | .13/.15/.18 | 0.152 | 稳定低 |
| workspace | .14/.15/.14 | 0.143 | 稳定低 |

观察:① hamiltonian 是唯一 3-seed 稳定机制;② 高方差主导(碰巧学会 vs 没学会),机制差异被 seed 方差淹没 → 该协议需更多 seeds/更长训练才能可靠区分;③ "稳定低组"(workspace/latent_stack/predictive_coding/global_coherence 0.14-0.17)在 pointer_chase 上稳定学不会(≈随机 0.125)。

### 队列 2:PCLiquidCore(5 seeds)

- **nchain**:PC 最差任务 0.965 vs GRU 1.327(4/5 配对);mean_forget 3/5
  → worst-task 稳定性优势复现(对齐 M1 E3)
- **genreplay**:重放消除遗忘(forget 无回放 +1.2 → 真实回放 +0.006 / 生成 +0.022);
  PC-gen 4/5 优于 GRU-gen;PC-gen 略逊真实回放(+0.016)

### 结论与后续

- hamiltonian_world_model 0.98 是唯一稳定信号,值得深挖(是否 Hamiltonian 辅助稳定了训练)
- 降方差方案:加 seeds(5-10)/ 降 lr / 加长训练至收敛;或 medium + 更长步数验证 top 机制
- mod_chain 对照与 2b 评估排队(算力余量时)

### mod_chain 对照队列(2026-09-09,probe,5000 步,batch=128,3 seeds)

几乎所有机制 accuracy = 1.0(baseline/selective_state/latent_*/workspace/predictive_coding/
world_model/hamiltonian/global_*/top_down 全 1.0;fast_weight 0.994、competitive 0.994、
global_rhythm 0.992 仅个 seed 略低)。

结论:mod_chain(顺序累积任务)对 probe 规模过简单 → **天花板效应**,无法区分机制;
价值 = 确认所有接入机制在简单顺序任务上都能学会(无机制损坏)。
区分机制须用 pointer_chase 类"查表/推理"任务(见队列 1:hamiltonian 稳定 0.98、
workspace/latent_stack/predictive_coding/global_coherence 稳定 ~0.15)。

### 降 lr 验证(2026-09-09,pointer_chase,lr=1e-4,4 机制 × 5 seeds)

| 机制 | 5-seed 值 | avg |
|---|---|---|
| hamiltonian_world_model | .42/.23/.33/.93/.77 | 0.536 |
| fast_weight_memory | .91/.32/.32/.19/.79 | 0.505 |
| baseline | .90/.29/.31/.16/.78 | 0.489 |
| workspace | .16/.99/.28/.28/.13 | 0.367 |

**关键结论:单 lr + 5000 步的 probe 机制对比不可靠。**
lr=3e-4 时 hamiltonian 稳定 0.98、workspace 稳定 0.14;降 lr=1e-4 后 hamiltonian
变高方差(0.23-0.93)、workspace 部分 seed 到 0.99——机制排名随 lr 翻转,seed 方差主导。
⇒ 区分机制需 lr 扫描/训练至收敛/更大模型,或换更敏感的任务与指标。

### medium 加长训练验证(2026-09-09,medium 17.7M,batch=128,pointer_chase,15000 步)

| 机制 | loss 轨迹 | accuracy |
|---|---|---|
| baseline | 1.95→1.81(缓慢下降) | 0.219(没学会) |
| **hamiltonian_world_model** | 1.95→1.63→0.06→**0.005**(9000 步后突降) | **1.0**(学会) |

**medium 规模澄清了 probe 的假象**:hamiltonian_world_model 在 medium 下稳定学会
pointer_chase(查表任务),baseline 学不会。结合 probe 高方差,结论是 medium + 加长训练
是可靠对比协议,hamiltonian 的机制优势需多 seed 确认(见下)。

### medium 多 seed 确认(2026-09-09,medium,15000 步,batch=128)

| 机制 | seeds(medium_long + medium_ms) |
|---|---|
| baseline | 0.219(seed0)/ **0.998**(seed1)/ **1.0**(seed2) |
| hamiltonian_world_model | 1.0(seed0)/ seed1-2 待完成 |

**关键反证:medium 下 baseline 也能学会 pointer_chase(2/3 seeds = 0.998/1.0)**。
此前 "hamiltonian 1.0 vs baseline 0.219" 是 **seed 假象**(baseline seed0 碰巧未收敛)。
⇒ medium + 15000 步 + batch=128 下 pointer_chase 可学会(高方差,seed0 例外),机制对比
须多 seed;hamiltonian 无明显机制优势(样本内)。
