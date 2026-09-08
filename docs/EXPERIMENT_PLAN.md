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
