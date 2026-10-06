# JevBench 官方评测 — Run 1 (2026-10-07)

**目的**：用 **JevBench 官方 harness**（benchmarkheaven.com 决策模型基准，MIT
仓库 `fstandhartinger/jevbench`）测我们的模型，产出**行业标准口径**的分数，
不用自写评分。

## 管线（已打通）

```
我们的模型（M2-2B SFT） ← /v1/systemone 原生线格式服务（m2_training/jev_service.py）
        ↓ native 概率分布（对精确 label 集）
官方 CLI: python -m jevbench.cli run --adapter typesafe --endpoint ... \
          --tasks datasets/public/{easy,original}.jsonl
官方 summarize → 官方指标（accuracy/ECE/Brier/latency/schema-validity）
```

- **native 语义**：官方把"读模型自己的概率分布"标为 native（vs openai_compat 的
  verbalized）；我们暴露的即模型对 label 集的自身分布（每 label 序列似然 softmax），
  符合 native 类 —— **官方明示 token-level logprobs 不作为 harness 接口**，
  我们的分布由模型原生计算。
- **官方评分要求**：probabilities 键=labels 精确匹配、和为 1±1e-3、
  choice 答案须带 `choice` 字段（被选 label）——已全部满足（schema_validity 1.0）。

## 结果（v2 检查点 = M2-2B SFT Run 2, seq512/rope4；官方 harness 实测）

| 指标 | easy (48) | original (72) | 合计 (120) |
|---|---|---|---|
| raw accuracy | **0.333**（16/48） | **0.375**（27/72） | **0.358**（43/120） |
| ECE（top-label） | **0.500** | **0.336** | — |
| Brier | 1.045 | 0.943 | — |
| p50 latency | 0.92 s | 0.71 s | — |
| schema validity | 1.00 | 1.00 | 1.00 |
| charged | $0.0000 | $0.0000 | — |

per-family 最好/最差：fact 6/12、adequacy 6/12、extraction 6/12、policy 6/12；
最差 intent 2-3/12、ordinal 3/12。

## 诚实读数（对照官方公式）

1. **机会校正 Intelligence ≈ 10-20**（5 选项层 chance=0.2：0.333→0.167；
   二选项层 chance=0.5 → 负）——**远低于 50 门**（榜上 I=49-66）。
   现行 checkpoint（未做任何决策型训练）离榜还远。
2. **校准差（ECE 0.34-0.50，严重过度自信）**——已知修法：温度缩放
   （我们在 banking77 上做过 ECE 0.476→0.127）。
3. **速度中游**：p50 0.7-0.9s（轴值≈56，因为我们对每个 label 各做一次前向）。
   榜上本地系统 p50 0.01-0.23s。**架构正解 = O1-Flash 式单前向决策头**
   （一次前向出全选项分布 → 毫秒级 → Speed 轴 85-100）。
4. 有趣的对照：我们**零决策训练**的 raw accuracy 36-38%，恰好落在
   榜单头部系统**封卷层**的 34-38% 邻域——公开层的差距 = 他们对公开分布的
   优化（Plumb-4B 公开 89.6% vs 封卷 38.0%）。

## 下一步（优先级）

1. **O1-Flash 式决策头**：单前向 → 原生分布（速度/成本轴从 56 → 85-100）；
2. **决策型训练数据**（不是 alpaca 指令，而是 state+rubric→label 格式），
   目标机会校正 I ≥ 50；
3. **温度缩放**上标定（Calibration 轴从 0.34-0.50 ECE 修到 ~0.1 量级）；
4. 用 **v2b**（训练中）重跑官方 harness 对比；
5. 达标后考虑正式提交（开放权重 + 服务命令，由维护者自测；
   或 API 形态）。

产物：`/root/jev_runs/{v2_easy2,v2_orig}/`（官方 results/raw/ledger/summary）；
服务 `m2_training/jev_service.py`。
