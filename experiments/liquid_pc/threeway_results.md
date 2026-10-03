# 三对照回放（A→B→C 持续学习）—— RESEARCH_RECOVERY 登记实验的结果

`experiments/liquid_pc/threeway.py`，3 模式（真实/生成式/无回放）×
PC/GRU × 5 seeds × 3 任务 × 40 epochs，同 buffer（16）同预算。
预注册主指标：worst_final（全系统是否仍可用）、mean_forget、c_final（新任务）。

| 臂 | worst_final | mean_forget | c_final |
|---|---|---|---|
| none/PC | 0.7625 ± 0.0182 | 0.3720 ± 0.0096 | 0.0067 |
| none/GRU | 1.0645 ± 0.1217 | 0.5078 ± 0.0605 | 0.0087 |
| real/PC | 0.0519 ± 0.0040 | 0.0098 ± 0.0015 | 0.0112 |
| real/GRU | 0.0497 ± 0.0027 | −0.0016 ± 0.0048 | 0.0113 |
| **dream/PC** | **0.0433 ± 0.0034** | 0.0186 ± 0.0028 | 0.0126 |
| dream/GRU | 0.0504 ± 0.0051 | 0.0085 ± 0.0026 | 0.0153 |

对照（同模型内模式差，worst_final）：
- PC：none−real = 0.711，none−dream = 0.719，**dream−real = −0.0086**
- GRU：none−real = 1.015，none−dream = 1.014，dream−real = +0.0007

**结论（诚实）**：
1. **回放是关键机制**：none 臂灾难遗忘（worst 0.76/1.06），任何回放都把
   最差任务误差降到 ~0.05（15-20×）。
2. **PC 的生成式梦全场最优**：dream/PC 0.0433 是最低 worst_final，且
   **PC 的 dream 胜 real（−0.0086）**，而 GRU 的 dream ≈ real（+0.0007）。
   这是"PC 的显式生成结构梦得更好"主张的**第一个正向证据**——方向一致、
   跨 5 seeds，但幅度小（±0.005 量级），定性为弱支持。
3. 新任务质量 c_final 各臂 0.007-0.015，回放代价小。
