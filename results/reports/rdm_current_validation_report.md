# RDM internal-confidence 当前验证汇总

## Single model 对照
| model | n | sure offered | sure choice/offered | residual corr | residual R2 | evidence R2 P(sure) | pre-TS corr | deterministic stability | population peak diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_single_seed | 400 | 0.802 | 0.576 | 0.718 | 0.516 | 0.278 | 0.457 | 1.000 | 0.197 |
| timed_single_seed | 400 | 0.495 | 0.505 | 0.533 | 0.285 | 0.473 | 0.306 | 1.000 | 0.061 |

## Multi-seed 稳健性
| model | seeds | strict pass | residual corr mean | residual corr std | residual corr min | residual R2 mean | pre-TS corr mean | pre-TS corr max | deterministic stability min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| timed_5seed | 5 | False | 0.599 | 0.049 | 0.531 | 0.361 | 0.298 | 0.567 | 1.000 |

## 汇报口径

- timed 版作为主模型：timing 更严格，sure offered 接近 50%，适合承载主要结论。
- original 版作为对照：sure-axis signal 更强，但 timing/sure availability confound 更多。
- 当前 timed 5-seed 显示 residual sure signal 稳定；pre-TS leakage 有 seed 依赖，应作为限制和后续优化点报告。
