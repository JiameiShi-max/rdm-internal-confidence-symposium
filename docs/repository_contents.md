# 仓库内容说明

这个发布包只保留 symposium 汇报需要的内容。

保留内容：

- `src/`：当前模型机制的可复现实现，包括 controlled timed task、internal-confidence teacher dataset、student readout。
- `analysis/`：教授要求的行为分析、population diagnostics、teacher calibration、matched-trial control、multiseed 和 parameter validation。
- `data/`：两个小型 teacher dataset，用于复现实验入口；timed fixed dataset 是主模型数据。
- `results/`：当前可汇报的表格、JSON summary 和 figure suite。
- `tests/`：保护关键分析和导出逻辑，避免协作时改坏。

刻意不放进公开包的内容：

- 早期 stage5/stage7 RL 探索测试和结果。
- broken timed dataset：`stage9b_timed_teacher_dataset_50k_auto.npz`。
- smoke dataset、缓存、历史扫描图和本地调试输出。

价值判断：

- 必要：behavior、population R²、pre-TS leakage、repeated-stimulus stability、teacher calibration、matched controls。
- 有辅助价值：multiseed、small parameter validation，用来说明主模型选择不是任意的。
- 不适合放在主叙事：早期 RL 尝试，因为当前项目主线是 supervised internal-confidence proof-of-concept。
