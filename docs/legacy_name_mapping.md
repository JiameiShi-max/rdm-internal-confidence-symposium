# Legacy Name Mapping

内部实现还保留少量历史文件名，这是为了不冒险破坏已经跑通和测试过的 import path。对外汇报时使用右侧的概念名。

| Legacy file or dataset | Public meaning |
| --- | --- |
| `generate_stage9_internal_teacher_dataset.py` | internal-confidence teacher dataset exporter |
| `sure_target_stage9_internal_readout.py` | supervised internal-readout student trainer |
| `sure_target_stage1_targeted_dr.py` | base RDM sure-target task implementation |
| `stage8_extended_analysis.py` | legacy helper functions used by the current task code |
| `stage9_internal_teacher_dataset_50k_auto.npz` | original-timing internal teacher dataset |
| `stage9b_timed_teacher_dataset_50k_fixed_auto.npz` | controlled timed fixed internal teacher dataset |

公开 release 中的数据集已重命名：

- `data/original_internal_teacher_dataset_50k.npz`
- `data/timed_internal_teacher_dataset_fixed_50k.npz`
