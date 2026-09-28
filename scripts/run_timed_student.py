import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from sure_target_stage9_internal_readout import main


DEFAULT_ARGS = [
    '--dataset',
    'data/model_freeze/timed_teacher_variable_ts_seed7.npz',
    '--training-iters',
    '50000',
    '--loss-epoch',
    '1000',
    '--n-batch',
    '50',
    '--n-eval-batches',
    '8',
    '--summary',
    'results/model_freeze/student_seed7_summary.json',
]


if __name__ == "__main__":
    main(DEFAULT_ARGS + sys.argv[1:])
