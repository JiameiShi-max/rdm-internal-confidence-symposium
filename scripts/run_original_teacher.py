import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from generate_stage9_internal_teacher_dataset import main


DEFAULT_ARGS = [
    '--training-iters',
    '50000',
    '--loss-epoch',
    '1000',
    '--n-batch',
    '50',
    '--n-batches',
    '20',
    '--output',
    'data/original_internal_teacher_dataset_50k.npz',
    '--summary',
    'results/reports/original_teacher_dataset_summary.json',
]


if __name__ == "__main__":
    main(DEFAULT_ARGS + sys.argv[1:])
