import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from generate_stage9_internal_teacher_dataset import main


DEFAULT_ARGS = [
    '--T',
    '3000',
    '--training-iters',
    '50000',
    '--loss-epoch',
    '1000',
    '--n-batch',
    '50',
    '--n-batches',
    '20',
    '--stimulus-dur-min',
    '400',
    '--stimulus-dur-max',
    '1200',
    '--delay-min',
    '800',
    '--delay-max',
    '1200',
    '--ts-delay',
    '500',
    '--min-post-ts',
    '100',
    '--min-response-dur',
    '300',
    '--sure-offer-prob',
    '0.5',
    '--internal-readout-anchor',
    'pre_go',
    '--internal-pre-go-offset-steps',
    '5',
    '--output',
    'data/timed_internal_teacher_dataset_fixed_50k.npz',
    '--summary',
    'results/reports/timed_teacher_dataset_summary.json',
]


if __name__ == "__main__":
    main(DEFAULT_ARGS + sys.argv[1:])
