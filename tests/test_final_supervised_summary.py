import inspect
from pathlib import Path

import pytest

from final_supervised_summary import assemble_key_results, load_sources, stage_audit_rows


ROOT = Path(__file__).resolve().parents[1]


def test_key_results_are_copied_from_frozen_stage_summaries():
    rows = assemble_key_results(load_sources(ROOT / "results"))
    assert [row["seed"] for row in rows] == [7, 8, 9]
    seed7 = rows[0]
    assert seed7["test_no_sure_direction_accuracy"] == pytest.approx(0.88)
    assert seed7["stage1_choice_cv_accuracy"] == pytest.approx(0.9954999438895747)
    assert seed7["stage2_confidence_operational_onset_ms"] == 140
    assert seed7["stage3_primary_beta"] == pytest.approx(-0.5812635276006586)


def test_stage_audit_has_all_four_completed_stages_and_limitations():
    rows = stage_audit_rows()
    assert len(rows) == 4
    assert all(row["main_limitation"] for row in rows)
    assert "not uniform" in rows[-1]["main_limitation"]


def test_synthesis_source_has_no_scientific_fitting_or_replay_calls():
    import final_supervised_summary as module

    source = inspect.getsource(module)
    assert "BasicSimulator(" not in source
    assert "Logistic" + "Regression(" not in source
    assert "Ri" + "dge(" not in source
    assert "PCA(" not in source
    assert "model" + ".train(" not in source
