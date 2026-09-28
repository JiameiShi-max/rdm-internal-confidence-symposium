from model_freeze_report import build_report


def test_report_stays_not_ready_when_empirical_artifacts_are_missing():
    report, status = build_report(
        branch="feature/model-freeze-validation",
        baseline_commit="abc123",
        calibration={"empirical_teacher_accuracy_available": False},
    )

    assert status["ready"] is False
    assert status["recommendation"] == "NOT READY FOR POPULATION ANALYSIS"
    assert "abc123" in report
    assert "proxy-on-proxy" in report
