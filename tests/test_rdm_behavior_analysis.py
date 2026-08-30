import os
import tempfile

from rdm_behavior_analysis import (
    build_trial_eval_records,
    default_behavior_paths,
    plot_behavior_summary,
    summarize_behavior,
    write_behavior_summary_json,
    write_trial_records_csv,
)


def make_records():
    trial_info = [
        {"coh": 0.0, "dir_choice": 0, "sure_available": True},
        {"coh": 0.0, "dir_choice": 1, "sure_available": True},
        {"coh": 0.0, "dir_choice": 0, "sure_available": False},
        {"coh": 0.5, "dir_choice": 1, "sure_available": True},
        {"coh": 0.5, "dir_choice": 1, "sure_available": False},
    ]
    choices = [3, 1, 1, 2, 1]
    return build_trial_eval_records(trial_info, choices)


def test_behavior_summary_defines_psure_no_sure_and_waived_sure_accuracy():
    summary = summarize_behavior(make_records())

    assert summary["n_trials"] == 5
    assert summary["n_sure_offered"] == 3
    assert abs(summary["overall_p_sure"] - (1.0 / 3.0)) < 1e-12
    assert abs(summary["no_sure_accuracy"] - 0.5) < 1e-12
    assert abs(summary["waived_sure_accuracy"] - 0.5) < 1e-12
    assert summary["n_waived_sure"] == 2
    low = summary["coherence"][0]
    high = summary["coherence"][1]
    assert low["coh"] == 0.0
    assert low["p_sure"] == 0.5
    assert high["coh"] == 0.5
    assert high["p_sure"] == 0.0


def test_behavior_writers_create_csv_json_and_plot():
    records = make_records()
    summary = summarize_behavior(records)
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "trials.csv")
        json_path = os.path.join(tmp_dir, "behavior.json")
        fig_path = os.path.join(tmp_dir, "behavior.png")

        write_trial_records_csv(csv_path, records)
        write_behavior_summary_json(json_path, summary)
        plot_behavior_summary(fig_path, summary)

        assert os.path.getsize(csv_path) > 0
        assert os.path.getsize(json_path) > 0
        assert os.path.getsize(fig_path) > 0


def test_default_behavior_paths_derive_from_summary_path():
    paths = default_behavior_paths("results/model_summary.json")

    assert paths["trial_csv"] == "results/model_summary_trials.csv"
    assert paths["behavior_summary"] == "results/model_summary_behavior.json"
    assert paths["behavior_fig"] == "results/model_summary_behavior.png"


def main():
    test_behavior_summary_defines_psure_no_sure_and_waived_sure_accuracy()
    test_behavior_writers_create_csv_json_and_plot()
    test_default_behavior_paths_derive_from_summary_path()
    print("rdm behavior analysis test passed")


if __name__ == "__main__":
    main()
