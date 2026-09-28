import numpy as np

from sure_target import RDM_SureTarget_InternalProxy_Task, set_global_seed


def test_timed_task_supports_variable_stimulus_and_explicit_sure_offer():
    set_global_seed(123)
    task = RDM_SureTarget_InternalProxy_Task(
        dt=10,
        tau=100,
        T=2000,
        N_batch=1,
        stimulus_dur_min=400,
        stimulus_dur_max=1200,
        delay_min=800,
        delay_max=1200,
        ts_delay=500,
        ts_latency_min=500,
        ts_latency_max=750,
        min_post_ts=100,
        min_response_dur=300,
        sure_offer_prob=0.5,
    )

    sure_flags = []
    stimulus_durations = []
    delay_durations = []
    ts_latencies = []
    for _ in range(400):
        x, _, _ = task.generate_trial(None)
        info = task.trial_info[-1]
        stimulus_durations.append(info["stimulus_dur"])
        delay_durations.append(info["delay_dur"])
        ts_latencies.append(info["ts_latency_from_motion_offset"])
        sure_flags.append(info["sure_available"])

        assert info["ts_onset"] == info["stimulus_end"] + int(info["ts_latency_from_motion_offset"] / task.dt)
        assert 500 <= info["ts_latency_from_motion_offset"] <= 750
        assert info["ts_latency_from_motion_offset"] % task.dt == 0
        assert info["ts_onset"] > info["stimulus_end"]
        assert info["ts_onset"] < info["delay_end"]
        assert info["delay_end"] >= info["ts_onset"] + int(100 / task.dt)
        assert info["delay_end"] <= task.N_steps - int(300 / task.dt)
        assert not info["timing_clipped"]
        if info["sure_available"]:
            assert np.max(x[: info["ts_onset"], 3]) == 0.0
            assert np.max(x[info["ts_onset"] : info["delay_end"], 3]) == 1.0
        else:
            assert np.max(x[:, 3]) == 0.0
        assert "integrated_sensory_margin" in info
        assert "mean_sensory_margin" in info
        assert info["sensory_margin"] == info["integrated_sensory_margin"]
        assert info["stimulus_duration"] == info["stimulus_dur"]
        assert info["delay_duration"] == info["delay_dur"]
        assert info["ts_onset_ms"] == info["ts_onset"] * task.dt
        assert info["go_cue_ms"] == info["delay_end"] * task.dt

    offered_rate = float(np.mean(sure_flags))
    assert 0.40 <= offered_rate <= 0.60
    assert len(set(ts_latencies)) > 1
    offered_latencies = np.asarray(ts_latencies)[np.asarray(sure_flags, dtype=bool)]
    unoffered_latencies = np.asarray(ts_latencies)[~np.asarray(sure_flags, dtype=bool)]
    assert abs(float(np.mean(offered_latencies)) - float(np.mean(unoffered_latencies))) < 30.0
    assert len(set(stimulus_durations)) > 1
    assert min(stimulus_durations) >= 400
    assert max(stimulus_durations) <= 1200
    assert min(delay_durations) >= 800
    assert max(delay_durations) <= 1200


def test_timed_task_raises_when_timing_constraints_are_impossible():
    task = RDM_SureTarget_InternalProxy_Task(
        dt=10,
        tau=100,
        T=1200,
        N_batch=1,
        stimulus_dur_min=1200,
        stimulus_dur_max=1200,
        delay_min=1200,
        delay_max=1200,
        min_response_dur=300,
        sure_offer_prob=0.5,
    )

    try:
        task.generate_trial(None)
    except ValueError as exc:
        assert "Cannot sample valid trial timing" in str(exc)
    else:
        raise AssertionError("Expected invalid timing constraints to raise ValueError")


def main():
    test_timed_task_supports_variable_stimulus_and_explicit_sure_offer()
    test_timed_task_raises_when_timing_constraints_are_impossible()
    print("stage9b timed task test passed")


if __name__ == "__main__":
    main()
