from rdm_task_params import task_params_from_task


class MinimalTaskWithoutGetTaskParams:
    def __init__(self):
        self.N_in = 4
        self.N_out = 4
        self.N_batch = 50
        self.dt = 10
        self.tau = 100
        self.T = 3000
        self.alpha = 0.1
        self.N_steps = 300


class TaskWithGetTaskParams:
    def get_task_params(self):
        return {"N_in": 2, "N_out": 3, "custom": "kept"}


def test_task_params_from_task_falls_back_when_psychrnn_method_is_missing():
    params = task_params_from_task(MinimalTaskWithoutGetTaskParams())

    assert params["N_in"] == 4
    assert params["N_out"] == 4
    assert params["N_batch"] == 50
    assert params["dt"] == 10
    assert params["tau"] == 100
    assert params["T"] == 3000
    assert params["alpha"] == 0.1
    assert params["N_steps"] == 300


def test_task_params_from_task_prefers_native_psychrnn_method():
    params = task_params_from_task(TaskWithGetTaskParams())

    assert params == {"N_in": 2, "N_out": 3, "custom": "kept"}


def main():
    test_task_params_from_task_falls_back_when_psychrnn_method_is_missing()
    test_task_params_from_task_prefers_native_psychrnn_method()
    print("rdm task params test passed")


if __name__ == "__main__":
    main()
