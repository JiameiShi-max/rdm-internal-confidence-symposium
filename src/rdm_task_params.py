def task_params_from_task(task):
    """Return PsychRNN network params across Task API versions."""
    get_params = getattr(task, "get_task_params", None)
    if callable(get_params):
        return dict(get_params())

    required = ["N_batch", "N_in", "N_out", "dt", "tau", "T", "alpha", "N_steps"]
    missing = [name for name in required if not hasattr(task, name)]
    if missing:
        raise AttributeError(
            "Task object has no get_task_params() and is missing fallback attributes: "
            + ", ".join(missing)
        )
    return {name: getattr(task, name) for name in required}
