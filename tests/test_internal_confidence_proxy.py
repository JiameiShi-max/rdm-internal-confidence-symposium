from internal_confidence_proxy import (
    compute_direction_success_proxy,
    compute_soft_sure_teacher,
    compute_soft_sure_teacher_from_margin,
)


def test_internal_margin_teacher_decreases_sure_strength_with_confidence():
    low = compute_soft_sure_teacher_from_margin(margin=0.2, sure_available=True)
    high = compute_soft_sure_teacher_from_margin(margin=6.0, sure_available=True)

    assert low["direction_success_proxy"] < high["direction_success_proxy"]
    assert low["expected_direction_value"] < high["expected_direction_value"]
    assert low["sure_strength"] > high["sure_strength"]


def test_internal_margin_teacher_matches_existing_sensory_margin_api():
    old = compute_soft_sure_teacher(sensory_margin=1.5, sure_available=True)
    new = compute_soft_sure_teacher_from_margin(margin=1.5, sure_available=True)

    comparable_keys = [
        "direction_success_proxy",
        "expected_direction_value",
        "sure_advantage",
        "raw_sure_strength",
        "sure_strength",
    ]
    for key in comparable_keys:
        assert old[key] == new[key]


def test_internal_margin_teacher_records_margin_source():
    result = compute_soft_sure_teacher_from_margin(
        margin=1.0,
        sure_available=True,
        margin_source="internal_output",
    )

    assert result["margin_source"] == "internal_output"
    assert result["teacher_margin"] == 1.0


def main():
    low = compute_soft_sure_teacher(sensory_margin=0.2, sure_available=True)
    high = compute_soft_sure_teacher(sensory_margin=6.0, sure_available=True)
    hidden = compute_soft_sure_teacher(sensory_margin=0.2, sure_available=False)

    assert low["direction_success_proxy"] < high["direction_success_proxy"]
    assert low["expected_direction_value"] < high["expected_direction_value"]
    assert low["sure_strength"] > high["sure_strength"]
    assert hidden["sure_strength"] == 0.0

    p1 = compute_direction_success_proxy(0.5)
    p2 = compute_direction_success_proxy(3.5)
    assert p1 < p2

    test_internal_margin_teacher_decreases_sure_strength_with_confidence()
    test_internal_margin_teacher_matches_existing_sensory_margin_api()
    test_internal_margin_teacher_records_margin_source()

    print("internal confidence proxy test passed")


if __name__ == "__main__":
    main()
