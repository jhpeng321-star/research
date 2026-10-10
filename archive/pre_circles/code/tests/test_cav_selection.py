"""Verify CAV penetration reproducibility."""

from env.vehicles import CAVSelector


def test_reproducibility():
    selector = CAVSelector(penetration=0.10, seed=42)
    first = [selector.consider(f"veh_{i}") for i in range(1000)]
    count_first = sum(first)

    selector.reset(seed=42)
    second = [selector.consider(f"veh_{i}") for i in range(1000)]
    count_second = sum(second)

    assert first == second, "CAV selection is not reproducible with the same seed"
    assert count_first == count_second
    assert abs(count_first - 100) < 30, f"unexpected penetration count {count_first}"
    print(f"test_reproducibility passed: {count_first} CAVs selected (expected ~100)")


def test_penetration_rates():
    for rate in [0.05, 0.10, 0.20]:
        selector = CAVSelector(penetration=rate, seed=42)
        selected = sum(selector.consider(f"veh_{i}") for i in range(2000))
        expected = int(rate * 2000)
        print(f"penetration {rate}: selected {selected} / 2000 (expected ~{expected})")
        assert abs(selected - expected) < 150, f"penetration {rate} deviated too much"
    print("test_penetration_rates passed")


if __name__ == "__main__":
    test_reproducibility()
    test_penetration_rates()
