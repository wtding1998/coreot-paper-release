from coreot.results.pbmc_s7_calibration import Q_VALUES


def test_calibration_quantiles_match_adr_0043() -> None:
    assert Q_VALUES == (
        0.85,
        0.86,
        0.87,
        0.88,
        0.89,
        0.90,
        0.91,
        0.92,
        0.93,
        0.94,
        0.95,
        0.96,
        0.97,
        0.975,
        0.98,
        0.99,
        0.995,
    )
