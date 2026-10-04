"""Tests for the TVD error decomposition (concentration vs location)."""

import math

import pandas as pd
import pytest

from simbench_exp.decompose import decompose_error, decompose_records


def test_perfect_prediction_is_all_zero():
    d = decompose_error({"A": 0.7, "B": 0.2, "C": 0.1},
                        {"A": 0.7, "B": 0.2, "C": 0.1})
    assert d["tvd"] == pytest.approx(0.0, abs=1e-12)
    assert d["concentration_err"] == pytest.approx(0.0, abs=1e-12)
    assert d["location_err"] == pytest.approx(0.0, abs=1e-12)
    assert d["entropy_gap"] == pytest.approx(0.0, abs=1e-12)
    assert d["mode_match"] is True
    assert d["mode_mass_err"] == pytest.approx(0.0, abs=1e-12)


def test_right_shape_wrong_options_is_pure_location_error():
    # Q is a permutation of P: identical profile, mass on the wrong options.
    truth = {"A": 0.7, "B": 0.2, "C": 0.1}
    pred = {"A": 0.1, "B": 0.7, "C": 0.2}
    d = decompose_error(pred, truth)
    assert d["concentration_err"] == pytest.approx(0.0, abs=1e-12)
    assert d["location_err"] == pytest.approx(d["tvd"], abs=1e-12)
    assert d["tvd"] > 0
    assert d["mode_match"] is False          # true mode A, predicted mode B
    # same multiset of probabilities -> identical entropy
    assert d["entropy_gap"] == pytest.approx(0.0, abs=1e-12)


def test_right_options_wrong_sharpness_is_pure_concentration_error():
    # Same ranking (A>B>C) on the same options, but Q is over-sharpened.
    truth = {"A": 0.5, "B": 0.3, "C": 0.2}
    pred = {"A": 0.8, "B": 0.15, "C": 0.05}
    d = decompose_error(pred, truth)
    assert d["location_err"] == pytest.approx(0.0, abs=1e-12)
    assert d["concentration_err"] == pytest.approx(d["tvd"], abs=1e-12)
    assert d["tvd"] > 0
    assert d["mode_match"] is True
    assert d["entropy_gap"] < 0               # prediction is sharper than truth
    assert d["mode_mass_err"] > 0             # over-weights the true dominant option


def test_decomposition_identity_holds_and_is_nonnegative():
    # TVD = concentration + location, both >= 0, on assorted pairs.
    cases = [
        ({"A": 0.4, "B": 0.4, "C": 0.2}, {"A": 0.1, "B": 0.6, "C": 0.3}),
        ({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}),
        ({"A": 0.9, "B": 0.1}, {"A": 0.2, "B": 0.8}),
    ]
    for pred, truth in cases:
        d = decompose_error(pred, truth)
        assert d["concentration_err"] >= -1e-12
        assert d["location_err"] >= -1e-12
        assert d["concentration_err"] + d["location_err"] == pytest.approx(d["tvd"], abs=1e-9)
        # concentration is the minimum TVD over relabelings -> never exceeds TVD
        assert d["concentration_err"] <= d["tvd"] + 1e-12


def test_missing_and_unaligned_labels_handled():
    # pred omits C (treated as 0 mass); union of options is used.
    d = decompose_error({"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.3, "C": 0.2})
    assert d["tvd"] > 0
    assert math.isclose(d["concentration_err"] + d["location_err"], d["tvd"], abs_tol=1e-9)


def test_decompose_records_frame_has_expected_columns_and_passthrough():
    records = [
        {"pred": {"A": 0.8, "B": 0.2}, "truth": {"A": 0.5, "B": 0.5},
         "split": "grouped", "dataset": "ESS", "truth_entropy": 1.0, "score": 12.0},
        {"pred": {"A": 0.5, "B": 0.5}, "truth": {"A": 0.5, "B": 0.5},
         "split": "pop", "dataset": "ISSP", "truth_entropy": 1.0, "score": 100.0},
    ]
    df = decompose_records(records)
    assert isinstance(df, pd.DataFrame) and len(df) == 2
    for col in ("tvd", "concentration_err", "location_err", "entropy_gap",
                "mode_match", "mode_mass_err", "split", "dataset", "truth_entropy", "score"):
        assert col in df.columns
    # second row is a perfect prediction
    assert df.iloc[1]["tvd"] == pytest.approx(0.0, abs=1e-12)


def test_score_scale_attribution_with_normalizer():
    # With a dataset normalizer Z, score-loss splits the same way as TVD.
    records = [
        {"pred": {"A": 0.8, "B": 0.2}, "truth": {"A": 0.5, "B": 0.5},
         "split": "grouped", "dataset": "ESS", "truth_entropy": 1.0, "score": 0.0},
    ]
    normalizers = {("grouped", "ESS"): 0.5}  # build_normalizers keys on (split, dataset)
    df = decompose_records(records, normalizers=normalizers)
    row = df.iloc[0]
    # loss columns present and additive on the SimBench scale
    assert "score_loss" in df.columns
    assert "conc_loss" in df.columns and "loc_loss" in df.columns
    assert row["conc_loss"] + row["loc_loss"] == pytest.approx(row["score_loss"], abs=1e-9)
    assert row["score_loss"] == pytest.approx(100.0 * row["tvd"] / 0.5, abs=1e-9)
