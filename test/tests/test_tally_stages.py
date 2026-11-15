"""scripts/tally_stages.py: feature -> label -> hypertune, tallied."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import tally_stages as ts  # noqa: E402

MODEL_COLS = (
    "combo",
    "holdout_grouping",
    "feature_set",
    "label",
    "scheme",
    "model",
    "avg_validation_r2",
    "pooled_validation_r2",
    "n_label",
    "best_params",
)


def _row(combo, label, scheme, model, r2, params=None):
    return {
        "combo": combo,
        "holdout_grouping": "bioproject",
        "feature_set": "tax_order_bio",
        "label": label,
        "scheme": scheme,
        "model": model,
        "avg_validation_r2": r2,
        "pooled_validation_r2": r2,
        "n_label": 100,
        "best_params": json.dumps(params or {}),
    }


def _sweep(root: Path, rows, holdout=None) -> Path:
    summary = root / "summary"
    summary.mkdir(parents=True)
    with (summary / "all_models.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, MODEL_COLS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    if holdout:
        cols = ("combo", "holdout_grouping", "label", "scheme", "model", "r2")
        with (summary / "holdout.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, cols, delimiter="\t")
            writer.writeheader()
            writer.writerows(holdout)
    return root


@pytest.fixture
def stages(tmp_path):
    rf, svm = "RandomForestRegressor", "ScaledSVR"
    feature = _sweep(
        tmp_path / "feature",
        [
            _row("clr_prev0p1", "ph", "month", rf, 0.40),
            _row("clr_prev0p1", "ph", "bioproject", rf, 0.20),
            # an ICC cut that removed nothing: the same score
            _row("clr_prev0p1_icc0p75", "ph", "month", rf, 0.40),
            _row("clr_prev0p1_icc0p75", "ph", "bioproject", rf, 0.20),
            _row("relabund_prev0p1", "ph", "month", svm, 0.30),
            _row("relabund_prev0p1", "ph", "bioproject", svm, 0.25),
        ],
    )
    label = _sweep(
        tmp_path / "label",
        [
            # `none` repeats the feature sweep: counted there
            _row("clr_prev0p1", "ph", "month", rf, 0.40),
            _row("clr_prev0p1", "ph@yeo_johnson", "month", rf, 0.42),
            _row("clr_prev0p1", "ph@yeo_johnson", "bioproject", rf, 0.15),
        ],
    )
    tuned = {"max_features": 0.33, "min_samples_leaf": 2}
    hypertune = _sweep(
        tmp_path / "hypertune",
        [
            _row("clr_prev0p1", "ph", "month", rf, 0.45, tuned),
            _row("clr_prev0p1", "ph", "bioproject", rf, 0.24, tuned),
        ],
        holdout=[
            {
                "combo": "clr_prev0p1",
                "holdout_grouping": "bioproject",
                "label": "ph",
                "scheme": "month",
                "model": rf,
                "r2": 0.31,
            },
        ],
    )
    return {"feature": feature, "label": label, "hypertune": hypertune}


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_the_tally_ranks_dedupes_and_picks(tmp_path, stages):
    out = tmp_path / "tally"
    ts.tally(stages, out, top=3, plot=False)

    candidates = _read(out / "candidates.tsv")
    # 6 + 3 + 2 rows, less the label sweep's repeated `none` row
    assert len(candidates) == 10
    month = [r for r in candidates if r["scheme"] == "month"]
    leaders = [r for r in month if r["tied"] == "False"]
    assert [(r["stage"], r["target_transform"]) for r in leaders] == [
        ("hypertune", "none"),
        ("label", "yeo_johnson"),
        ("feature", "none"),
        ("feature", "none"),
    ]
    assert leaders[0]["params"] == "max_features=0.33, min_samples_leaf=2"
    assert leaders[0]["holdout_r2"] == "0.31"
    # the tie: listed once, as the recipe without the cut
    third = leaders[2]
    assert third["combo"] == "clr_prev0p1" and third["n_tied"] == "1"

    best = _read(out / "best.tsv")
    assert len([r for r in best if r["scheme"] == "month"]) == 3

    # mean over both schemes: tuned 0.345 beats the defaults (0.30) and the
    # transform (0.285), though the transform beat defaults on month alone
    picks = _read(out / "picks.tsv")
    assert {(r["stage"], r["combo"], r["model"]) for r in picks} == {
        ("hypertune", "clr_prev0p1", "rf")
    }
    assert {r["scheme"] for r in picks} == {"month", "bioproject"}

    stage_rows = _read(out / "stages.tsv")
    month_stages = {
        r["stage"]: r["pooled_r2"]
        for r in stage_rows
        if r["scheme"] == "month"
    }
    assert month_stages == {
        "feature": "0.4",
        "label": "0.42",
        "hypertune": "0.45",
    }

    text = (out / "SUMMARY.txt").read_text()
    assert "PICKS" in text and "(+1 tied)" in text and "holdout 0.310" in text


def test_one_stage_alone_is_enough(tmp_path, stages):
    out = tmp_path / "tally"
    ts.tally({"feature": stages["feature"]}, out, plot=False)
    assert (out / "SUMMARY.txt").exists()


def test_the_figure_is_drawn(tmp_path, stages):
    pytest.importorskip("matplotlib")
    out = tmp_path / "tally"
    ts.tally(stages, out, plot=True)
    assert (out / "top_candidates.png").stat().st_size > 0


def test_labels_and_params_parse():
    assert ts.split_label("geo_ph@yeo_johnson") == ("geo_ph", "yeo_johnson")
    assert ts.split_label("geo_ph") == ("geo_ph", "none")
    assert ts.params_text("{}") == "defaults"
    assert ts.params_text('{"b": 1, "a": 0.5}') == "a=0.5, b=1"
    assert ts.n_filters("clr_prev0p1_icc0p75_var1e-06") == 2
    assert ts.fold_sd("[0.1, 0.3, null, NaN]") == pytest.approx(
        0.1414, abs=1e-4
    )
    assert ts.fold_sd("[0.2]") is None and ts.fold_sd("") is None
