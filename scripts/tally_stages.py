"""Tally the feature, label and hypertune sweeps: what to carry forward.

The three sweeps each settle one choice, on the same holdout split and the
same CV folds, so their scores compare directly:

  feature    the feature recipe: transform, prevalence, ICC, variance cut
  label      the target transform, on each label's best recipes
  hypertune  the hyperparameters, on the same recipes (plus the raw control)

Each model a sweep fitted is one candidate: stage x recipe x target
transform x model x hyperparameters, scored per label x CV scheme on pooled
validation R2 (the score the sweeps rank on). A candidate two stages both
fitted (the label sweep's `none` rows repeat the feature sweep) counts
once, at the earlier stage.

Ranking is on the validation folds only. The holdout R2 (the hypertune
sweep runs the holdout) is shown beside it and never ranked on, so it
stays a test the choice did not see.

--baseline adds the untuned baseline run (a Snakemake run's base_dir:
cv_sweep_all.tsv and holdout_sweep_summary.tsv) as a control: ranked with
the rest, kept to the labels and schemes the sweeps scored, never picked,
and set beside each pick and in every figure panel. It is a separate run,
so its rank and CV folds can differ from the sweeps'; with the same
holdout test samples, holdout R2 is the like-for-like comparison.

Writes to --out:
  candidates.tsv      every candidate, ranked within label x scheme
  best.tsv            the top --top per label x scheme
  stages.tsv          each stage's best per label x scheme: what each step added
  picks.tsv           one pick per label, and its settings per scheme
  SUMMARY.txt         all of the above, to read
  top_candidates.png  the top --top per label x scheme

Usage (any of the three sweeps, at least one; --baseline optional):

  pixi run python scripts/tally_stages.py \
    --baseline  .../ML_outputs/baseline/genus/ITS_marine_spatial \
    --feature   .../ML_outputs/feature/order/ITS_marine_spatial \
    --label     .../ML_outputs/label/order/ITS_marine_spatial \
    --hypertune .../ML_outputs/hypertune/order/ITS_marine_spatial \
    --out       .../ML_outputs/tally/order/ITS_marine_spatial

Each sweep input is a sweep's --out (its summary/ folder is read) or that
summary/ folder itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("tally-stages")

CONTROL = "baseline"
STAGES = (CONTROL, "feature", "label", "hypertune")
TUNED = STAGES[1:]
MODELS = {
    "RandomForestRegressor": "rf",
    "XGBRegressor": "xgb",
    "ScaledSVR": "svm",
}
# The reference palette's first three categorical slots, in its fixed
# order: feature blue, label orange, hypertune aqua; the control is a
# neutral grey, outside the categorical order.
STAGE_COLOURS = {
    CONTROL: "#8a8884",
    "feature": "#2a78d6",
    "label": "#eb6834",
    "hypertune": "#1baf7a",
}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
GOLD = "#e0a800"  # the holdout diamond, ink-edged on the light surface
AXIS = (-1.0, 1.0)  # the figure's R2 range; values past it are marked

COLUMNS = (
    "rank",
    "tied",
    "n_tied",
    "holdout_grouping",
    "label",
    "scheme",
    "stage",
    "combo",
    "target_transform",
    "model",
    "params",
    "pooled_r2",
    "mean_r2",
    "fold_min_r2",
    "fold_max_r2",
    "holdout_r2",
    "n_label",
    "feature_set",
)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def summary_dir(path: Path) -> Path:
    """A sweep's summary/ folder, given the sweep or the folder."""
    return path if (path / "all_models.tsv").exists() else path / "summary"


def stage_files(path: Path) -> Tuple[Path, Path]:
    """(models, holdout) tables: a sweep's summary, or a plain run's."""
    if (path / "cv_sweep_all.tsv").exists():
        return path / "cv_sweep_all.tsv", path / "holdout_sweep_summary.tsv"
    folder = summary_dir(path)
    return folder / "all_models.tsv", folder / "holdout.tsv"


def run_name(feature_set: str) -> str:
    """A plain run has no recipe id: `baseline_<rank>` from its feature set."""
    rank_name = str(feature_set or "").replace("tax_", "", 1)
    return f"{CONTROL}_{rank_name}" if rank_name else CONTROL


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def split_label(raw: str) -> Tuple[str, str]:
    """`geo_ph@yeo_johnson` -> (geo_ph, yeo_johnson); no `@` -> none."""
    label, _, transform = str(raw).partition("@")
    return label, transform or "none"


def params_text(raw: Any) -> str:
    """`best_params` JSON as `k=v, ...`; library defaults as `defaults`."""
    if raw is None or raw == "":
        return "defaults"
    try:
        params = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return str(raw)
    if not params:
        return "defaults"
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def fold_range(raw: Any) -> Tuple[Optional[float], Optional[float]]:
    """(worst, best) per-fold R2 (`cross_val_scores`, a JSON list).

    The range, not a spread statistic: with five folds one bad fold would
    stretch an SD to both sides, past scores no fold reached.
    """
    try:
        scores = json.loads(raw) if isinstance(raw, str) and raw else []
    except ValueError:
        return None, None
    values = [
        v
        for v in (number(s) for s in scores)
        if v is not None and math.isfinite(v)
    ]
    if not values:
        return None, None
    return min(values), max(values)


def read_stage(stage: str, path: Path) -> List[Dict[str, Any]]:
    """One candidate per row of the sweep's all_models.tsv (or a run's
    cv_sweep_all.tsv)."""
    models_path, holdout_path = stage_files(path)
    models = read_tsv(models_path)
    if not models:
        raise SystemExit(f"{stage}: no {models_path}")
    holdout = {
        (
            r.get("combo"),
            r.get("holdout_grouping"),
            r.get("label"),
            r.get("scheme"),
            r.get("model"),
        ): number(r.get("r2"))
        for r in read_tsv(holdout_path)
    }
    rows = []
    for r in models:
        label, transform = split_label(r["label"])
        key = (
            r.get("combo"),
            r.get("holdout_grouping"),
            r.get("label"),
            r.get("scheme"),
            r.get("model"),
        )
        fold_min, fold_max = fold_range(r.get("cross_val_scores"))
        rows.append(
            {
                "holdout_grouping": r.get("holdout_grouping") or "",
                "label": label,
                "scheme": r.get("scheme") or "",
                "stage": stage,
                "combo": r.get("combo") or run_name(r.get("feature_set", "")),
                "target_transform": transform,
                "model": MODELS.get(r.get("model", ""), r.get("model", "")),
                "params": params_text(r.get("best_params")),
                "pooled_r2": number(r.get("pooled_validation_r2")),
                "mean_r2": number(r.get("avg_validation_r2")),
                "fold_min_r2": fold_min,
                "fold_max_r2": fold_max,
                "holdout_r2": holdout.get(key),
                "n_label": r.get("n_label") or "",
                "feature_set": r.get("feature_set") or "",
            }
        )
    LOGGER.info("%s: %d candidate(s) from %s", stage, len(rows), models_path)
    return rows


def dedupe(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per fitted model: the earliest stage that fitted it."""
    kept: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    order = {stage: i for i, stage in enumerate(STAGES)}
    for row in sorted(rows, key=lambda r: order[r["stage"]]):
        key = tuple(
            str(row[k])
            for k in (
                "holdout_grouping",
                "label",
                "scheme",
                "combo",
                "target_transform",
                "model",
                "params",
                "feature_set",
            )
        )
        kept.setdefault(key, row)
    return list(kept.values())


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def _score(row: Dict[str, Any]) -> float:
    return -math.inf if row["pooled_r2"] is None else row["pooled_r2"]


def groups(
    rows: Iterable[Dict[str, Any]], keys: Sequence[str]
) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    out: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(tuple(str(row[k]) for k in keys), []).append(row)
    return out


CELL = ("holdout_grouping", "label", "scheme")


def n_filters(combo: str) -> int:
    """ICC and variance cuts in a recipe id (`clr_prev0p1_icc0p75_var1e-06`)."""
    return sum(part[:3] in ("icc", "var") for part in str(combo).split("_"))


def _simplest(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        STAGES.index(row["stage"]),
        n_filters(row["combo"]),
        row["target_transform"] != "none",
        row["params"] != "defaults",
        row["combo"],
        row["model"],
    )


def rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank within label x scheme on pooled R2 (unscored last).

    Exactly equal scores are one result: a cut that removed nothing, or
    a model predicting a constant whatever its features. They share a
    rank, the simplest is listed (`tied` False, `n_tied` the others) and
    the rest are kept in candidates.tsv with `tied` True.
    """
    ranked = []
    for _, block in groups(rows, CELL).items():
        block.sort(key=lambda r: (-round(_score(r), 9), _simplest(r)))
        rank_no, last, lead = 0, None, None
        for row in block:
            score = round(_score(row), 9)
            if (
                lead is not None
                and score == last
                and row["pooled_r2"] is not None
            ):
                lead["n_tied"] += 1
                ranked.append(
                    {**row, "rank": rank_no, "tied": True, "n_tied": 0}
                )
                continue
            rank_no += 1
            last = score
            lead = {**row, "rank": rank_no, "tied": False, "n_tied": 0}
            ranked.append(lead)
    return ranked


def best_per_stage(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The top candidate of each stage per label x scheme."""
    out = []
    for _, block in groups(rows, (*CELL, "stage")).items():
        out.append(min(block, key=lambda r: (r["rank"], r["tied"])))
    return out


SETTING = ("stage", "combo", "target_transform", "model")


def picks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One setting per label: the best mean pooled R2 over its schemes.

    A setting is stage x recipe x target transform x model, so the pick
    holds across CV schemes rather than winning the easiest one. Grid
    search tunes each scheme on its own, so a tuned setting's parameters
    are listed per scheme. The baseline control is never picked; its own
    best setting on the pick's schemes is set beside the pick.
    """
    out = []
    for (grouping, label), block in groups(
        rows, ("holdout_grouping", "label")
    ).items():
        schemes = {r["scheme"] for r in block}
        scored = scored_settings([r for r in block if r["stage"] != CONTROL])
        if not scored:
            continue
        mean, key, per_scheme = scored[0]
        control = scored_settings(
            [
                r
                for r in block
                if r["stage"] == CONTROL and r["scheme"] in per_scheme
            ]
        )
        out.append(
            {
                "holdout_grouping": grouping,
                "label": label,
                **dict(zip(SETTING, key)),
                "mean_pooled_r2": mean,
                "n_schemes": len(per_scheme),
                "of_schemes": len(schemes),
                "per_scheme": [per_scheme[s] for s in sorted(per_scheme)],
                "runner_up": (
                    None
                    if len(scored) < 2
                    else {
                        **dict(zip(SETTING, scored[1][1])),
                        "mean": scored[1][0],
                    }
                ),
                # the control's best setting on the pick's schemes
                "baseline": (
                    None
                    if not control
                    else {
                        **dict(zip(SETTING, control[0][1])),
                        "mean": control[0][0],
                        "per_scheme": control[0][2],
                    }
                ),
            }
        )
    return out


def scored_settings(
    rows: List[Dict[str, Any]],
) -> List[Tuple[float, Tuple[str, ...], Dict[str, Dict[str, Any]]]]:
    """(mean pooled R2 over schemes, setting, best row per scheme), best first.

    Only settings scored on the most schemes are compared, so a setting
    that skipped the hard scheme cannot win by it.
    """
    settings = groups([r for r in rows if r["pooled_r2"] is not None], SETTING)
    if not settings:
        return []
    most = max(len({r["scheme"] for r in s}) for s in settings.values())
    scored = []
    for key, members in settings.items():
        per_scheme: Dict[str, Dict[str, Any]] = {}
        for r in members:
            if r["scheme"] not in per_scheme or _score(r) > _score(
                per_scheme[r["scheme"]]
            ):
                per_scheme[r["scheme"]] = r
        if len(per_scheme) < most:
            continue
        mean = sum(r["pooled_r2"] for r in per_scheme.values()) / len(
            per_scheme
        )
        scored.append((mean, key, per_scheme))
    scored.sort(
        key=lambda t: (
            -round(t[0], 9),
            STAGES.index(t[1][0]),
            n_filters(t[1][1]),
            t[1][2] != "none",
            t[1],
        )
    )
    return scored


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def fmt(value: Any, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def write_tsv(
    rows: List[Dict[str, Any]], path: Path, cols: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(cols)
        for row in rows:
            writer.writerow(
                ["" if row.get(c) is None else row.get(c) for c in cols]
            )


def candidate_line(row: Dict[str, Any], mark: str = " ") -> List[str]:
    holdout = (
        ""
        if row["holdout_r2"] is None
        else f"  holdout {row['holdout_r2']:.3f}"
    )
    tied = f"  (+{row['n_tied']} tied)" if row.get("n_tied") else ""
    return [
        "{} {:>2} {:>8}  {:<9} {:<24} {:<15} {:<4}{}{}".format(
            mark,
            row["rank"],
            fmt(row["pooled_r2"]),
            row["stage"],
            row["combo"],
            row["target_transform"],
            row["model"],
            holdout,
            tied,
        ).rstrip(),
        "{}params: {}".format(" " * 15, row["params"]),
    ]


def report(
    ranked: List[Dict[str, Any]],
    stage_best: List[Dict[str, Any]],
    chosen: List[Dict[str, Any]],
    top: int,
    inputs: Dict[str, Path],
) -> str:
    lines: List[str] = []
    add = lines.append
    rule = "=" * 78
    add(rule)
    add("STAGE TALLY: " + " -> ".join(s for s in STAGES if s in inputs))
    add(rule)
    for stage in STAGES:
        if stage in inputs:
            add(f"  {stage:<10} {inputs[stage]}")
    add("")
    add("Scores are pooled validation R2 on the CV folds, ranked within each")
    add(
        "label x CV scheme. `holdout` is the final test where the sweep ran it,"
    )
    add(
        "shown, never ranked on. A model two stages both fitted counts once, at"
    )
    add("the earlier stage.")
    add("")
    add(rule)
    add("PICKS: one setting per label, the best mean pooled R2 over its CV")
    add("schemes (a setting = stage x recipe x target transform x model)")
    if CONTROL in inputs:
        add(
            "Under each: the baseline's best model on the same schemes and the"
        )
        add(
            "pick's gain over it; each scheme's baseline row sits below the pick's."
        )
    add(rule)
    for pick in chosen:
        add("")
        add(
            "label={}  grouping={}".format(
                pick["label"], pick["holdout_grouping"]
            )
        )
        add(
            "  {} | recipe {} | target {} | model {}".format(
                pick["stage"],
                pick["combo"],
                pick["target_transform"],
                pick["model"],
            )
        )
        add(
            "  mean pooled R2 {} over {} of {} scheme(s)".format(
                fmt(pick["mean_pooled_r2"]),
                pick["n_schemes"],
                pick["of_schemes"],
            )
        )
        control = pick["baseline"]
        if control:
            add(
                "  vs baseline {} | model {}: mean pooled R2 {} ({:+.3f})".format(
                    control["combo"],
                    control["model"],
                    fmt(control["mean"]),
                    pick["mean_pooled_r2"] - control["mean"],
                )
            )
        for row in pick["per_scheme"]:
            holdout = (
                ""
                if row["holdout_r2"] is None
                else f", holdout {row['holdout_r2']:.3f}"
            )
            add(
                "    {:<12} {}{}  [{}]".format(
                    row["scheme"],
                    fmt(row["pooled_r2"]),
                    holdout,
                    row["params"],
                )
            )
            base = (control or {}).get("per_scheme", {}).get(row["scheme"])
            if base:
                base_holdout = (
                    ""
                    if base["holdout_r2"] is None
                    else f", holdout {base['holdout_r2']:.3f}"
                )
                add(
                    "    {:<12} {}{}  (baseline)".format(
                        "", fmt(base["pooled_r2"]), base_holdout
                    )
                )
        runner = pick["runner_up"]
        if runner:
            add(
                "  runner-up: {} | {} | {} | {}  ({})".format(
                    runner["stage"],
                    runner["combo"],
                    runner["target_transform"],
                    runner["model"],
                    fmt(runner["mean"]),
                )
            )
    add("")

    by_cell = groups(ranked, CELL)
    stage_cells = groups(stage_best, CELL)
    for (grouping, label), cells in groups(
        ranked, ("holdout_grouping", "label")
    ).items():
        add(rule)
        add(f"label={label}  grouping={grouping}")
        add(rule)
        present = [s for s in STAGES if any(r["stage"] == s for r in cells)]
        add(
            "Each stage's own best pooled R2 (baseline = the untuned "
            "control, label = with a target transform, hypertune = tuned):"
        )
        add("  {:<12}".format("scheme") + "".join(f"{s:>11}" for s in present))
        schemes = sorted({r["scheme"] for r in cells})
        for scheme in schemes:
            best = {
                r["stage"]: r
                for r in stage_cells.get((grouping, label, scheme), [])
            }
            add(
                "  {:<12}".format(scheme)
                + "".join(
                    "{:>11}".format(
                        fmt(best[s]["pooled_r2"]) if s in best else "-"
                    )
                    for s in present
                )
            )
        add("")
        for scheme in schemes:
            add(f"  scheme={scheme}: top {top}")
            block = sorted(
                (
                    r
                    for r in by_cell[(grouping, label, scheme)]
                    if not r["tied"]
                ),
                key=lambda r: r["rank"],
            )[:top]
            for row in block:
                for text in candidate_line(
                    row, "*" if row["rank"] == 1 else " "
                ):
                    add("  " + text)
            add("")
    add(rule)
    add("Read with care:")
    add(
        "  - Validation R2 picked these, so it flatters them. The holdout R2 is"
    )
    add("    the one to report; where it disagrees, it wins.")
    add(
        "  - The label stage only tried transforms on each label's best recipes,"
    )
    add("    and the hypertune stage only tuned those recipes (plus the raw")
    add("    control): a recipe dropped early never got the later steps.")
    if CONTROL in inputs:
        add(
            "  - The baseline is a separate run: its rank (see its recipe name)"
        )
        add(
            "    and CV folds can differ from the sweeps'. With the same holdout"
        )
        add(
            "    test samples, its holdout R2 is the like-for-like comparison."
        )
    add(rule)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def row_name(row: Dict[str, Any]) -> str:
    """`recipe [@transform] model [tuned]  pooled R2` for a row's tick label."""
    parts = [row["combo"]]
    if row["target_transform"] != "none":
        parts.append("@" + row["target_transform"])
    parts.append(row["model"])
    if row["params"] != "defaults":
        parts.append("tuned")
    return "{}  {}".format(" ".join(parts), fmt(row["pooled_r2"], ".2f"))


def figure(
    ranked: List[Dict[str, Any]],
    top: int,
    path: Path,
    dpi: int = 150,
    holdout: bool = True,
) -> Optional[Path]:
    """Top candidates per label (rows) x CV scheme (columns).

    holdout=False leaves the holdout R2 diamonds out (the tables keep it).
    """
    if not holdout:
        ranked = [{**r, "holdout_r2": None} for r in ranked]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cells = groups(ranked, CELL)
    labels = list(
        dict.fromkeys((r["holdout_grouping"], r["label"]) for r in ranked)
    )
    schemes = sorted({r["scheme"] for r in ranked})
    if not labels or not schemes:
        return None

    def leaders(block: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The top tuned candidates, then the control's best below them."""
        ranked_block = sorted(
            (r for r in block if not r["tied"]), key=lambda r: r["rank"]
        )
        tuned = [r for r in ranked_block if r["stage"] != CONTROL][:top]
        control = [r for r in ranked_block if r["stage"] == CONTROL][:1]
        return tuned + control

    n_rows = top + any(r["stage"] == CONTROL for r in ranked)

    shown = [r for block in cells.values() for r in leaders(block)]

    def spread(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Worst to best fold R2: a short whisker is a stable model."""
        if row["fold_min_r2"] is None or row["fold_max_r2"] is None:
            return None
        return row["fold_min_r2"], row["fold_max_r2"]

    # A fixed -1..1 axis keeps the positive side readable. Anything below
    # -1 is drawn at the edge as an arrow and the row's lowest value is
    # written beside it; a whisker past 1 ends in an arrow.
    low, high = AXIS

    fig, axes = plt.subplots(
        len(labels),
        len(schemes),
        figsize=(4.6 * len(schemes), 0.5 + 0.45 * n_rows * len(labels)),
        squeeze=False,
        sharex=True,
    )
    fig.patch.set_facecolor(SURFACE)
    for i, (grouping, label) in enumerate(labels):
        for j, scheme in enumerate(schemes):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            block = leaders(cells.get((grouping, label, scheme), []))
            # tuned rows from the top; the control always on the bottom row
            ys = [
                0 if r["stage"] == CONTROL else n_rows - 1 - k
                for k, r in enumerate(block)
            ]
            if any(r["stage"] == CONTROL for r in block):
                ax.axhline(0.5, color=GRID, linewidth=0.8, zorder=0)
            names = []
            for y, row in zip(ys, block):
                colour = STAGE_COLOURS[row["stage"]]
                whisker = spread(row)
                if whisker is not None:
                    lo, hi = whisker
                    ax.hlines(
                        y,
                        max(lo, low),
                        min(hi, high),
                        color=colour,
                        linewidth=2,
                        zorder=2,
                    )
                    for end, edge, mark in ((lo, low, "<"), (hi, high, ">")):
                        if (end < edge) if mark == "<" else (end > edge):
                            ax.plot(
                                edge,
                                y,
                                marker=mark,
                                markersize=6,
                                color=colour,
                                zorder=2,
                            )
                pooled, held = row["pooled_r2"], row["holdout_r2"]
                if pooled is not None:
                    inside = pooled >= low
                    ax.plot(
                        max(pooled, low),
                        y,
                        marker="o" if inside else "<",
                        markersize=8 if inside else 9,
                        color=colour,
                        markeredgecolor=SURFACE,
                        markeredgewidth=2 if inside else 1,
                        zorder=4,
                    )
                if held is not None:
                    ax.plot(
                        max(held, low),
                        y,
                        marker="D",
                        markersize=6,
                        color=GOLD,
                        markeredgecolor=INK,
                        markeredgewidth=0.8,
                        zorder=6,
                    )
                values = [
                    v
                    for v in (pooled, held, whisker[0] if whisker else None)
                    if v is not None
                ]
                if values and min(values) < low:
                    ax.annotate(
                        f"{min(values):.2f}",
                        (low, y),
                        xytext=(3, 4),
                        textcoords="offset points",
                        fontsize=6.5,
                        color=INK_SOFT,
                        ha="left",
                        va="bottom",
                        zorder=7,
                    )
                names.append(row_name(row))
            ax.set_yticks(ys)
            ax.set_yticklabels(names, fontsize=7, color=INK_SOFT)
            ax.tick_params(axis="y", length=0)
            ax.set_xlim(low - 0.08, high + 0.05)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_ylim(-0.6, n_rows - 0.4)
            ax.axvline(0, color=INK_SOFT, linewidth=0.8, zorder=1)
            ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(axis="x", colors=INK_SOFT, labelsize=7)
            if i == 0:
                ax.set_title(f"scheme: {scheme}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, color=INK)
            if i == len(labels) - 1:
                ax.set_xlabel(
                    "pooled validation R2", fontsize=8, color=INK_SOFT
                )
    present = [s for s in STAGES if any(r["stage"] == s for r in shown)]
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=7,
            linewidth=2,
            color=STAGE_COLOURS[s],
            label=s,
        )
        for s in present
    ]
    if any(r["holdout_r2"] is not None for r in shown):
        handles.append(
            Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                color=GOLD,
                markeredgecolor=INK,
                markeredgewidth=0.8,
                label="holdout R2",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        f"Top {top} per label x CV scheme, ranked on pooled validation R2"
        " (dot: pooled R2; whisker: worst to best CV fold R2)"
        + (", baseline's best on the bottom row" if n_rows > top else "")
        + "\nAxis cut at -1: an arrow marks a value below it, and the row's"
        " lowest value is written beside it",
        fontsize=10,
        color=INK,
        y=1.0 + 0.25 / len(labels),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CONTROL_COLS = (
    "baseline_combo",
    "baseline_model",
    "baseline_mean_pooled_r2",
    "baseline_pooled_r2",
    "baseline_holdout_r2",
)


def _control_cols(pick: Dict[str, Any], scheme: str) -> Dict[str, Any]:
    """picks.tsv's baseline columns for one of the pick's schemes."""
    control = pick.get("baseline")
    if not control:
        return {}
    row = control["per_scheme"].get(scheme) or {}
    return {
        "baseline_combo": control["combo"],
        "baseline_model": control["model"],
        "baseline_mean_pooled_r2": control["mean"],
        "baseline_pooled_r2": row.get("pooled_r2"),
        "baseline_holdout_r2": row.get("holdout_r2"),
    }


def tally(
    inputs: Dict[str, Path],
    out: Path,
    top: int = 3,
    plot: bool = True,
    holdout: bool = True,
) -> Path:
    rows: List[Dict[str, Any]] = []
    for stage in STAGES:
        if stage in inputs:
            rows += read_stage(stage, inputs[stage])
    # The control only where the sweeps scored: a baseline run covers every
    # label and CV scheme, the sweeps a few.
    swept = {tuple(r[k] for k in CELL) for r in rows if r["stage"] != CONTROL}
    rows = [
        r
        for r in rows
        if r["stage"] != CONTROL or tuple(r[k] for k in CELL) in swept
    ]
    ranked = rank(dedupe(rows))
    stage_best = best_per_stage(ranked)
    chosen = picks(ranked)

    out.mkdir(parents=True, exist_ok=True)

    def order(r: Dict[str, Any]) -> Tuple[Any, ...]:
        return (r["holdout_grouping"], r["label"], r["scheme"], r["rank"])

    write_tsv(sorted(ranked, key=order), out / "candidates.tsv", COLUMNS)
    write_tsv(
        sorted(
            [r for r in ranked if r["rank"] <= top and not r["tied"]],
            key=order,
        ),
        out / "best.tsv",
        COLUMNS,
    )
    write_tsv(
        sorted(
            stage_best, key=lambda r: (*order(r)[:3], STAGES.index(r["stage"]))
        ),
        out / "stages.tsv",
        COLUMNS,
    )
    pick_rows = [
        {
            **{k: p[k] for k in ("holdout_grouping", "label", *SETTING)},
            "mean_pooled_r2": p["mean_pooled_r2"],
            "scheme": row["scheme"],
            "pooled_r2": row["pooled_r2"],
            "holdout_r2": row["holdout_r2"],
            "params": row["params"],
            "feature_set": row["feature_set"],
            **_control_cols(p, row["scheme"]),
        }
        for p in chosen
        for row in p["per_scheme"]
    ]
    write_tsv(
        pick_rows,
        out / "picks.tsv",
        (
            "holdout_grouping",
            "label",
            *SETTING,
            "mean_pooled_r2",
            "scheme",
            "pooled_r2",
            "holdout_r2",
            "params",
            "feature_set",
            *CONTROL_COLS,
        ),
    )
    summary = out / "SUMMARY.txt"
    summary.write_text(
        report(ranked, stage_best, chosen, top, inputs), encoding="utf-8"
    )
    if plot:
        figure(ranked, top, out / "top_candidates.png", holdout=holdout)
    LOGGER.info("Wrote %s", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for stage in STAGES:
        parser.add_argument(
            f"--{stage}",
            type=Path,
            help=(
                "the baseline run's base_dir, as a control"
                if stage == CONTROL
                else f"the {stage} sweep's --out"
            ),
        )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument(
        "--no-holdout",
        action="store_true",
        help="leave the holdout R2 out of the figure (the tables keep it)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    inputs = {s: getattr(args, s) for s in STAGES if getattr(args, s)}
    if not any(s in inputs for s in TUNED):
        parser.error("give at least one of --feature, --label, --hypertune")
    print(
        tally(
            inputs,
            args.out,
            args.top,
            not args.no_figure,
            holdout=not args.no_holdout,
        )
    )


if __name__ == "__main__":
    main()
