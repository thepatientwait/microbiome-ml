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

Writes to --out:
  candidates.tsv      every candidate, ranked within label x scheme
  best.tsv            the top --top per label x scheme
  stages.tsv          each stage's best per label x scheme: what each step added
  picks.tsv           one pick per label, and its settings per scheme
  SUMMARY.txt         all of the above, to read
  top_candidates.png  the top --top per label x scheme

Usage (any of the three; at least one):

  pixi run python scripts/tally_stages.py \
    --feature   .../ML_outputs/feature/order/ITS_marine_spatial \
    --label     .../ML_outputs/label/order/ITS_marine_spatial \
    --hypertune .../ML_outputs/hypertune/order/ITS_marine_spatial \
    --out       .../ML_outputs/tally/order/ITS_marine_spatial

Each input is a sweep's --out (its summary/ folder is read) or that
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

STAGES = ("feature", "label", "hypertune")
MODELS = {
    "RandomForestRegressor": "rf",
    "XGBRegressor": "xgb",
    "ScaledSVR": "svm",
}
# The reference palette's first three categorical slots, in its fixed
# order: feature blue, label orange, hypertune aqua.
STAGE_COLOURS = {
    "feature": "#2a78d6",
    "label": "#eb6834",
    "hypertune": "#1baf7a",
}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

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
    "sd_r2",
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


def fold_sd(raw: Any) -> Optional[float]:
    """Standard deviation of the per-fold R2 (`cross_val_scores`, a JSON list)."""
    try:
        scores = json.loads(raw) if isinstance(raw, str) and raw else []
    except ValueError:
        return None
    values = [
        v
        for v in (number(s) for s in scores)
        if v is not None and math.isfinite(v)
    ]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def read_stage(stage: str, path: Path) -> List[Dict[str, Any]]:
    """One candidate per row of the sweep's all_models.tsv."""
    folder = summary_dir(path)
    models = read_tsv(folder / "all_models.tsv")
    if not models:
        raise SystemExit(f"{stage}: no {folder / 'all_models.tsv'}")
    holdout = {
        (
            r.get("combo"),
            r.get("holdout_grouping"),
            r.get("label"),
            r.get("scheme"),
            r.get("model"),
        ): number(r.get("r2"))
        for r in read_tsv(folder / "holdout.tsv")
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
        rows.append(
            {
                "holdout_grouping": r.get("holdout_grouping") or "",
                "label": label,
                "scheme": r.get("scheme") or "",
                "stage": stage,
                "combo": r.get("combo") or "",
                "target_transform": transform,
                "model": MODELS.get(r.get("model", ""), r.get("model", "")),
                "params": params_text(r.get("best_params")),
                "pooled_r2": number(r.get("pooled_validation_r2")),
                "mean_r2": number(r.get("avg_validation_r2")),
                "sd_r2": fold_sd(r.get("cross_val_scores")),
                "holdout_r2": holdout.get(key),
                "n_label": r.get("n_label") or "",
                "feature_set": r.get("feature_set") or "",
            }
        )
    LOGGER.info("%s: %d candidate(s) from %s", stage, len(rows), folder)
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
    are listed per scheme. Only settings scored on the most schemes are
    compared, so a setting that skipped the hard scheme cannot win by it.
    """
    out = []
    for (grouping, label), block in groups(
        rows, ("holdout_grouping", "label")
    ).items():
        schemes = {r["scheme"] for r in block}
        settings = groups(
            [r for r in block if r["pooled_r2"] is not None], SETTING
        )
        if not settings:
            continue
        most = max(len({r["scheme"] for r in s}) for s in settings.values())
        scored = []
        for key, members in settings.items():
            per_scheme = {}
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
        mean, key, per_scheme = scored[0]
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
            }
        )
    return out


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
            "Each stage's own best pooled R2 (label = with a target "
            "transform, hypertune = tuned):"
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
    ranked: List[Dict[str, Any]], top: int, path: Path, dpi: int = 150
) -> Optional[Path]:
    """Top candidates per label (rows) x CV scheme (columns)."""
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
        return sorted(
            (r for r in block if not r["tied"]), key=lambda r: r["rank"]
        )[:top]

    shown = [r for block in cells.values() for r in leaders(block)]

    def spread(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Pooled R2 +- the SD of the fold R2: the score with its fold noise."""
        if row["pooled_r2"] is None or row["sd_r2"] is None:
            return None
        return row["pooled_r2"] - row["sd_r2"], row["pooled_r2"] + row["sd_r2"]

    # The axis spans the points and, as far as they stay readable, the
    # whiskers: a fold scoring -200 would flatten everything else, so a
    # whisker past one span beyond the points is cut and ends in an arrow.
    core = [
        v
        for r in shown
        for v in (r["pooled_r2"], r["holdout_r2"])
        if v is not None
    ]
    core_low, core_high = min([0.0, *core]), max([0.0, *core])
    reach = max(1.0, core_high - core_low)
    ends = [v for r in shown if spread(r) for v in spread(r)]
    low = max(min([core_low, *ends]), core_low - reach)
    high = min(max([core_high, *ends]), core_high + reach)
    pad = 0.05 * (high - low or 1.0)
    low, high = low - pad, high + pad

    fig, axes = plt.subplots(
        len(labels),
        len(schemes),
        figsize=(4.6 * len(schemes), 0.5 + 1.35 * len(labels)),
        squeeze=False,
        sharex=True,
    )
    fig.patch.set_facecolor(SURFACE)
    for i, (grouping, label) in enumerate(labels):
        for j, scheme in enumerate(schemes):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            block = leaders(cells.get((grouping, label, scheme), []))
            ys = list(range(len(block)))[::-1]
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
                if row["pooled_r2"] is not None:
                    ax.plot(
                        row["pooled_r2"],
                        y,
                        marker="o",
                        markersize=8,
                        color=colour,
                        markeredgecolor=SURFACE,
                        markeredgewidth=2,
                        zorder=4,
                    )
                if row["holdout_r2"] is not None:
                    ax.plot(
                        row["holdout_r2"],
                        y,
                        marker="D",
                        markersize=6,
                        color=INK,
                        markeredgecolor=SURFACE,
                        markeredgewidth=1.5,
                        zorder=6,
                    )
                names.append(row_name(row))
            ax.set_yticks(ys)
            ax.set_yticklabels(names, fontsize=7, color=INK_SOFT)
            ax.tick_params(axis="y", length=0)
            ax.set_xlim(low, high)
            ax.set_ylim(-0.6, top - 0.4)
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
                [], [], marker="D", linestyle="", color=INK, label="holdout R2"
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
        " (dot: pooled R2; whisker: ± 1 SD of the CV folds' R2)",
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
def tally(
    inputs: Dict[str, Path], out: Path, top: int = 3, plot: bool = True
) -> Path:
    rows: List[Dict[str, Any]] = []
    for stage in STAGES:
        if stage in inputs:
            rows += read_stage(stage, inputs[stage])
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
        ),
    )
    summary = out / "SUMMARY.txt"
    summary.write_text(
        report(ranked, stage_best, chosen, top, inputs), encoding="utf-8"
    )
    if plot:
        figure(ranked, top, out / "top_candidates.png")
    LOGGER.info("Wrote %s", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for stage in STAGES:
        parser.add_argument(
            f"--{stage}", type=Path, help=f"the {stage} sweep's --out"
        )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    inputs = {s: getattr(args, s) for s in STAGES if getattr(args, s)}
    if not inputs:
        parser.error("give at least one of --feature, --label, --hypertune")
    print(tally(inputs, args.out, args.top, not args.no_figure))


if __name__ == "__main__":
    main()
