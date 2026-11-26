"""Diagnostic figures for regression models and their validation.

* :func:`plot_prediction_density` - actual vs predicted as a hexbin density with
  the 1:1 line, the regression line and the fit statistics in the title
* :func:`plot_two_panels` - two such panels side by side (cross-validation
  next to a held-out set, or a group's CV next to its holdout)
* :func:`plot_predictions_by_group` - the density plus a stacked histogram of
  the errors coloured by a category (e.g. the held-out group of LOGO)
* :func:`plot_logi_groups` - one two-panel figure per group (LOGI)
* :func:`plot_pca_overview` - PCA of the modelled features coloured by group
  factors and by the label, train versus projected test
* :func:`plot_top_importances` - bar chart of the largest importances

All functions take plain arrays / frames and explicit options (no globals).
``style`` is any matplotlib style (name, file or list) applied only while the
figure is drawn. With ``file_name`` the figure is saved and closed; without it
the open ``Figure`` is returned.

The look follows the workflow's figures; the numbers come from
:func:`~microbiome_ml.train.metrics.regression_metrics`.
"""

import contextlib
import logging
from collections import Counter
from pathlib import Path
from textwrap import wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

from microbiome_ml.train.metrics import regression_metrics

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
Style = Union[None, str, Path, Sequence[Union[str, Path]]]
_BASELINE = "#7272a1"


def _plt() -> Any:
    import matplotlib.pyplot as plt

    return plt


@contextlib.contextmanager
def _style(style: Style) -> Iterator[None]:
    plt = _plt()
    if style is None:
        yield
        return
    names = [style] if isinstance(style, (str, Path)) else list(style)
    with plt.style.context([str(s) for s in names]):
        yield


def _finish(fig: Any, file_name: Optional[PathLike], dpi: int) -> Any:
    if file_name is None:
        return fig
    plt = _plt()
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(file_name), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return None


def prediction_stats(
    predictions: np.ndarray, values: np.ndarray
) -> Tuple[float, float, float, float]:
    """``(mse, r2, mae, pcc)`` of predictions against the actual values."""
    m = regression_metrics(values, predictions)
    # only r2 and pcc can be undefined for non-empty input
    assert m["mse"] is not None and m["mae"] is not None
    return (
        float(m["mse"]),
        float(m["r2"]) if m["r2"] is not None else float("nan"),
        float(m["mae"]),
        float(m["pcc"]) if m["pcc"] is not None else float("nan"),
    )


def count_formatter(x: float, pos: int) -> str:
    """Tick label: ``1.2k`` from a thousand up."""
    if x >= 1000:
        return f"{float(x * 1e-3):.1f}k"
    return str(int(x))


def remove_outliers_zscore(
    data: np.ndarray,
    colours: Optional[pl.Series],
    threshold: float,
) -> Tuple[np.ndarray, Optional[List[Any]]]:
    """Drop points more than ``threshold`` SD from the mean (for histograms).

    Returns the kept data and the kept entries of ``colours`` (or ``None``).
    """
    from scipy.stats import zscore

    z = zscore(data)
    kept = data[(z < threshold) & (z > -threshold)]
    if colours is None:
        return kept, None
    return kept, [c for c, zi in zip(colours.to_list(), z) if -threshold < zi < threshold]


def set_axis_limits(
    ax: Any, predictions: np.ndarray, values: np.ndarray, units: str
) -> Tuple[float, float, float]:
    """Square axes with 10 % padding; labels with units. Returns
    ``(lower, upper, buffer)``."""
    everything = np.concatenate((predictions, values))
    buffer = np.ptp(everything) * 0.1
    lower, upper = min(everything) - buffer, max(everything) + buffer
    ax.set_xlim((lower, upper))
    ax.set_ylim((lower, upper))
    ax.set_xlabel(f"Actual Values ({units})")
    ax.set_ylabel(f"Predicted Values ({units})")
    return lower, upper, buffer


def hexbin_panel(
    fig: Any,
    ax: Any,
    values: np.ndarray,
    predictions: np.ndarray,
    cmap: str,
    cax: Any,
    lower: float,
    upper: float,
    orientation: Optional[str] = None,
) -> Tuple[float, float]:
    """Hexbin density of actual vs predicted, the 1:1 line, the regression
    line and a colour bar in ``cax``. Returns the regression ``(slope,
    intercept)``."""
    import cmasher as cmr
    import matplotlib.ticker as ticker

    sub = cmr.get_sub_cmap(cmap, 0.2, 0.8)
    hb = ax.hexbin(
        values,
        predictions,
        cmap=sub,
        bins="log",
        zorder=3,
        gridsize=50,
        mincnt=1,
        linewidths=0.2,
        extent=(lower, upper, lower, upper),
    )
    ax.axline((0, 0), slope=1, color=_BASELINE, linestyle="dashed")
    if orientation is None:
        orientation = "horizontal" if len(fig.axes) == 4 else "vertical"
    fig.colorbar(
        hb, cax=cax, label="Count (log_10)", drawedges=False, orientation=orientation
    )
    cax.grid()
    m, b = np.polyfit(values, predictions, 1)
    ax.plot(
        values,
        m * np.array(values, dtype=np.float32) + b,
        color="red",
        alpha=0.75,
        zorder=2,
        linewidth=4,
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    return float(m), float(b)


def _diagnostics_text(
    pcc: float,
    r2: float,
    m: float,
    b: float,
    mse: float,
    mae: float,
    n: int,
    two_lines: bool,
) -> str:
    sep = "\n" if two_lines else " | "
    return (
        f"R2: {r2:.3f} | PCC: {pcc:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}"
        f"{sep}m: {m:.3f} | b: {b:.3f} | n: {n}"
    )


def _set_title(
    fig: Any,
    title: str,
    pcc: float,
    r2: float,
    m: float,
    b: float,
    mse: float,
    mae: float,
    n: int,
    two_lines: bool,
    y_title: float,
    y_stats: float,
) -> None:
    fig.text(0.5, y_title, title, ha="center", va="center", fontsize="large", weight="bold")
    fig.text(
        0.5,
        y_stats,
        _diagnostics_text(pcc, r2, m, b, mse, mae, n, two_lines),
        ha="center",
        va="center",
        fontsize="medium",
        weight="normal",
    )


def plot_prediction_density(
    predictions: np.ndarray,
    values: np.ndarray,
    title: str = "Actual vs. Predicted Values",
    units: str = "",
    density_cmap: str = "inferno_r",
    file_name: Optional[PathLike] = None,
    style: Style = None,
    dpi: int = 500,
) -> Any:
    """One hexbin panel of actual vs predicted values."""
    import matplotlib.gridspec as gridspec

    plt = _plt()
    predictions, values = np.asarray(predictions), np.asarray(values)
    with _style(style):
        mse, r2, mae, pcc = prediction_stats(predictions, values)
        fig = plt.figure(figsize=(7, 5), constrained_layout=True)
        axs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5, 0.25])
        ax, cax = plt.subplot(axs[0]), plt.subplot(axs[1])
        lower, upper, _ = set_axis_limits(ax, predictions, values, units)
        m, b = hexbin_panel(fig, ax, values, predictions, density_cmap, cax, lower, upper)
        cax.set_ylabel("Count (log_10)", rotation=270, labelpad=15)
        _set_title(fig, title, pcc, r2, m, b, mse, mae, len(values), True, 1.14, 1.05)
        return _finish(fig, file_name, dpi)


def plot_two_panels(
    left: Tuple[np.ndarray, np.ndarray],
    right: Tuple[np.ndarray, np.ndarray],
    title: str,
    units: str,
    density_cmap: str,
    left_label: str = "CV",
    right_label: str = "Validation",
    file_name: Optional[PathLike] = None,
    style: Style = None,
    dpi: int = 500,
) -> Any:
    """Two hexbin panels, each given as ``(predictions, values)``."""
    import matplotlib.gridspec as gridspec

    plt = _plt()
    with _style(style):
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        axs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[5, 0.25, 5, 0.25])
        panels = (
            (left_label, left, axs[0], axs[1]),
            (right_label, right, axs[2], axs[3]),
        )
        for label, (predictions, values), plot_slot, cbar_slot in panels:
            predictions, values = np.asarray(predictions), np.asarray(values)
            ax, cax = plt.subplot(plot_slot), plt.subplot(cbar_slot)
            lower, upper, _ = set_axis_limits(ax, predictions, values, units)
            m, b = hexbin_panel(
                fig, ax, values, predictions, density_cmap, cax, lower, upper,
                orientation="vertical",
            )
            mse, r2, mae, pcc = prediction_stats(predictions, values)
            ax.set_title(
                f"{label}\n" + _diagnostics_text(pcc, r2, m, b, mse, mae, len(values), True),
                ha="center", va="center", fontsize="medium", weight="normal",
            )
        fig.suptitle(title, fontsize="large", weight="bold")
        return _finish(fig, file_name, dpi)


def _group_palette(c: pl.Series, c_list: Any, cmap: str) -> Tuple[Any, Any, bool]:
    """Colours for the error histogram: ``(palette, hue, show_legend)``."""
    import cmasher as cmr

    plt = _plt()
    sub = cmr.get_sub_cmap(cmap, 0.2, 0.8)
    if c is None:
        return None, None, False
    if c.dtype == pl.Utf8:
        if len(set(c_list)) > 10:
            sub = plt.get_cmap("tab20")
        if len(set(c_list)) > 20:
            from pandas import Categorical

            most_common = [item for item, _ in Counter(c_list).most_common(20)] + ["Other"]
            c_list = [
                item if item in most_common and item != "NA" else "Other"
                for item in c_list
            ]
            colors: Any = {
                label: sub(idx)
                for idx, label in zip(np.linspace(0, 1, len(most_common)), most_common)
            }
            colors["Other"] = (0.6, 0.6, 0.6, 1.0)
            c_list = Categorical(c_list, categories=most_common, ordered=True)
        else:
            unique = sorted(set(c_list))
            colors = [sub(i) for i in np.linspace(0, 1, len(unique))]
        return colors, c_list, True
    return sub, c_list, False


def _error_histogram(
    fig: Any,
    hax: Any,
    predictions: np.ndarray,
    values: np.ndarray,
    units: str,
    bins: Any,
    trim: Any,
    cax: Any,
    c: Any,
    cmap: str,
    cbar_label: Any,
) -> Any:
    import matplotlib.ticker as ticker
    import seaborn as sns

    plt = _plt()
    diff = values - predictions
    c_list: Any = c.to_list() if c is not None else None
    if trim:
        diff, c_list = remove_outliers_zscore(diff, c, trim)
    palette, hue, legend = _group_palette(c, c_list, cmap)
    sns.histplot(
        x=diff, bins=bins, hue=hue, palette=palette, multiple="stack",
        zorder=2, ec=None, alpha=1, legend=legend, ax=hax,
    )
    hax.axvline(x=0, color=_BASELINE, linestyle="dashed", zorder=1)
    hax.set_xlabel(f"Error [True - Pred] ({units})")
    hax.set_ylabel("")
    hax.yaxis.tick_right()
    hax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    hax.yaxis.set_major_formatter(ticker.FuncFormatter(count_formatter))
    if c is None:
        cax.axis("off")
    elif legend:
        handles = hax.get_legend().legend_handles
        labels = ["\n".join(wrap(t.get_text(), 20)) for t in hax.legend_.texts]
        hax.legend(handles[:10], labels[:10])
        sns.move_legend(hax, "upper left", bbox_to_anchor=(1.15, 1), title=None, frameon=False)
        cax.axis("off")
    else:
        norm = plt.Normalize(min(c_list), max(c_list))
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(mappable, cax=cax, label=cbar_label, drawedges=False, orientation="horizontal")
        cax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
        cax.grid()


def plot_predictions_by_group(
    predictions: np.ndarray,
    values: np.ndarray,
    groups: Optional[pl.Series] = None,
    title: str = "Actual vs. Predicted Values",
    units: str = "",
    hist_cmap: str = "viridis",
    density_cmap: str = "inferno_r",
    cbar_label: Optional[str] = None,
    hist_trim: int = 4,
    hist_bins: int = 20,
    file_name: Optional[PathLike] = None,
    style: Style = None,
    dpi: int = 500,
) -> Any:
    """Hexbin density plus a stacked histogram of the errors coloured by
    ``groups`` (a text series: one colour per category, legend of the first
    ten; a numeric series: a colour bar). ``groups=None`` (or a single value)
    gives the histogram without colours."""
    import matplotlib.gridspec as gridspec

    plt = _plt()
    predictions, values = np.asarray(predictions), np.asarray(values)
    c = groups
    if c is None:
        figsize = (10, 5.25)
    elif c.unique().len() == 1:
        figsize, c = (10, 5.25), None
    elif c.dtype == pl.Utf8:
        figsize = (12, 6.25)
        c = c.fill_null("NA").str.replace("N/A", "NA")
    else:
        figsize, c = (10, 5.25), c.fill_null(0)
    with _style(style):
        mse, r2, mae, pcc = prediction_stats(predictions, values)
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        axs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[5, 0.25])
        ax, hax, cax1, cax2 = (plt.subplot(axs[i]) for i in range(4))
        lower, upper, _ = set_axis_limits(ax, predictions, values, units)
        m, b = hexbin_panel(fig, ax, values, predictions, density_cmap, cax1, lower, upper)
        _error_histogram(fig, hax, predictions, values, units, hist_bins, hist_trim, cax2, c, hist_cmap, cbar_label)
        _set_title(fig, title, pcc, r2, m, b, mse, mae, len(values), False, 1.1, 1.04)
        return _finish(fig, file_name, dpi)


def plot_logi_groups(
    cv_predictions: pl.DataFrame,
    holdout_predictions: pl.DataFrame,
    out_dir: PathLike,
    title: str,
    units: str,
    density_cmap: str = "inferno_r",
    style: Style = None,
    dpi: int = 500,
    skip_groups: Sequence[Any] = (None, "N/A", "NA"),
) -> List[Path]:
    """One CV-vs-holdout figure per group.

    ``cv_predictions`` has ``group, real, pred``; ``holdout_predictions`` also
    ``split_method``. ``title`` may contain ``{group}``. Groups without both
    CV and holdout predictions are skipped. Returns the files written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv = cv_predictions.drop_nulls(subset=["real", "pred"])
    ho = holdout_predictions.drop_nulls(subset=["real", "pred"])
    groups = set(cv["group"].to_list()) | set(ho["group"].to_list())
    written: List[Path] = []
    for group in sorted(groups, key=str):
        if group in skip_groups:
            continue
        cv_g, ho_g = cv.filter(pl.col("group") == group), ho.filter(pl.col("group") == group)
        if cv_g.is_empty() or ho_g.is_empty():
            logger.warning("Skipping %s: missing CV or holdout predictions", group)
            continue
        label = "; ".join(str(m) for m in ho_g.get_column("split_method").unique().to_list())
        path = out / f"{str(group).replace(' ', '_')}.png"
        plot_two_panels(
            (cv_g["pred"].cast(pl.Float64).to_numpy(), cv_g["real"].cast(pl.Float64).to_numpy()),
            (ho_g["pred"].cast(pl.Float64).to_numpy(), ho_g["real"].cast(pl.Float64).to_numpy()),
            title=title.format(group=group),
            units=units,
            density_cmap=density_cmap,
            left_label="CV",
            right_label=f"Holdout\n{label}",
            file_name=path,
            style=style,
            dpi=dpi,
        )
        written.append(path)
    return written


# --------------------------------------------------------------------------
# PCA overview
# --------------------------------------------------------------------------
NA_LABEL, OTHER_LABEL = "NA", "other"


def pca_embedding(
    X: np.ndarray, is_train: np.ndarray, n_components: int = 2, scale: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaler and PCA fitted on the training rows only; every row projected.

    Returns ``(coordinates, explained_variance_ratio)``.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    n_components = min(n_components, X.shape[1], int(is_train.sum()))
    if n_components < 2:
        raise ValueError(
            f"Cannot plot a 2D PCA with {X.shape[1]} features and {int(is_train.sum())} train samples"
        )
    if scale:
        X = StandardScaler().fit(X[is_train]).transform(X)
    pca = PCA(n_components=n_components).fit(X[is_train])
    return pca.transform(X), pca.explained_variance_ratio_


def collapse_levels(
    values: Sequence[Any], top_n: int = 10
) -> Tuple[np.ndarray, List[Any], int, bool, bool]:
    """Keep the ``top_n`` most frequent levels, bucket the rest as ``other``
    and nulls as ``NA``. Returns ``(labels, top_levels, n_levels, has_other,
    has_na)``."""
    series = pl.Series(values)
    non_null = series.drop_nulls()
    n_levels = non_null.n_unique()
    vc = non_null.value_counts(sort=True)
    level_col, count_col = vc.columns[0], vc.columns[1]
    vc = vc.sort([count_col, level_col], descending=[True, False])
    top_levels = vc[level_col].head(top_n).to_list()
    labels = np.array(
        [NA_LABEL if v is None else (v if v in top_levels else OTHER_LABEL) for v in series.to_list()],
        dtype=object,
    )
    return labels, top_levels, n_levels, bool((labels == OTHER_LABEL).any()), bool((labels == NA_LABEL).any())


def qualitative_colours(n: int) -> List[Any]:
    """A palette with no greys (reserved for ``other`` / ``NA``)."""
    plt = _plt()
    tab10, tab20 = plt.get_cmap("tab10"), plt.get_cmap("tab20")
    palette = [tab10(i) for i in range(10) if i != 7]
    palette += [tab20(i) for i in range(1, 20, 2) if i != 15]
    return [palette[i % len(palette)] for i in range(n)]


def pca_view_limits(
    coords: np.ndarray, clip_percentile: Optional[float] = 0.5
) -> Tuple[Any, Any, int]:
    """Axis limits from pooled percentiles so outliers do not squash the plot.
    Returns ``(xlim, ylim, n_outside)``; ``None`` limits when disabled."""
    if not clip_percentile:
        return None, None, 0
    lo, hi = clip_percentile, 100 - clip_percentile
    limits = []
    for i in range(2):
        a, b = np.asarray(np.percentile(coords[:, i], [lo, hi]))
        pad = 0.1 * (b - a if b > a else 1.0)
        limits.append((a - pad, b + pad))
    xlim, ylim = limits
    outside = (
        (coords[:, 0] < xlim[0]) | (coords[:, 0] > xlim[1])
        | (coords[:, 1] < ylim[0]) | (coords[:, 1] > ylim[1])
    )
    return xlim, ylim, int(outside.sum())


def _pca_factor_panel(
    ax: Any,
    coords: np.ndarray,
    is_train: np.ndarray,
    factor_values: Sequence[Any],
    name: str,
    evr: np.ndarray,
    xlim: Any,
    ylim: Any,
    top_n: int,
    legend_max_levels: int,
    other_colour: Any,
) -> None:
    from matplotlib.lines import Line2D

    labels, top_levels, n_levels, has_other, has_na = collapse_levels(factor_values, top_n)
    colors = dict(zip(top_levels, qualitative_colours(len(top_levels))))
    order = []
    if has_na:
        order.append((NA_LABEL, "dimgrey"))
    if has_other:
        order.append((OTHER_LABEL, other_colour))
    order += [(lvl, colors[lvl]) for lvl in reversed(top_levels)]
    for level, color in order:
        is_bg = level in (NA_LABEL, OTHER_LABEL)
        in_level = labels == level
        for mask, marker, edge in ((is_train, "o", "none"), (~is_train, "^", "k")):
            sel = in_level & mask
            if not sel.any():
                continue
            ax.scatter(
                coords[sel, 0], coords[sel, 1], s=14 if marker == "o" else 20,
                c=[color], marker=marker, alpha=0.5 if is_bg else 0.8,
                edgecolors=edge, linewidths=0.4, rasterized=True,
            )
    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    if n_levels > top_n:
        ax.set_title(f"{name} ({n_levels} levels, top {top_n} shown)")
    else:
        ax.set_title(f"{name} ({n_levels} levels)")
    n_legend = len(top_levels) + int(has_other) + int(has_na)
    if n_legend <= legend_max_levels:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=5, color=colors[lvl], label=str(lvl))
            for lvl in top_levels
        ]
        if has_other:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=5, color=other_colour, label=OTHER_LABEL))
        if has_na:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=5, color="dimgrey", label=NA_LABEL))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="x-small", frameon=False)
    else:
        logger.info("Omitting legend for %s: %d entries", name, n_legend)


def plot_pca_overview(
    coords: np.ndarray,
    evr: np.ndarray,
    is_train: np.ndarray,
    factors: Dict[str, Sequence[Any]],
    target: np.ndarray,
    target_name: str,
    title: str,
    top_n: int = 10,
    legend_max_levels: int = 12,
    other_colour: str = "lightgrey",
    clip_percentile: Optional[float] = 0.5,
    file_name: Optional[PathLike] = None,
    style: Style = None,
    dpi: int = 300,
) -> Any:
    """PCA scatter: one panel per grouping factor plus one coloured by the
    label (clipped to its 1st-99th percentile). Training samples are circles,
    test samples (projected) triangles."""
    from matplotlib.lines import Line2D

    plt = _plt()
    with _style(style):
        xlim, ylim, n_outside = pca_view_limits(coords, clip_percentile)
        names = list(factors)
        n = len(names) + 1
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7 * ncols, 5 * nrows), constrained_layout=True, squeeze=False
        )
        flat = axes.ravel()
        for ax, name in zip(flat, names):
            _pca_factor_panel(ax, coords, is_train, list(factors[name]), name, evr, xlim, ylim, top_n, legend_max_levels, other_colour)
        ax = flat[len(names)]
        y = np.asarray(target, dtype=float)
        lo, hi = np.asarray(np.percentile(y, [1, 99]))
        order = np.argsort(y)
        sc = None
        for mask, marker, edge, size in ((is_train, "o", "none", 14), (~is_train, "^", "k", 20)):
            idx = [i for i in order if mask[i]]
            sc = ax.scatter(
                coords[idx, 0], coords[idx, 1], c=y[idx], cmap="viridis", vmin=lo, vmax=hi,
                s=size, marker=marker, edgecolors=edge, linewidths=0.4, alpha=0.85, rasterized=True,
            )
        if xlim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
        ax.set_title(f"label: {target_name}")
        fig.colorbar(sc, ax=ax, shrink=0.8, label=target_name)
        for ax in flat[n:]:
            ax.set_visible(False)
        fig.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="", markersize=5, color="grey", label="train"),
                Line2D([0], [0], marker="^", linestyle="", markersize=6, markerfacecolor="grey", markeredgecolor="k", label="test (projected)"),
            ],
            loc="outside lower center", ncol=2, frameon=False,
        )
        fig.suptitle(
            title
            + (f"\n{n_outside} outlying points outside the axis limits are not shown" if n_outside else "")
        )
        return _finish(fig, file_name, dpi)


def plot_top_importances(
    names: Sequence[str],
    importances: Sequence[float],
    file_name: Optional[PathLike] = None,
    top: int = 10,
    style: Style = None,
    dpi: int = 300,
) -> Any:
    """Horizontal bars of the ``top`` largest importances (``names`` and
    ``importances`` already sorted, largest first)."""
    import cmasher as cmr

    plt = _plt()
    with _style(style):
        colours = cmr.take_cmap_colors("viridis_r", 10, cmap_range=(0.2, 0.8))
        fig = plt.figure(figsize=[5, 5])
        plt.barh(list(names)[:top], list(importances)[:top], color=colours, zorder=2)
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.grid(axis="y")
        plt.gca().invert_yaxis()
        return _finish(fig, file_name, dpi)


def short_taxon_names(names: Sequence[str]) -> List[str]:
    """``d__A;p__B;g__Foo`` -> ``Foo`` (the last rank, without its prefix)."""
    return [str(n).split(";")[-1].split("__")[-1] for n in names]


__all__ = [
    "prediction_stats",
    "plot_prediction_density",
    "plot_two_panels",
    "plot_predictions_by_group",
    "plot_logi_groups",
    "pca_embedding",
    "plot_pca_overview",
    "plot_top_importances",
    "short_taxon_names",
]
