"""The filled-profile check: parent rows exist and cover their children."""

import polars as pl

from microbiome_ml.wrangle.profiles import (
    FILLED,
    UNFILLED,
    TaxonomicProfiles,
)


def _profiles(rows):
    df = pl.DataFrame(
        rows, schema=["sample", "taxonomy", "relabund"], orient="row"
    )
    return TaxonomicProfiles(df.lazy(), check_filled=False)


def _filled_rows(leaves, decimals=None):
    """Cumulative rank rows (domain..species); each row sums its leaves."""
    prefixes = ["d", "p", "c", "o", "f", "g", "s"]
    totals = {}
    for path, value in leaves.items():
        parts = ["A"] + path.split("/")
        for depth in range(1, len(parts) + 1):
            key = ";".join(f"{prefixes[i]}__{parts[i]}" for i in range(depth))
            totals[key] = totals.get(key, 0.0) + value
    return [
        ("S1", tax, v if decimals is None else round(v, decimals))
        for tax, v in totals.items()
    ]


def test_filled_check_tolerates_rounded_abundances():
    leaves = {
        f"X/X{i}/o/f/g/s{i}": v
        for i, v in enumerate([0.1234, 0.2345, 0.3456, 0.1111], 1)
    }
    profiles = _profiles(_filled_rows(leaves, decimals=3))
    assert profiles._detect_layout() == FILLED


def test_filled_check_rejects_parents_smaller_than_their_children():
    # Parent rows exist but hold only their own reads, as in an unfilled
    # profile: the phylum is far below the sum of its classes.
    rows = [
        ("S1", "d__A", 0.9),
        ("S1", "d__A;p__X", 0.01),
        ("S1", "d__A;p__X;c__X1", 0.4),
        ("S1", "d__A;p__X;c__X2", 0.3),
    ]
    assert _profiles(rows)._detect_layout() == UNFILLED


def test_unrounded_small_parents_get_no_rounding_allowance():
    # Unrounded values, parents holding only their own reads, all below the
    # 3-decimal rounding unit: no tolerance applies, so this stays unfilled.
    rows = [
        ("S1", "d__A", 0.00031),
        ("S1", "d__A;p__X", 0.00011),
        ("S1", "d__A;p__X;c__X1", 0.00017),
        ("S1", "d__A;p__X;c__X2", 0.00023),
    ]
    assert _profiles(rows)._detect_layout() == UNFILLED
