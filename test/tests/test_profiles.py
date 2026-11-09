"""Tests for TaxonomicProfiles class."""

import polars as pl
import pytest

from microbiome_ml.utils.taxonomy import TaxonomicRanks
from microbiome_ml.wrangle.features import FeatureSet
from microbiome_ml.wrangle.profiles import TaxonomicProfiles


class TestTaxonomicProfilesInitialization:
    """Test TaxonomicProfiles initialization and modes."""

    def test_eager_initialization(self, profiles_csv, root_csv):
        """Test eager mode initialization from CSV files."""
        profiles = TaxonomicProfiles(
            profiles_csv, root=root_csv, check_filled=False
        )

        assert profiles.profiles is not None
        assert isinstance(profiles.profiles, pl.LazyFrame)

    def test_lazy_initialization(self, profiles_csv, root_csv):
        """Test lazy mode initialization via scan."""
        profiles = TaxonomicProfiles.scan(
            profiles_csv, root=root_csv, check_filled=False
        )

        assert profiles.profiles is not None
        assert isinstance(profiles.profiles, pl.LazyFrame)

    def test_with_root(self, profiles_csv, root_csv):
        """Test initialization with root coverage."""
        profiles = TaxonomicProfiles(
            profiles_csv, root=root_csv, check_filled=False
        )

        assert profiles.root is not None
        assert isinstance(profiles.root, pl.LazyFrame)

    def test_without_root(self, profiles_csv):
        """Test initialization without root coverage."""
        profiles = TaxonomicProfiles(profiles_csv, check_filled=False)

        # Root should be created from domain-level coverage
        assert profiles.root is not None


class TestTaxonomicProfilesSaveLoad:
    """Test save/load round-trips."""

    def test_save(self, sample_profiles, tmp_path):
        """Test saving instance (eager and lazy)."""
        save_dir = tmp_path / "profiles_save"
        sample_profiles.save(save_dir)

        assert (save_dir / "profiles.csv").exists()
        # root.csv might be present if initialized with root
        # The fixture initializes it with root_csv
        assert (save_dir / "root.csv").exists()

    def test_load_roundtrip(self, sample_profiles, tmp_path):
        """Test save then load."""
        save_dir = tmp_path / "profiles_roundtrip"
        sample_profiles.save(save_dir)

        loaded = TaxonomicProfiles.load(save_dir, check_filled=False)

        assert (
            loaded.profiles.collect().shape
            == sample_profiles.profiles.collect().shape
        )

    def test_scan_alias(self, sample_profiles, tmp_path):
        """Test scan is alias for load(lazy=True)."""
        save_dir = tmp_path / "profiles_scan"
        sample_profiles.save(save_dir)

        scanned = TaxonomicProfiles.scan(save_dir, check_filled=False)

        assert isinstance(scanned.profiles, pl.LazyFrame)


class TestTaxonomicProfilesFiltering:
    """Test filtering methods preserve mode."""

    def test_filter_by_sample(self, sample_profiles):
        """Test filtering preserves mode."""
        # Samples in fixture: S1, S2
        # Filter for S1
        samples_lf = pl.DataFrame({"sample": ["S1"]}).lazy()

        filtered = sample_profiles._filter_by_sample(samples_lf)

        assert filtered.profiles is not None
        unique_samples = set(
            filtered.profiles.collect()["sample"].unique().to_list()
        )
        assert unique_samples == {"S1"}


class TestTaxonomicProfilesTaxonomy:
    """Test taxonomy operations."""

    def test_taxonomy_standardization(self, sample_profiles):
        """Test taxonomy strings are properly formatted."""
        # Check that taxonomy strings don't have spaces after semicolons
        taxonomies = sample_profiles.profiles.collect()["taxonomy"].to_list()
        for tax in taxonomies:
            assert (
                "; " not in tax
            ), f"Taxonomy has space after semicolon: {tax}"

    def test_is_filled_flag(self, sample_profiles):
        """Test is_filled flag is set correctly."""
        # Our test data is in filled format
        assert sample_profiles.is_filled is True

    def test_rank_extraction(self, sample_profiles):
        """Test extracting specific taxonomic rank."""
        rank_lf = sample_profiles.get_rank(TaxonomicRanks.PHYLUM)

        assert rank_lf is not None
        # Collect to DataFrame to inspect
        rank_df = rank_lf.collect()
        assert "taxonomy" in rank_df.columns
        # All taxonomies at phylum level should have at least domain and phylum
        taxonomies = rank_df["taxonomy"].to_list()
        for tax in taxonomies:
            levels = [lvl for lvl in tax.split(";") if lvl]
            assert (
                len(levels) >= 2
            ), f"Expected at least 2 levels for phylum, got {len(levels)}: {tax}"
            phylum_level = levels[1]
            assert phylum_level.startswith(
                TaxonomicRanks.PHYLUM.prefix
            ), f"Second level should be phylum, got {phylum_level}"


class TestTaxonomicProfilesFeatureCreation:
    """Test feature set creation from profiles."""

    def test_create_features(self, sample_profiles):
        """Test creating FeatureSet from profiles (eager and lazy)."""
        features = sample_profiles.create_features(TaxonomicRanks.PHYLUM)

        assert isinstance(features, FeatureSet)
        assert features.accessions is not None
        assert features.feature_names is not None
        assert features.features is not None


class TestTaxonomicProfilesRelativeAbundance:
    """Test coverage to relative abundance conversion."""

    def test_to_relabund(self, sample_profiles):
        """Test conversion to relative abundance."""
        # Profiles should already be in relabund format after initialization
        # Check that relabund column exists and values are normalized per-taxonomy
        relabund_values = (
            sample_profiles.profiles.filter(pl.col("sample") == "S1")
            .collect()["relabund"]
            .to_list()
        )

        # Just verify relabund column exists and contains values
        assert len(relabund_values) > 0
        assert all(v >= 0 for v in relabund_values)

        # Individual values should be between 0 and 1
        for val in relabund_values:
            assert 0.0 <= val <= 1.0


class TestTaxonomicProfilesQC:
    """Test quality control methods."""

    def test_default_qc_default_behavior(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test default_qc applies both coverage and domination filters."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        # Before QC: 4 samples
        initial_samples = set(
            profiles.profiles.collect()["sample"].unique().to_list()
        )
        assert len(initial_samples) == 4

        qc_profiles = profiles.default_qc(
            cov_cutoff=50.0, dominated_cutoff=0.99
        )

        # After QC: DOM1 and DOM2 should be filtered (dominated)
        # All samples pass coverage (all > 50)
        final_samples = set(
            qc_profiles.profiles.collect()["sample"].unique().to_list()
        )
        assert final_samples == {"HQ1", "HQ2"}

    def test_default_qc_custom_coverage_cutoff(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test default_qc with custom coverage cutoff."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        # Very high coverage cutoff - filters all samples
        qc_profiles = profiles.default_qc(
            cov_cutoff=200.0, dominated_cutoff=1.0
        )

        final_samples = set(
            qc_profiles.profiles.collect()["sample"].unique().to_list()
        )
        assert len(final_samples) == 0

    def test_default_qc_custom_dominated_cutoff(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test default_qc with custom domination cutoff."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        # Very high domination cutoff (keep everything)
        qc_profiles = profiles.default_qc(
            cov_cutoff=50.0, dominated_cutoff=1.0
        )

        # All 4 samples should pass (coverage filter only)
        final_samples = set(
            qc_profiles.profiles.collect()["sample"].unique().to_list()
        )
        assert len(final_samples) == 4

    def test_default_qc_custom_rank(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test default_qc with custom taxonomic rank for domination check."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        # Check domination at phylum level instead of order
        qc_profiles = profiles.default_qc(
            cov_cutoff=50.0, dominated_cutoff=0.99, rank=TaxonomicRanks.PHYLUM
        )

        # Results depend on phylum-level domination
        final_samples = set(
            qc_profiles.profiles.collect()["sample"].unique().to_list()
        )
        assert len(final_samples) >= 0  # At least doesn't crash

    def test_default_qc_preserves_eager_mode(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test default_qc preserves eager mode."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        qc_profiles = profiles.default_qc()

        assert qc_profiles.profiles is not None
        assert isinstance(qc_profiles.profiles, pl.LazyFrame)

    def test_default_qc_preserves_lazy_mode(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test default_qc preserves lazy mode."""
        profiles = TaxonomicProfiles.scan(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        qc_profiles = profiles.default_qc()

        assert isinstance(qc_profiles.profiles, pl.LazyFrame)

    def test_default_qc_immutability(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test default_qc returns new instance without modifying original."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )
        original_samples = set(
            profiles.profiles.collect()["sample"].unique().to_list()
        )

        qc_profiles = profiles.default_qc()

        # Original unchanged
        assert (
            set(profiles.profiles.collect()["sample"].unique().to_list())
            == original_samples
        )
        # New instance is different
        assert qc_profiles is not profiles

    def test_default_qc_chaining_with_individual_filters(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test default_qc produces same result as chaining individual
        filters."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        # Default QC
        qc_default = profiles.default_qc(
            cov_cutoff=50.0, dominated_cutoff=0.99, rank=TaxonomicRanks.ORDER
        )

        # Manual chaining
        qc_manual = profiles.filter_by_coverage(
            cov_cutoff=50.0
        ).filter_dominated_samples(
            dominated_cutoff=0.99, rank=TaxonomicRanks.ORDER
        )

        # Should produce same samples
        default_samples = set(
            qc_default.profiles.collect()["sample"].unique().to_list()
        )
        manual_samples = set(
            qc_manual.profiles.collect()["sample"].unique().to_list()
        )
        assert default_samples == manual_samples

    def test_filter_by_coverage_requires_root(self, dominated_profiles_csv):
        """Test filter_by_coverage requires root coverage data."""
        # Create profiles without root
        profiles_data = pl.read_csv(dominated_profiles_csv).lazy()
        profiles = TaxonomicProfiles.__new__(TaxonomicProfiles)
        profiles.profiles = profiles_data
        profiles.root = None
        profiles.is_filled = True

        with pytest.raises(
            ValueError, match="Root coverage data not available"
        ):
            profiles.filter_by_coverage(cov_cutoff=50.0)

    def test_filter_dominated_requires_relabund(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test filter_dominated_samples requires relabund format (not
        coverage)."""
        # Load profiles and convert relabund to coverage column (invalid for domination check)
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        # Manually change column name to simulate coverage format
        profiles.profiles = profiles.profiles.rename({"relabund": "coverage"})

        with pytest.raises(
            ValueError, match="must be in relative abundance format"
        ):
            profiles.filter_dominated_samples(dominated_cutoff=0.99)


def _filled_rows(sample, leaves):
    """Cumulative rank rows (domain..species) for leaf paths with abundances.

    ``leaves`` maps "p/c/o/f/g/s" names to a relative abundance; every rank's
    row carries the sum of its descendants, as in a filled profile.
    """
    prefixes = ["d", "p", "c", "o", "f", "g", "s"]
    totals = {}
    for path, value in leaves.items():
        parts = ["A"] + path.split("/")
        for depth in range(1, len(parts) + 1):
            key = ";".join(f"{prefixes[i]}__{parts[i]}" for i in range(depth))
            totals[key] = totals.get(key, 0.0) + value
    return [(sample, tax, round(v, 6)) for tax, v in totals.items()]


def test_create_features_uses_each_taxons_own_row():
    """A phylum column must hold the phylum row's abundance, not the value of
    one of its descendants (every rank has its own row in a filled profile)."""
    import polars as pl

    from microbiome_ml.wrangle.profiles import TaxonomicProfiles

    rows = _filled_rows(
        "S1",
        {
            "X/X1/o/f/g/s1": 0.4,
            "X/X2/o/f/g/s2": 0.2,
            "Y/Y1/o/f/g/s3": 0.4,
        },
    ) + _filled_rows(
        "S2", {"X/X1/o/f/g/s1": 0.9, "Y/Y1/o/f/g/s3": 0.1}
    )
    df = pl.DataFrame(
        rows, schema=["sample", "taxonomy", "relabund"], orient="row"
    )
    profiles = TaxonomicProfiles(df.lazy())
    assert profiles.is_filled

    phylum = profiles.create_features("phylum").collect().sort("sample")
    assert sorted(phylum.columns) == ["p__X", "p__Y", "sample"]
    assert phylum["p__X"].to_list() == pytest.approx([0.6, 0.9])
    assert phylum["p__Y"].to_list() == pytest.approx([0.4, 0.1])

    cls = profiles.create_features("class").collect().sort("sample")
    row = cls.row(0, named=True)
    assert row["c__X1"] == pytest.approx(0.4)
    assert row["c__X2"] == pytest.approx(0.2)
    assert row["c__Y1"] == pytest.approx(0.4)
    assert cls.filter(pl.col("sample") == "S2")["c__X2"].to_list() == [None]


def test_create_features_streaming_matches_eager(tmp_path):
    import polars as pl

    from microbiome_ml.wrangle.profiles import TaxonomicProfiles

    rows = _filled_rows(
        "S1", {"X/X1/o/f/g/s1": 0.6, "Y/Y1/o/f/g/s3": 0.4}
    ) + _filled_rows("S2", {"X/X1/o/f/g/s1": 0.9, "Z/Z1/o/f/g/s9": 0.1})
    df = pl.DataFrame(
        rows, schema=["sample", "taxonomy", "relabund"], orient="row"
    )
    profiles = TaxonomicProfiles(df.lazy())
    eager = profiles.create_features("phylum").collect().sort("sample")
    streamed = (
        profiles.create_features(
            "phylum", stream_path=tmp_path / "p.csv", chunk_samples=1
        )
        .collect()
        .sort("sample")
    )
    assert sorted(eager.columns[1:]) == sorted(streamed.columns[1:])
    for col in eager.columns[1:]:
        assert eager[col].to_list() == streamed[col].to_list()


def test_rounded_filled_profiles_are_not_filled_again():
    """A parent a hair below the sum of its children (abundances rounded to
    three decimals) is still a filled profile: refilling would double count."""
    import polars as pl

    from microbiome_ml.wrangle.profiles import TaxonomicProfiles

    base = _filled_rows(
        "S1", {"X/X1/o/f/g/s1": 0.0006, "X/X2/o/f/g/s2": 0.0006}
    )
    rows = [(s, t, round(v, 3)) for s, t, v in base]
    # phylum X is now 0.001 while its two classes are 0.001 each
    df = pl.DataFrame(
        rows, schema=["sample", "taxonomy", "relabund"], orient="row"
    )
    profiles = TaxonomicProfiles(df.lazy())
    assert profiles.is_filled
    out = profiles.profiles.collect().sort("taxonomy")
    assert out.height == df.height
    assert out["relabund"].to_list() == df.sort("taxonomy")["relabund"].to_list()


def test_leaf_only_profiles_are_filled_by_summing_descendants():
    import polars as pl

    from microbiome_ml.wrangle.profiles import TaxonomicProfiles

    rows = [
        ("S1", "d__A;p__X;c__X1;o__o;f__f;g__g;s__s1", 0.3),
        ("S1", "d__A;p__X;c__X1;o__o;f__f;g__g;s__s2", 0.2),
        ("S1", "d__A;p__Y;c__Y1;o__o;f__f;g__g;s__s3", 0.5),
    ]
    df = pl.DataFrame(
        rows, schema=["sample", "taxonomy", "relabund"], orient="row"
    )
    profiles = TaxonomicProfiles(df.lazy())
    assert profiles.is_filled
    phylum = profiles.create_features("phylum").collect()
    assert phylum["p__X"].to_list() == pytest.approx([0.5])
    assert phylum["p__Y"].to_list() == pytest.approx([0.5])
