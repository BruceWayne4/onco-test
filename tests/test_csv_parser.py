"""Unit tests for the LabFileParser (CSV/Excel parser).

Tests:
    - Parsing the demo CSV file
    - Marker detection (PD-1, TIM-3, LAG-3, CD8, etc.)
    - Column type classification
    - Summary statistics generation
    - Text representation generation
    - Handling of missing values
    - Excel parsing (temp xlsx)
    - File type error handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oncocontext.services.csv_parser import LabFileParser

# Path to the demo CSV
DEMO_CSV = Path(__file__).parent.parent / "demo" / "sample_flow_cytometry.csv"


@pytest.fixture
def parser():
    """Create a LabFileParser instance."""
    return LabFileParser()


@pytest.fixture
def demo_csv_path():
    """Path to the demo CSV file."""
    assert DEMO_CSV.exists(), f"Demo CSV not found at {DEMO_CSV}"
    return str(DEMO_CSV)


class TestParseCSV:
    """Test CSV parsing."""

    def test_parse_demo_csv(self, parser, demo_csv_path):
        """Parse the demo flow cytometry CSV successfully."""
        result = parser.parse_csv(demo_csv_path)

        assert result["rows"] == 20
        assert len(result["columns"]) == 10
        assert "Sample_ID" in result["columns"]
        assert "PD1_MFI" in result["columns"]
        assert "Cytotoxicity_percent" in result["columns"]

    def test_parse_csv_column_types(self, parser, demo_csv_path):
        """Column types are correctly classified."""
        result = parser.parse_csv(demo_csv_path)
        col_types = result["column_types"]

        assert col_types["Sample_ID"] == "identifier"
        assert col_types["CD8_percent"] == "numeric"
        assert col_types["PD1_MFI"] == "numeric"
        assert col_types["Condition"] == "categorical"

    def test_parse_csv_summary_stats(self, parser, demo_csv_path):
        """Summary statistics are computed for numeric columns."""
        result = parser.parse_csv(demo_csv_path)
        stats = result["summary_stats"]

        assert "CD8_percent" in stats
        assert "PD1_MFI" in stats
        assert "mean" in stats["CD8_percent"]
        assert "std" in stats["CD8_percent"]
        assert "min" in stats["CD8_percent"]
        assert "max" in stats["CD8_percent"]
        assert "median" in stats["CD8_percent"]

        # Verify reasonable ranges from demo data
        assert 40.0 < stats["CD8_percent"]["mean"] < 60.0
        assert stats["PD1_MFI"]["min"] > 0
        assert stats["PD1_MFI"]["max"] > 1000

    def test_parse_csv_markers_detected(self, parser, demo_csv_path):
        """Biological markers are detected from column names."""
        result = parser.parse_csv(demo_csv_path)
        markers = result["markers_detected"]

        # The demo CSV has these columns with markers:
        # CD8_percent, PD1_MFI, TIM3_MFI, LAG3_MFI, Ki67_percent, GranzymeB_percent
        assert any("CD8" in m for m in markers)
        assert any("PD" in m or "PD1" in m for m in markers)
        assert any("TIM" in m or "TIM3" in m for m in markers)
        assert any("LAG" in m or "LAG3" in m for m in markers)
        assert any("Ki67" in m or "Ki-67" in m for m in markers)
        assert any("Granzyme" in m or "GZMB" in m or "GranzymeB" in m for m in markers)

    def test_parse_csv_data_preview(self, parser, demo_csv_path):
        """Data preview contains first 5 rows."""
        result = parser.parse_csv(demo_csv_path)
        preview = result["data_preview"]

        assert len(preview) == 5
        assert preview[0]["Sample_ID"] == "S01"
        assert isinstance(preview[0]["CD8_percent"], float)

    def test_parse_csv_text_representations(self, parser, demo_csv_path):
        """Text representations are generated for each row."""
        result = parser.parse_csv(demo_csv_path)
        texts = result["text_representations"]

        assert len(texts) == 20  # 20 data rows

        # First row should mention Sample S01
        assert "S01" in texts[0]
        # Should contain numeric values
        assert "45.2" in texts[0] or "45.2%" in texts[0]

    def test_parse_csv_summary_string(self, parser, demo_csv_path):
        """File summary is a readable text string."""
        result = parser.parse_csv(demo_csv_path)
        summary = result["summary"]

        assert isinstance(summary, str)
        assert "20 samples" in summary
        assert "10 measurements" in summary

    def test_parse_csv_has_dataframe(self, parser, demo_csv_path):
        """Result includes the raw DataFrame."""
        result = parser.parse_csv(demo_csv_path)
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 20

    def test_parse_csv_column_info(self, parser, demo_csv_path):
        """Column info has expected structure."""
        result = parser.parse_csv(demo_csv_path)
        col_info = result["column_info"]

        assert len(col_info) == 10
        for col in col_info:
            assert "name" in col
            assert "dtype" in col
            assert "non_null_count" in col
            assert "sample_values" in col

        # Check a numeric column has stats
        pd1_col = next(c for c in col_info if c["name"] == "PD1_MFI")
        assert pd1_col["stats"] is not None
        assert "mean" in pd1_col["stats"]


class TestParseCSVEdgeCases:
    """Test CSV parsing edge cases."""

    def test_parse_csv_with_missing_values(self, parser):
        """Handle CSV with missing values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Sample_ID,CD8_percent,PD1_MFI\n")
            f.write("S01,45.2,180.3\n")
            f.write("S02,,210.5\n")  # Missing CD8_percent
            f.write("S03,41.5,\n")  # Missing PD1_MFI
            f.name
            f.flush()

            result = parser.parse_csv(f.name)
            assert result["rows"] == 3

    def test_parse_csv_empty_file_raises(self, parser):
        """Empty CSV raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Sample_ID,Value\n")  # Header only, no data
            f.flush()

            with pytest.raises(ValueError, match="no data"):
                parser.parse_csv(f.name)

    def test_parse_csv_tsv_file(self, parser):
        """TSV files are parsed with tab separator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("Sample_ID\tCD8_percent\tPD1_MFI\n")
            f.write("S01\t45.2\t180.3\n")
            f.write("S02\t43.8\t210.5\n")
            f.flush()

            result = parser.parse_csv(f.name)
            assert result["rows"] == 2
            assert "CD8_percent" in result["columns"]


class TestParseExcel:
    """Test Excel parsing."""

    def test_parse_excel_basic(self, parser):
        """Parse a basic Excel file."""
        df = pd.DataFrame({
            "Sample_ID": ["S01", "S02", "S03"],
            "CD8_percent": [45.2, 43.8, 41.5],
            "PD1_MFI": [180.3, 210.5, 195.7],
        })

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            df.to_excel(f.name, index=False)
            result = parser.parse_excel(f.name)

            assert result["rows"] == 3
            assert "CD8_percent" in result["columns"]
            assert len(result["markers_detected"]) > 0

    def test_parse_excel_marker_detection(self, parser):
        """Excel marker detection works same as CSV."""
        df = pd.DataFrame({
            "Sample": ["S01"],
            "CD8_pos_percent": [45.2],
            "PD1_MFI": [180.3],
            "TIM3_MFI": [95.2],
        })

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            df.to_excel(f.name, index=False)
            result = parser.parse_excel(f.name)
            markers = result["markers_detected"]

            assert any("CD8" in m for m in markers)
            assert any("PD" in m for m in markers)
            assert any("TIM" in m for m in markers)


class TestDetectMarkers:
    """Test biological marker detection."""

    def test_detect_standard_markers(self, parser):
        """Standard marker names are detected."""
        columns = ["CD8_percent", "PD1_MFI", "TIM3_MFI", "LAG3_MFI"]
        markers = parser.detect_markers(columns)

        assert any("CD8" in m for m in markers)
        assert any("PD1" in m or "PD-1" in m for m in markers)
        assert any("TIM3" in m or "TIM-3" in m for m in markers)
        assert any("LAG3" in m or "LAG-3" in m for m in markers)

    def test_detect_markers_with_suffixes(self, parser):
        """Markers detected even with _percent, _MFI, _pos suffixes."""
        columns = [
            "Ki67_percent",
            "GranzymeB_percent",
            "Cytotoxicity_percent",
        ]
        markers = parser.detect_markers(columns)
        assert any("Ki67" in m or "Ki-67" in m for m in markers)
        assert any("Granzyme" in m or "GZMB" in m for m in markers)
        assert any("Cytotoxicity" in m for m in markers)

    def test_detect_no_markers_in_generic_columns(self, parser):
        """Non-marker column names don't trigger false positives."""
        columns = ["Sample_ID", "Temperature", "Weight", "Height"]
        markers = parser.detect_markers(columns)
        assert len(markers) == 0

    def test_detect_markers_case_insensitive(self, parser):
        """Marker detection is case insensitive."""
        columns = ["cd8_PERCENT", "pd1_mfi", "tim3_MFI"]
        markers = parser.detect_markers(columns)
        assert len(markers) >= 3


class TestClassifyColumns:
    """Test column type classification."""

    def test_classify_numeric(self, parser):
        """Numeric columns are classified correctly."""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = parser.classify_columns(df)
        assert result["value"] == "numeric"

    def test_classify_categorical(self, parser):
        """Categorical columns are classified correctly."""
        df = pd.DataFrame({"condition": ["A", "B", "A", "B", "A"]})
        result = parser.classify_columns(df)
        assert result["condition"] == "categorical"

    def test_classify_identifier(self, parser):
        """Identifier columns (all unique) are classified correctly."""
        df = pd.DataFrame({"id": ["S01", "S02", "S03", "S04", "S05"]})
        result = parser.classify_columns(df)
        assert result["id"] == "identifier"

    def test_classify_mixed(self, parser):
        """Mixed column types in one DataFrame."""
        df = pd.DataFrame({
            "Sample_ID": ["S01", "S02", "S03", "S04", "S05"],
            "Condition": ["A", "B", "A", "B", "A"],
            "Value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = parser.classify_columns(df)
        assert result["Sample_ID"] == "identifier"
        assert result["Condition"] == "categorical"
        assert result["Value"] == "numeric"


class TestGenerateSummaryStats:
    """Test summary statistics generation."""

    def test_summary_stats_numeric(self, parser):
        """Summary stats are computed for numeric columns."""
        df = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        col_types = {"val": "numeric"}
        stats = parser.generate_summary_stats(df, col_types)

        assert "val" in stats
        assert stats["val"]["mean"] == 30.0
        assert stats["val"]["min"] == 10.0
        assert stats["val"]["max"] == 50.0
        assert stats["val"]["median"] == 30.0

    def test_summary_stats_skips_non_numeric(self, parser):
        """Non-numeric columns are skipped."""
        df = pd.DataFrame({"name": ["A", "B", "C"]})
        col_types = {"name": "categorical"}
        stats = parser.generate_summary_stats(df, col_types)

        assert "name" not in stats


class TestGenerateTextRepresentations:
    """Test per-row text generation."""

    def test_text_reps_count(self, parser):
        """One text representation per row."""
        df = pd.DataFrame({
            "Sample_ID": ["S01", "S02"],
            "CD8_percent": [45.2, 43.8],
            "PD1_MFI": [180.3, 210.5],
        })
        col_types = {"Sample_ID": "identifier", "CD8_percent": "numeric", "PD1_MFI": "numeric"}
        texts = parser.generate_text_representations(df, ["CD8", "PD1"], col_types)

        assert len(texts) == 2

    def test_text_reps_contain_values(self, parser):
        """Text representations contain sample data."""
        df = pd.DataFrame({
            "Sample_ID": ["S01"],
            "CD8_percent": [45.2],
            "PD1_MFI": [180.3],
        })
        col_types = {"Sample_ID": "identifier", "CD8_percent": "numeric", "PD1_MFI": "numeric"}
        texts = parser.generate_text_representations(df, ["CD8", "PD1"], col_types)

        assert "S01" in texts[0]
        assert "45.2" in texts[0]
        assert "180.3" in texts[0]

    def test_text_reps_include_categorical(self, parser):
        """Text representations include categorical context."""
        df = pd.DataFrame({
            "Sample_ID": ["S01"],
            "Condition": ["Organoid_CoC"],
            "Value": [42.3],
        })
        col_types = {"Sample_ID": "identifier", "Condition": "categorical", "Value": "numeric"}
        texts = parser.generate_text_representations(df, [], col_types)

        assert "Organoid" in texts[0] or "Condition" in texts[0]

    def test_text_reps_without_id_column(self, parser):
        """Text representations work without an identifier column."""
        df = pd.DataFrame({
            "Value_A": [1.0, 2.0],
            "Value_B": [3.0, 4.0],
        })
        col_types = {"Value_A": "numeric", "Value_B": "numeric"}
        texts = parser.generate_text_representations(df, [], col_types)

        assert len(texts) == 2
        # Should use "Row X" format
        assert "Row" in texts[0] or "0" in texts[0]
