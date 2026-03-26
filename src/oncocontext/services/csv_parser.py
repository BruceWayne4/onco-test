"""CSV/Excel parser — lab file ingestion, marker detection, summary generation.

Handles flow cytometry and other tabular lab data formats.
Parses files into structured data with auto-detected markers and text representations.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LabFileParser:
    """Parse CSV and Excel lab files into structured data with marker detection.

    Handles:
        - CSV (.csv, .tsv) and Excel (.xlsx, .xls) files
        - Encoding detection (UTF-8, Latin-1 fallback)
        - Column analysis: dtype, summary statistics, sample values
        - Biological marker detection in column names
        - Structured text summary generation
        - Per-row text representations for embedding
    """

    # Known biological markers for column name matching
    KNOWN_MARKERS: set[str] = {
        "CD3", "CD4", "CD8", "CD8a", "CD8b",
        "PD-1", "PD1", "PDCD1", "CD279",
        "TIM-3", "TIM3", "HAVCR2",
        "LAG-3", "LAG3", "CD223",
        "CTLA-4", "CTLA4", "CD152",
        "CD45", "CD45RA", "CD45RO",
        "CD25", "CD127", "FoxP3", "FOXP3",
        "Ki67", "Ki-67", "MKI67",
        "IFNg", "IFN-gamma", "IFNG",
        "TNFa", "TNF-alpha", "TNF",
        "IL-2", "IL2",
        "Granzyme B", "GZMB", "GranzymeB", "GrB",
        "Perforin", "PRF1",
        "TOX", "TCF1", "TCF-1", "TCF7",
        "T-bet", "Tbet", "TBX21",
        "EOMES", "Eomesodermin",
        "CD107a", "LAMP1",
        "CD28", "CD27", "CD57", "CD56",
        "HLA-DR", "CD38", "CD69",
        "PD-L1", "PDL1", "CD274",
        "TIGIT",
        "Cytotoxicity",
    }

    def _normalize(self, name: str) -> str:
        """Normalize a column name for fuzzy matching.

        Strips spaces, underscores, hyphens, lowercases, and removes
        common suffixes like _percent, _mfi, _pos.
        """
        n = name.strip().lower()
        # Remove common measurement suffixes
        n = re.sub(r'[_\s]+(percent|pct|mfi|pos|neg|ratio|mean|median)$', '', n)
        # Remove underscores, spaces, hyphens for matching
        n = re.sub(r'[_\s-]+', '', n)
        return n

    def detect_markers(self, column_names: list[str]) -> list[str]:
        """Detect biological markers in column names.

        Uses case-insensitive fuzzy matching against KNOWN_MARKERS.

        Args:
            column_names: List of column name strings.

        Returns:
            List of recognized marker names (canonical forms).
        """
        detected = []
        seen_normalized = set()

        # Build a normalized lookup from KNOWN_MARKERS
        marker_lookup: dict[str, str] = {}
        for marker in self.KNOWN_MARKERS:
            key = marker.lower().replace("-", "").replace(" ", "").replace("_", "")
            marker_lookup[key] = marker

        for col in column_names:
            col_normalized = self._normalize(col)

            # Try exact normalized match
            if col_normalized in marker_lookup and col_normalized not in seen_normalized:
                seen_normalized.add(col_normalized)
                detected.append(marker_lookup[col_normalized])
                continue

            # Try substring match — check if any known marker is a substring of the column
            for key, canonical in marker_lookup.items():
                if key in col_normalized and key not in seen_normalized:
                    seen_normalized.add(key)
                    detected.append(canonical)
                    break

        return detected

    def classify_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """Classify columns as numeric, categorical, or identifier.

        Args:
            df: pandas DataFrame.

        Returns:
            Dict mapping column name → 'numeric', 'categorical', or 'identifier'.
        """
        result: dict[str, str] = {}

        for col in df.columns:
            series = df[col]

            # Try to coerce to numeric
            numeric_series = pd.to_numeric(series, errors="coerce")
            non_null_count = series.dropna().shape[0]
            numeric_count = numeric_series.dropna().shape[0]

            if non_null_count > 0 and numeric_count / max(non_null_count, 1) > 0.8:
                result[col] = "numeric"
            elif non_null_count > 0:
                unique_ratio = series.nunique() / max(non_null_count, 1)
                # If nearly all values are unique, likely an identifier
                if unique_ratio > 0.9 and non_null_count > 3:
                    result[col] = "identifier"
                else:
                    result[col] = "categorical"
            else:
                result[col] = "categorical"

        return result

    def generate_summary_stats(self, df: pd.DataFrame, column_types: dict[str, str]) -> dict[str, dict]:
        """Generate summary statistics for numeric columns.

        Args:
            df: pandas DataFrame.
            column_types: Column type classification from classify_columns.

        Returns:
            Dict mapping column name → {mean, std, min, max, median, q25, q75}.
        """
        stats: dict[str, dict] = {}

        for col, col_type in column_types.items():
            if col_type != "numeric":
                continue

            numeric_series = pd.to_numeric(df[col], errors="coerce").dropna()
            if numeric_series.empty:
                continue

            stats[col] = {
                "mean": round(float(numeric_series.mean()), 2),
                "std": round(float(numeric_series.std()), 2),
                "min": round(float(numeric_series.min()), 2),
                "max": round(float(numeric_series.max()), 2),
                "median": round(float(numeric_series.median()), 2),
                "q25": round(float(numeric_series.quantile(0.25)), 2),
                "q75": round(float(numeric_series.quantile(0.75)), 2),
            }

        return stats

    def generate_file_summary(
        self,
        df: pd.DataFrame,
        markers: list[str],
        column_types: dict[str, str],
        summary_stats: dict[str, dict],
    ) -> str:
        """Generate a structured natural language summary of the file.

        Args:
            df: pandas DataFrame.
            markers: Detected biological markers.
            column_types: Column type classification.
            summary_stats: Summary statistics for numeric columns.

        Returns:
            Human-readable summary string.
        """
        n_rows = len(df)
        n_cols = len(df.columns)
        numeric_cols = [c for c, t in column_types.items() if t == "numeric"]

        parts = [
            f"Flow cytometry data with {n_rows} samples and {n_cols} measurements."
        ]

        if markers:
            parts.append(f"Detected markers: {', '.join(markers)}.")

        # Add key statistics for important columns
        stat_lines = []
        for col in numeric_cols[:6]:  # Limit to 6 most important
            if col in summary_stats:
                s = summary_stats[col]
                # Detect if this is a percentage or MFI-type column
                col_clean = col.replace("_", " ")
                if "percent" in col.lower() or "pct" in col.lower():
                    stat_lines.append(
                        f"{col_clean} ranges from {s['min']}% to {s['max']}% "
                        f"(mean: {s['mean']}%)"
                    )
                elif "mfi" in col.lower():
                    stat_lines.append(
                        f"{col_clean} ranges from {s['min']} to {s['max']} "
                        f"(mean: {s['mean']})"
                    )
                else:
                    stat_lines.append(
                        f"{col_clean} ranges from {s['min']} to {s['max']} "
                        f"(mean: {s['mean']})"
                    )

        if stat_lines:
            parts.append("Key statistics: " + "; ".join(stat_lines) + ".")

        return " ".join(parts)

    def generate_text_representations(
        self,
        df: pd.DataFrame,
        markers: list[str],
        column_types: dict[str, str],
    ) -> list[str]:
        """Convert each row into a text representation for embedding.

        Example output for one row:
        "Sample S07: CD8 percentage 55.8%, PD1 MFI 1850.3, TIM3 MFI 810.2,
         LAG3 MFI 450.5, Cytotoxicity 42.3%, Ki67 32.1%"

        Args:
            df: pandas DataFrame.
            markers: Detected biological markers.
            column_types: Column type classification.

        Returns:
            List of text strings, one per row.
        """
        texts = []

        # Find an identifier column for sample naming
        id_col = None
        for col, col_type in column_types.items():
            if col_type == "identifier":
                id_col = col
                break

        # Find categorical columns for context
        cat_cols = [c for c, t in column_types.items() if t == "categorical"]
        # Find numeric columns for values
        num_cols = [c for c, t in column_types.items() if t == "numeric"]

        for idx, row in df.iterrows():
            parts = []

            # Add sample identifier
            if id_col:
                parts.append(f"Sample {row[id_col]}")
            else:
                parts.append(f"Row {idx}")

            # Add categorical context (Condition, Timepoint, etc.)
            for col in cat_cols:
                val = row.get(col)
                if pd.notna(val):
                    col_clean = col.replace("_", " ")
                    parts.append(f"{col_clean}: {val}")

            # Add numeric values
            for col in num_cols:
                val = row.get(col)
                if pd.notna(val):
                    col_clean = col.replace("_", " ")
                    try:
                        float_val = float(val)
                        if "percent" in col.lower() or "pct" in col.lower():
                            parts.append(f"{col_clean} {float_val:.1f}%")
                        elif "mfi" in col.lower():
                            parts.append(f"{col_clean} {float_val:.1f}")
                        else:
                            parts.append(f"{col_clean} {float_val:.1f}")
                    except (ValueError, TypeError):
                        parts.append(f"{col_clean} {val}")

            texts.append(", ".join(parts))

        return texts

    def parse_csv(self, file_path: str) -> dict:
        """Parse a CSV or TSV file into structured data.

        Args:
            file_path: Path to the CSV/TSV file.

        Returns:
            Dict with:
                - rows: int
                - columns: list[str]
                - column_types: dict[str, str]
                - summary_stats: dict[str, dict]
                - markers_detected: list[str]
                - data_preview: list[dict] (first 5 rows)
                - text_representations: list[str]
                - data: pandas DataFrame
        """
        path = Path(file_path)

        # Detect separator
        sep = "\t" if path.suffix.lower() == ".tsv" else ","

        # Try UTF-8 first, then Latin-1
        df = None
        for encoding in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(str(path), sep=sep, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                raise ValueError(f"Failed to parse CSV file: {exc}") from exc

        if df is None:
            raise ValueError(f"Failed to read file with any supported encoding: {file_path}")

        if df.empty:
            raise ValueError("File contains no data rows")

        return self._analyze_dataframe(df)

    def parse_excel(self, file_path: str, sheet_name: str | int = 0) -> dict:
        """Parse an Excel file (.xlsx, .xls) into structured data.

        Args:
            file_path: Path to the Excel file.
            sheet_name: Sheet name or index (default: 0 = first sheet).

        Returns:
            Same dict structure as parse_csv.
        """
        try:
            df = pd.read_excel(str(file_path), sheet_name=sheet_name)
        except Exception as exc:
            raise ValueError(f"Failed to parse Excel file: {exc}") from exc

        if df.empty:
            raise ValueError("File contains no data rows")

        return self._analyze_dataframe(df)

    def _analyze_dataframe(self, df: pd.DataFrame) -> dict:
        """Analyze a DataFrame and produce structured output.

        Args:
            df: pandas DataFrame.

        Returns:
            Dict with analysis results.
        """
        columns = list(df.columns)
        column_types = self.classify_columns(df)
        markers = self.detect_markers(columns)
        summary_stats = self.generate_summary_stats(df, column_types)
        file_summary = self.generate_file_summary(df, markers, column_types, summary_stats)
        text_reps = self.generate_text_representations(df, markers, column_types)

        # Build per-column info
        column_info = []
        for col in columns:
            series = df[col]
            non_null = int(series.dropna().shape[0])
            sample_vals = series.dropna().head(5).tolist()
            # Convert numpy types to native Python for JSON serialization
            sample_vals = [
                float(v) if isinstance(v, (np.floating, float)) else
                int(v) if isinstance(v, (np.integer,)) else
                str(v)
                for v in sample_vals
            ]

            info: dict = {
                "name": col,
                "dtype": str(series.dtype),
                "non_null_count": non_null,
                "sample_values": sample_vals,
                "stats": summary_stats.get(col),
            }
            column_info.append(info)

        # Data preview (first 5 rows as dicts)
        preview = []
        for _, row in df.head(5).iterrows():
            row_dict = {}
            for col in columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.floating, float)):
                    row_dict[col] = float(val)
                elif isinstance(val, (np.integer,)):
                    row_dict[col] = int(val)
                else:
                    row_dict[col] = str(val)
            preview.append(row_dict)

        return {
            "rows": len(df),
            "columns": columns,
            "column_types": column_types,
            "column_info": column_info,
            "summary_stats": summary_stats,
            "markers_detected": markers,
            "data_preview": preview,
            "summary": file_summary,
            "text_representations": text_reps,
            "data": df,
        }
