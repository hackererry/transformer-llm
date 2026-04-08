"""
Tests for document_converter — Parquet conversion and chunking
"""
import os
import tempfile
import pytest

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_parquet(path, columns, data):
    """Write a list of dicts to a parquet file."""
    table = pa.table({col: [row[col] for row in data] for col in columns})
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# ParquetExtractor.extract()
# ---------------------------------------------------------------------------

class TestParquetExtractorExtract:

    def test_basic_extract(self, tmp_path):
        """Small file, verify '列名: 值' format and metadata."""
        from src.data_processing.document_converter import ParquetExtractor

        data = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"},
        ]
        pq_path = str(tmp_path / "test.parquet")
        _write_parquet(pq_path, ["name", "age"], data)

        ext = ParquetExtractor(pq_path)
        text, meta = ext.extract()

        assert "name: Alice" in text
        assert "age: 30" in text
        assert "name: Bob" in text
        assert meta["record_count"] == 2
        assert "name" in meta["columns"]
        assert "age" in meta["columns"]

    def test_empty_parquet(self, tmp_path):
        """Parquet with no rows returns empty text."""
        from src.data_processing.document_converter import ParquetExtractor

        pq_path = str(tmp_path / "empty.parquet")
        table = pa.table({"col": pa.array([], type=pa.string())})
        pq.write_table(table, pq_path)

        ext = ParquetExtractor(pq_path)
        text, meta = ext.extract()

        assert text == ""
        assert meta["record_count"] == 0


# ---------------------------------------------------------------------------
# ParquetExtractor.extract_chunks()
# ---------------------------------------------------------------------------

class TestParquetExtractorExtractChunks:

    def test_single_chunk(self, tmp_path):
        """Small data fits in one chunk. Single column → no label prefix."""
        from src.data_processing.document_converter import ParquetExtractor

        data = [{"k": f"value_{i}"} for i in range(5)]
        pq_path = str(tmp_path / "test.parquet")
        _write_parquet(pq_path, ["k"], data)

        ext = ParquetExtractor(pq_path)
        chunks = ext.extract_chunks(max_file_size=1024 * 1024)  # 1MB

        assert len(chunks) == 1
        text, meta = chunks[0]
        # Single column: output value directly, no "k: " prefix
        assert "value_0" in text
        assert "value_4" in text
        assert "k: value_0" not in text
        assert meta["record_count"] == 5

    def test_multiple_chunks(self, tmp_path):
        """Small threshold forces multiple chunks."""
        from src.data_processing.document_converter import ParquetExtractor

        # Each row is ~20 bytes; use 50-byte threshold
        data = [{"col": f"data_row_{i:03d}"} for i in range(10)]
        pq_path = str(tmp_path / "test.parquet")
        _write_parquet(pq_path, ["col"], data)

        ext = ParquetExtractor(pq_path)
        chunks = ext.extract_chunks(max_file_size=50)

        assert len(chunks) >= 2
        total_records = sum(m["record_count"] for _, m in chunks)
        assert total_records == 10

    def test_empty_parquet_returns_empty(self, tmp_path):
        """Empty parquet returns empty list."""
        from src.data_processing.document_converter import ParquetExtractor

        pq_path = str(tmp_path / "empty.parquet")
        table = pa.table({"col": pa.array([], type=pa.string())})
        pq.write_table(table, pq_path)

        ext = ParquetExtractor(pq_path)
        chunks = ext.extract_chunks()
        assert chunks == []

    def test_none_values_skipped(self, tmp_path):
        """None and empty values are skipped."""
        from src.data_processing.document_converter import ParquetExtractor

        data = [
            {"a": "hello", "b": None},
            {"a": None, "b": "world"},
            {"a": "", "b": "test"},
        ]
        pq_path = str(tmp_path / "test.parquet")
        _write_parquet(pq_path, ["a", "b"], data)

        ext = ParquetExtractor(pq_path)
        text, _ = ext.extract()

        # None values should be skipped
        assert "a: hello" in text
        assert "b: world" in text
        assert "b: test" in text
        # Empty string should be skipped
        assert "a: \n" not in text

    def test_single_column_no_label(self, tmp_path):
        """Single-column parquet outputs values directly, no 'col: ' prefix."""
        from src.data_processing.document_converter import ParquetExtractor

        data = [{"text": "hello world"}]
        pq_path = str(tmp_path / "test.parquet")
        _write_parquet(pq_path, ["text"], data)

        ext = ParquetExtractor(pq_path)
        text, _ = ext.extract()
        assert text == "hello world"
        assert "text: " not in text

    def test_no_blank_lines_between_rows(self, tmp_path):
        """Rows are separated by single newline, not blank lines."""
        from src.data_processing.document_converter import ParquetExtractor

        data = [{"text": f"line{i}"} for i in range(3)]
        pq_path = str(tmp_path / "test.parquet")
        _write_parquet(pq_path, ["text"], data)

        ext = ParquetExtractor(pq_path)
        text, _ = ext.extract()
        assert text == "line0\nline1\nline2"
        assert "\n\n" not in text


# ---------------------------------------------------------------------------
# convert_parquet_to_txt()
# ---------------------------------------------------------------------------

class TestConvertParquetToTxt:

    def test_single_output(self, tmp_path):
        """Small file → single .txt output. Single column → no label prefix."""
        from src.data_processing.document_converter import convert_parquet_to_txt

        data = [{"x": f"val_{i}"} for i in range(3)]
        pq_path = str(tmp_path / "data.parquet")
        _write_parquet(pq_path, ["x"], data)

        output_files = convert_parquet_to_txt(pq_path)
        assert len(output_files) == 1
        assert output_files[0].endswith(".txt")
        assert os.path.exists(output_files[0])

        with open(output_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
        # Single column: output value directly
        assert "val_0" in content
        assert "x: val_0" not in content

    def test_multiple_outputs(self, tmp_path):
        """Large file with small threshold → multiple .txt files."""
        from src.data_processing.document_converter import convert_parquet_to_txt

        data = [{"col": f"data_row_number_{i:05d}"} for i in range(20)]
        pq_path = str(tmp_path / "big.parquet")
        _write_parquet(pq_path, ["col"], data)

        output_files = convert_parquet_to_txt(pq_path, max_file_size=80)
        assert len(output_files) >= 2
        for f in output_files:
            assert os.path.exists(f)

        # All output files should have _NNNN naming
        for f in output_files:
            assert "_" in os.path.basename(f)

    def test_custom_output_path(self, tmp_path):
        """Custom output path is respected for single chunk."""
        from src.data_processing.document_converter import convert_parquet_to_txt

        data = [{"c": "v"}]
        pq_path = str(tmp_path / "data.parquet")
        _write_parquet(pq_path, ["c"], data)

        out = str(tmp_path / "custom_output.txt")
        output_files = convert_parquet_to_txt(pq_path, output_path=out)
        assert len(output_files) == 1
        assert output_files[0] == out


# ---------------------------------------------------------------------------
# convert_to_txt routing
# ---------------------------------------------------------------------------

class TestConvertToTxtRouting:

    def test_parquet_routing(self, tmp_path):
        """convert_to_txt correctly routes .parquet files."""
        from src.data_processing.document_converter import convert_to_txt

        data = [{"k": "value"}]
        pq_path = str(tmp_path / "route.parquet")
        _write_parquet(pq_path, ["k"], data)

        result = convert_to_txt(pq_path)
        assert result is not None
        assert result.endswith(".txt")
        assert os.path.exists(result)

    def test_unsupported_format(self, tmp_path):
        """Unsupported format raises ValueError."""
        from src.data_processing.document_converter import convert_to_txt

        dummy = str(tmp_path / "test.xyz")
        with open(dummy, 'w') as f:
            f.write("data")

        with pytest.raises(ValueError, match="不支持的格式"):
            convert_to_txt(dummy)


# ---------------------------------------------------------------------------
# batch_convert with parquet
# ---------------------------------------------------------------------------

class TestBatchConvertWithParquet:

    def test_batch_parquet(self, tmp_path):
        """batch_convert handles parquet files."""
        from src.data_processing.document_converter import batch_convert

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        data = [{"col": f"row_{i}"} for i in range(5)]
        pq_path = str(input_dir / "batch.parquet")
        _write_parquet(pq_path, ["col"], data)

        results = batch_convert(str(input_dir), str(output_dir), max_workers=1)
        assert len(results) >= 1
        for f in results:
            assert os.path.exists(f)

    def test_batch_mixed_formats(self, tmp_path):
        """batch_convert handles mixed format directory."""
        from src.data_processing.document_converter import batch_convert

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a CSV file
        csv_path = str(input_dir / "test.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("name,age\nAlice,30\nBob,25\n")

        # Create a parquet file
        data = [{"col": "val"}]
        pq_path = str(input_dir / "test.parquet")
        _write_parquet(pq_path, ["col"], data)

        results = batch_convert(str(input_dir), str(output_dir), max_workers=1)
        assert len(results) >= 2  # at least csv + parquet
