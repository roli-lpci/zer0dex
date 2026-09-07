"""Tests for zer0dex seed — file collection and markdown chunking."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zer0dex.seed import collect_files, chunk_markdown, get_all_for_user, search_for_user


class TestCollectFiles:
    def test_single_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("hello")
        result = collect_files([str(f)])
        assert len(result) == 1
        assert result[0] == f

    def test_directory_finds_md_files(self, tmp_path):
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / "c.txt").write_text("c")
        result = collect_files([str(tmp_path)])
        names = [r.name for r in result]
        assert "a.md" in names
        assert "b.md" in names
        assert "c.txt" not in names

    def test_nested_directory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("nested")
        result = collect_files([str(tmp_path)])
        assert any(r.name == "nested.md" for r in result)

    def test_missing_path_returns_empty(self, tmp_path):
        result = collect_files([str(tmp_path / "nonexistent")])
        assert result == []

    def test_empty_directory(self, tmp_path):
        result = collect_files([str(tmp_path)])
        assert result == []

    def test_multiple_sources(self, tmp_path):
        f1 = tmp_path / "one.md"
        f2 = tmp_path / "two.md"
        f1.write_text("one")
        f2.write_text("two")
        result = collect_files([str(f1), str(f2)])
        assert len(result) == 2


class TestChunkMarkdown:
    def test_single_section(self):
        text = "# Title\nSome content here."
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        assert "Title" in chunks[0]

    def test_splits_on_h2(self):
        text = "# Title\nIntro\n## Section A\nContent A\n## Section B\nContent B"
        chunks = chunk_markdown(text)
        assert len(chunks) == 3  # title+intro, section A, section B

    def test_does_not_emit_heading_only_preamble(self):
        text = "# Memory\n\n## Project Atlas\nDeployment target: staging."
        chunks = chunk_markdown(text)
        assert chunks == [text]

    def test_no_empty_chunks(self):
        text = "\n\n\n## A\nstuff\n\n\n## B\nmore\n\n"
        chunks = chunk_markdown(text)
        for c in chunks:
            assert c.strip() != ""

    def test_large_section_gets_split(self):
        big = "## Big\n" + ("word " * 1000)
        chunks = chunk_markdown(big, max_chunk=500)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 600  # some tolerance for word boundaries

    def test_empty_input(self):
        assert chunk_markdown("") == []

    def test_whitespace_only(self):
        assert chunk_markdown("   \n  \n  ") == []

    def test_preserves_content(self):
        text = "## Section\nImportant fact: zer0dex achieves 91% recall."
        chunks = chunk_markdown(text)
        joined = " ".join(chunks)
        assert "91% recall" in joined

    def test_h3_does_not_split(self):
        text = "## Main\nContent\n### Sub\nMore content"
        chunks = chunk_markdown(text)
        assert len(chunks) == 1  # h3 should not cause a split


class TestMem0ApiCompatibility:
    def test_get_all_uses_current_filters_api(self):
        memory = MagicMock()
        memory.get_all.return_value = {"results": []}

        assert get_all_for_user(memory, "agent") == {"results": []}
        memory.get_all.assert_called_once_with(
            filters={"user_id": "agent"}, top_k=100
        )

    def test_get_all_expands_past_current_default_limit(self):
        memory = MagicMock()
        first_page = {"results": [{"memory": str(i)} for i in range(100)]}
        complete = {"results": [{"memory": str(i)} for i in range(125)]}
        memory.get_all.side_effect = [first_page, complete]

        assert get_all_for_user(memory, "agent") == complete
        assert memory.get_all.call_args_list[0].kwargs == {
            "filters": {"user_id": "agent"},
            "top_k": 100,
        }
        assert memory.get_all.call_args_list[1].kwargs == {
            "filters": {"user_id": "agent"},
            "top_k": 200,
        }

    def test_search_uses_current_filters_api(self):
        memory = MagicMock()
        memory.search.return_value = {"results": []}

        assert search_for_user(memory, "question", "agent", 5) == {"results": []}
        memory.search.assert_called_once_with("question", filters={"user_id": "agent"}, top_k=5)

    def test_get_all_falls_back_to_legacy_entity_argument(self):
        memory = MagicMock()
        memory.get_all.side_effect = [TypeError("unexpected filters"), {"results": []}]

        assert get_all_for_user(memory, "agent") == {"results": []}
        assert memory.get_all.call_args_list[1].kwargs == {"user_id": "agent"}
