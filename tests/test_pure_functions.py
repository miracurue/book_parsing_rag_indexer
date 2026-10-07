"""Тесты чистых функций (без API, сети и тяжёлых зависимостей)."""

from src.chunking import MarkdownChunker, parse_book_name, chunk_extracted_file
from src.clean_table_headers import (
    fix_latex_delimiters,
    remove_code_block_markers,
    remove_html_p_tags,
    remove_html_styles,
    remove_table_hash,
)
from src.normalize_tables import (
    MARKER_END,
    MARKER_START,
    count_markers,
    has_markers,
    normalize_table_boundaries,
)
from src.page_numbering import add_page_numbers, extract_page_number
from src.table_refs import (
    extract_figure_refs,
    extract_table_id_from_caption,
    extract_table_refs,
)


# ── page_numbering ────────────────────────────────────────────────────

def test_extract_page_number():
    assert extract_page_number("005.md") == 5
    assert extract_page_number("intro.md") is None


def test_add_page_numbers_tags_paragraphs_but_not_headings():
    out = add_page_numbers("# H\n\n\nabc  def\n\nxyz.....\n", "005.md")
    lines = out.split("\n")
    assert lines[0] == "# H"
    assert "<005> abc def" in lines
    assert "<005> xyz..." in lines


def test_add_page_numbers_without_digits_in_name_is_noop():
    assert add_page_numbers("text", "intro.md") == "text"


# ── table_refs ────────────────────────────────────────────────────────

def test_extract_table_refs_unique_in_order():
    text = "см. табл. 3.1, затем в таблице 5 и снова табл. 3.1"
    assert extract_table_refs(text) == ["3.1", "5"]
    assert extract_table_refs("без таблиц") == []


def test_extract_table_id_from_caption():
    assert extract_table_id_from_caption("Таблица 5.1 Характеристики") == "5.1"
    assert extract_table_id_from_caption("Описание процесса") is None


def test_extract_figure_refs():
    assert extract_figure_refs("см. рис. 2.3 и рисунок 4") == ["2.3", "4"]


# ── clean_table_headers ───────────────────────────────────────────────

def test_remove_table_hash():
    assert remove_table_hash("# Таблица 5") == ("Таблица 5", 1)


def test_remove_code_block_markers():
    text, n = remove_code_block_markers("```markdown\nx\n```")
    assert "```" not in text and "x" in text
    assert n == 2


def test_fix_latex_delimiters():
    assert fix_latex_delimiters(r"a \(x\) b \[y\]")[0] == "a $x$ b $$y$$"


def test_remove_html_styles_and_p_tags():
    assert remove_html_styles('<td style="a:b">1</td>')[0] == "<td>1</td>"
    assert remove_html_p_tags("<p class='a'>t</p>")[0] == "t"


# ── normalize_tables ──────────────────────────────────────────────────

_TABLE_TEXT = (
    "Текст\n"
    "<12> Таблица 3.1 Свойства\n"
    "<table><tr><td>1</td></tr></table>\n"
    "Далее текст\n"
)


def test_normalize_table_boundaries_adds_markers_around_table():
    out = normalize_table_boundaries(_TABLE_TEXT)
    assert has_markers(out)
    assert count_markers(out) == {"starts": 1, "ends": 1}
    lines = out.split("\n")
    start = lines.index(MARKER_START)
    end = lines.index(MARKER_END)
    assert "Таблица 3.1" in lines[start + 1]
    assert "<table>" in lines[end - 1]
    assert lines[0] == "Текст" and lines[-2] == "Далее текст"


def test_normalize_table_boundaries_is_idempotent():
    once = normalize_table_boundaries(_TABLE_TEXT)
    assert normalize_table_boundaries(once) == once


def test_normalize_without_caption_keeps_text():
    text = "Просто текст без таблиц\n"
    assert normalize_table_boundaries(text) == text


# ── chunking ──────────────────────────────────────────────────────────

def test_parse_book_name():
    assert parse_book_name("Автор, Название книги") == ("Автор", "Название книги")
    assert parse_book_name("Только название") == ("", "Только название")


def test_markdown_chunker_splits_and_keeps_headers_and_pages():
    md = "# Глава 1\n## Раздел\n<1> " + "Слово " * 60 + "\n<2> " + "Другое " * 40
    result = MarkdownChunker(chunk_size=100, overlap=20, min_chunk_size=30).chunk_document(md)
    chunks = result.text_chunks
    assert len(chunks) > 1
    for ch in chunks:
        meta = ch["metadata"]
        assert meta["chunk_type"] == "text"
        assert meta["headers"] == {1: "Глава 1", 2: "Раздел"}
        assert "<1>" not in ch["text"] and "<2>" not in ch["text"]
    # Страница фиксируется по тегу <N>, попавшему внутрь диапазона чанка
    assert chunks[0]["metadata"]["page_numbers"] == [1]
    all_pages = {p for ch in chunks for p in ch["metadata"]["page_numbers"]}
    assert all_pages == {1, 2}


def test_chunk_extracted_file_page_from_filename():
    md = "# Глава\n<86> Таблица 1\n| a | b |\n|---|---|\n| 1 | 2 |"
    chunks = chunk_extracted_file(md, "table", source_file="086.md", chunk_size=1200)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_numbers"] == [86]
