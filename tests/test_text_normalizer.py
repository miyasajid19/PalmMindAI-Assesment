from app.rag.ingestion.text_normalizer import TextNormalizer


def test_normalize_removes_noise_and_preserves_text():
    raw_text = (
        "  EMPLOYEE POLICY   \r\n"
        "\r\n"
        "This   is   a sample paragraph.\x00\r\n"
        "\r\n"
        "Section 1:\tLeave Policy\r\n"
        "Employees are entitled to annual leave."
    )

    normalized = TextNormalizer.normalize(raw_text)

    assert "\x00" not in normalized
    assert "\r" not in normalized
    assert "EMPLOYEE POLICY" in normalized
    assert "Section 1:" in normalized
    assert "Leave Policy" in normalized
    assert "This is a sample paragraph." in normalized


def test_normalize_and_combine_pages():
    pages = [
        "  First page text.   ",
        "Second page text.\x00",
        "Third page text.",
    ]

    normalized_pages = TextNormalizer.normalize_pages(pages)
    combined = TextNormalizer.combine_pages(pages)

    assert normalized_pages == [
        "First page text.",
        "Second page text.",
        "Third page text.",
    ]
    assert "[PAGE 1]" in combined
    assert "[PAGE 2]" in combined
    assert "[PAGE 3]" in combined


def test_normalize_preserves_zero_width_joiners():
    text = "नम\u200dस्ते न\u200cमस्ते"

    normalized = TextNormalizer.normalize(text)

    assert "\u200d" in normalized
    assert "\u200c" in normalized
