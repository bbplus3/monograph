from monograph.narrate import chapterize as C


def test_clean_text_normalizes_punctuation():
    out = C.clean_text("He paused — then “spoke…” to her - softly")
    assert "—" not in out and "…" not in out and "“" not in out
    assert "  " not in out


def test_split_document_title_and_paragraphs():
    doc = "The Weight.\n\nFirst paragraph here.\n\nSecond paragraph here."
    title, paras = C.split_document(doc)
    assert title == "The Weight."
    assert paras == ["First paragraph here.", "Second paragraph here."]


def test_split_document_no_title_when_first_block_is_long():
    doc = ("This opening block is clearly prose and far too long to be a chapter "
           "title, so it should be treated as body.\n\nSecond paragraph.")
    title, paras = C.split_document(doc)
    assert title == ""
    assert len(paras) == 2


def test_split_chunks_respects_limit():
    text = " ".join(f"Sentence number {i} here." for i in range(40))
    chunks = C.split_chunks(text, limit=60)
    assert chunks and all(len(c) <= 60 for c in chunks)
