from monograph.narrate import pronunciation as P


def test_load_and_apply(tmp_path):
    f = tmp_path / "pron.txt"
    f.write_text("# a comment\nOMNI = AHM-nee\nEurope = YOOR-up  # trailing note\n",
                 encoding="utf-8")
    pairs = P.load(str(f))
    assert ("OMNI", "AHM-nee") in pairs
    out = P.apply("OMNI reached Europe.", pairs)
    assert out == "AHM-nee reached YOOR-up."


def test_missing_file_is_noop():
    assert P.load(None) == []
    assert P.apply("unchanged", []) == "unchanged"


def test_word_boundary_only():
    out = P.apply("omnipresent OMNI", [("OMNI", "AHM-nee")])
    assert out == "omnipresent AHM-nee"   # lowercase substring untouched
