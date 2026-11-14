import json
from datetime import date

from monograph.ingest import parse_diary
from monograph.types import DiaryEntry


def test_dated_header_text(tmp_path):
    f = tmp_path / "diary.txt"
    f.write_text(
        "Monday 1/2/19\nFirst day. It rained.\n\n"
        "Wedensday 1/4/219\nTypo weekday and a 3-digit year that means 2019.\n",
        encoding="utf-8")
    entries = parse_diary(str(f))
    assert len(entries) == 2
    assert entries[0].entry_date == date(2019, 1, 2)
    assert entries[1].entry_date == date(2019, 1, 4)        # 219 salvaged -> 2019
    assert "rained" in entries[0].text


def test_impossible_date_is_flagged(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("13/40/20\nnot a real date\n\n2/3/20\nreal one\n", encoding="utf-8")
    entries, flagged = parse_diary(str(f), return_flagged=True)
    assert any(e.entry_date == date(2020, 2, 3) for e in entries)
    assert flagged                                           # the impossible header noted


def test_json_records(tmp_path):
    f = tmp_path / "entries.json"
    f.write_text(json.dumps([
        {"date": "2021-06-02", "text": "entry one"},
        {"Date": "6/9/2021", "Entry": "entry two"},
    ]), encoding="utf-8")
    entries = parse_diary(str(f))
    assert len(entries) == 2
    assert all(isinstance(e, DiaryEntry) for e in entries)
    assert entries[0].entry_date == date(2021, 6, 2)
    assert entries[1].entry_date == date(2021, 6, 9)


def test_dict_of_date_to_text(tmp_path):
    f = tmp_path / "map.json"
    f.write_text(json.dumps({"2020-01-01": "new year", "2020-01-02": "day two"}),
                 encoding="utf-8")
    entries = parse_diary(str(f))
    assert {e.text for e in entries} == {"new year", "day two"}


def test_directory_sorts_by_date(tmp_path):
    (tmp_path / "b.md").write_text("5/1/21\nlater entry\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("1/1/21\nearlier entry\n", encoding="utf-8")
    entries = parse_diary(str(tmp_path))
    assert [e.entry_date for e in entries] == [date(2021, 1, 1), date(2021, 5, 1)]
