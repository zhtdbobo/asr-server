import json

import pytest

from textgrid_to_json import convert_file, main, parse_textgrid, read_textgrid

LONG_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 10
tiers? <exists>
size = 3
item []:
    item [1]:
        class = "IntervalTier"
        name = "F_SPK0002"
        xmin = 0
        xmax = 10
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 2
            text = ""
        intervals [2]:
            xmin = 2
            xmax = 4
            text = "我说""好的""！"
        intervals [3]:
            xmin = 4
            xmax = 10
            text = "   "
    item [2]:
        class = "IntervalTier"
        name = "R0001_M0001_M_SPK0001"
        xmin = 0
        xmax = 10
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 1.25e+00
            text = ""
        intervals [2]:
            xmin = 1.25e+00
            xmax = 3.5
            text = "第一行
第二行 ! [123] 42"
        intervals [3]:
            xmin = 3.5
            xmax = 10
            text = ""
    item [3]:
        class = "TextTier"
        name = "events"
        xmin = 0
        xmax = 10
        points: size = 1
        points [1]:
            number = 3
            mark = "提示"
"""

SHORT_TEXTGRID = """File type = "ooTextFile short"
"TextGrid"
0
5
<exists>
2
"IntervalTier"
"F_SPK0093"
0
5
2
0
1.5
"你好"
1.5
5
""
"IntervalTier"
"R0001_M0001_M_SPK0094"
0
5
2
0
1
""
1
5
"收到"
"""

EXPECTED_LONG = [
    {"speaker": "SPK0001", "start": 1.25, "end": 3.5, "text": "第一行\n第二行 ! [123] 42"},
    {"speaker": "SPK0002", "start": 2.0, "end": 4.0, "text": '我说"好的"！'},
]
EXPECTED_SHORT = [
    {"speaker": "SPK0093", "start": 0.0, "end": 1.5, "text": "你好"},
    {"speaker": "SPK0094", "start": 1.0, "end": 5.0, "text": "收到"},
]


def test_full_format_preserves_overlap_quotes_multiline_and_sorts_speakers():
    assert parse_textgrid(LONG_TEXTGRID) == EXPECTED_LONG


def test_short_format_and_comments_outside_strings():
    text = SHORT_TEXTGRID.replace("\n2\n", '\n2 ! ignore 99 "fake tier"\n', 1)
    assert parse_textgrid(text) == EXPECTED_SHORT


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-be"])
def test_unicode_encodings_and_chinese_json_output(tmp_path, encoding):
    source, target = tmp_path / "会议.TextGrid", tmp_path / "out" / "会议.json"
    data = LONG_TEXTGRID.encode(encoding)
    if encoding == "utf-16-be":
        data = b"\xfe\xff" + data
    source.write_bytes(data)
    assert convert_file(source, target) == 2
    result = target.read_text(encoding="utf-8")
    assert "第一行" in result
    assert json.loads(result) == EXPECTED_LONG
    assert not list(target.parent.glob("*.tmp"))


def test_near_field_filename_fallback_and_generic_speaker_name():
    text = SHORT_TEXTGRID.replace("F_SPK0093", "transcript")
    assert parse_textgrid(text, "R0003_M0046_F_SPK0093")[0]["speaker"] == "SPK0093"
    assert parse_textgrid(text)[0]["speaker"] == "transcript"


def test_explicit_legacy_encoding_and_wrong_encoding_error(tmp_path):
    source = tmp_path / "会议.TextGrid"
    source.write_bytes(SHORT_TEXTGRID.encode("gb18030"))
    with pytest.raises(ValueError, match="--encoding"):
        read_textgrid(source)
    assert read_textgrid(source, "gb18030") == EXPECTED_SHORT


@pytest.mark.parametrize(
    "text",
    [
        LONG_TEXTGRID.rsplit('mark = "提示"', 1)[0],
        SHORT_TEXTGRID.replace('"收到"', '"未闭合'),
        SHORT_TEXTGRID.replace("\n1.5\n", "\n-1\n", 1),
        SHORT_TEXTGRID.replace("\n1.5\n", "\n1e999\n", 1),
        SHORT_TEXTGRID.replace("\n2\n", "\n1.5\n", 1),
        SHORT_TEXTGRID + '\n"extra data"\n',
        "this is not a TextGrid",
    ],
)
def test_corrupt_annotations_fail_instead_of_producing_partial_results(tmp_path, text):
    source, target = tmp_path / "bad.TextGrid", tmp_path / "existing.json"
    source.write_text(text, encoding="utf-8")
    target.write_text("previous result", encoding="utf-8")
    with pytest.raises(ValueError):
        convert_file(source, target)
    assert target.read_text(encoding="utf-8") == "previous result"


def test_empty_transcript_is_a_valid_empty_list():
    text = SHORT_TEXTGRID.replace('"你好"', '""').replace('"收到"', '""')
    assert parse_textgrid(text) == []
    assert parse_textgrid('"ooTextFile"\n"TextGrid"\n0 1 <absent>') == []


def test_directory_cli_keeps_subdirectories_and_continues_after_failure(tmp_path, capsys):
    source, output = tmp_path / "dataset", tmp_path / "reference"
    far, near = source / "far" / "textgrid_dir", source / "near" / "textgrid_dir"
    far.mkdir(parents=True)
    near.mkdir(parents=True)
    (far / "meeting.TextGrid").write_text(LONG_TEXTGRID, encoding="utf-8")
    (near / "meeting.TEXTGRID").write_text(SHORT_TEXTGRID, encoding="utf-8")
    (far / "broken.TextGrid").write_text("broken", encoding="utf-8")
    (far / "notes.txt").write_text("ignore", encoding="utf-8")
    assert main([str(source), "--output-dir", str(output)]) == 1
    for mic, expected in (("far", EXPECTED_LONG), ("near", EXPECTED_SHORT)):
        target = output / mic / "textgrid_dir" / "meeting.json"
        assert json.loads(target.read_text(encoding="utf-8")) == expected
    assert not (output / "far" / "textgrid_dir" / "broken.json").exists()
    assert "broken.TextGrid" in capsys.readouterr().err


def test_single_file_cli_and_invalid_invocations(tmp_path):
    source, target = tmp_path / "meeting.TextGrid", tmp_path / "reference.json"
    source.write_text(SHORT_TEXTGRID, encoding="utf-8")
    assert main([str(source), "--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == EXPECTED_SHORT
    for args in (
        [str(tmp_path), "--output", str(target)],
        [str(tmp_path / "missing")],
        [str(source), "--output", str(source)],
    ):
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 2


def test_binary_format_gives_actionable_error(tmp_path):
    source = tmp_path / "binary.TextGrid"
    source.write_bytes(b"ooBinaryFile\x08TextGrid")
    with pytest.raises(ValueError, match="Praat"):
        read_textgrid(source)
