import csv
import itertools
import json
import random
import re

import pytest

from compare_transcripts import (
    compare_segments,
    discover_json,
    edit_distance,
    load_segments,
    main,
    match_files,
    maximum_assignment,
    normalize_text,
    summarize,
)


def segment(speaker, start, end, text):
    return {"speaker": speaker, "start": start, "end": end, "text": text}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8-sig")
    return path


def test_different_boundaries_and_permuted_names_do_not_create_errors():
    reference = [
        segment("SPK8013", 0, 1, "你好，"),
        segment("SPK8013", 1, 2, "开始。"),
        segment("SPK8014", 2, 4, "好的"),
    ]
    hypothesis = [segment("SPEAKER_02", 0, 2, "你好开始"), segment("SPEAKER_00", 2, 4, "好的")]
    result = compare_segments(reference, hypothesis)
    assert result["speaker_mapping"] == {"SPEAKER_02": "SPK8013", "SPEAKER_00": "SPK8014"}
    assert result["text_score"]["cer"] == 0
    assert result["diarization"]["der"] == 0
    assert result["reference_segments"] != result["hypothesis_segments"]


def test_users_split_turns_and_overlap_keep_text_score_separate_from_silence_gaps():
    reference = [
        segment("SPK8013", 6.9, 18.29, "第一段"),
        segment("SPK8013", 18.73, 22.43, "第二段"),
        segment("SPK8013", 23.65, 24.43, "你给阐一下吧。"),
        segment("SPK8014", 23.7, 24.04, "啊。"),
    ]
    hypothesis = [
        segment("发言人A", 6.9, 24.43, "第一段第二段你给阐一下吧"),
        segment("发言人B", 23.7, 24.04, "啊"),
    ]
    result = compare_segments(reference, hypothesis)
    assert result["text_score"]["cer"] == 0
    assert result["diarization"]["false_alarm_seconds"] == pytest.approx(1.66)
    assert result["diarization"]["reference_overlap_seconds"] == pytest.approx(0.34)
    assert result["diarization"]["confusion_seconds"] == 0


def test_mapping_is_global_optimum_not_greedy():
    assert set(maximum_assignment([[9, 8], [8, 0]])) == {(0, 1), (1, 0)}
    assert maximum_assignment([]) == []
    assert maximum_assignment([[]]) == []
    assert len(maximum_assignment([[5], [3], [1]])) == 1


def test_hungarian_mapping_matches_exhaustive_permutations_with_unequal_speaker_counts():
    rng = random.Random(42)
    for rows in range(1, 5):
        for columns in range(1, 5):
            weights = [[rng.randrange(15) / 10 for _ in range(columns)] for _ in range(rows)]
            size = max(rows, columns)
            optimum = max(
                sum(weights[r][c] for r, c in enumerate(order) if r < rows and c < columns)
                for order in itertools.permutations(range(size))
            )
            assigned = maximum_assignment(weights)
            assert sum(weights[r][c] for r, c in assigned) == pytest.approx(optimum)


def test_text_attached_to_wrong_time_mapped_speaker_is_an_error():
    result = compare_segments(
        [segment("A", 0, 2, "甲乙"), segment("B", 2, 4, "丙丁")],
        [segment("X", 0, 2, "丙丁"), segment("Y", 2, 4, "甲乙")],
    )
    assert result["diarization"]["der"] == 0
    assert result["text_score"]["cer"] == 1


def test_missing_and_extra_speakers_are_not_discarded():
    result = compare_segments(
        [segment("A", 0, 1, "你好"), segment("B", 1, 2, "再见")],
        [segment("X", 0, 1, "你好"), segment("Z", 3, 4, "多余")],
    )
    assert result["speaker_mapping"] == {"X": "A"}
    assert result["text_score"]["edit_distance"] == 4
    assert result["text_score"]["cer"] == 1
    assert result["diarization"]["miss_seconds"] == 1
    assert result["diarization"]["false_alarm_seconds"] == 1
    assert result["diarization"]["der"] == 1
    assert len(result["speakers"]) == 3


def test_reference_overlap_scores_speaker_seconds_and_single_speaker_limit():
    result = compare_segments(
        [segment("A", 0, 2, "甲"), segment("B", 1, 3, "乙")], [segment("X", 0, 3, "甲乙")]
    )
    assert result["diarization"]["reference_speaker_seconds"] == 4
    assert result["diarization"]["miss_seconds"] == 1
    assert result["diarization"]["confusion_seconds"] == 1
    assert result["diarization"]["der"] == 0.5


def test_overlapping_segments_of_same_speaker_count_only_once_in_der():
    result = compare_segments(
        [segment("A", 0, 2, "你"), segment("A", 1, 3, "好")], [segment("X", 0, 3, "你好")]
    )
    assert result["diarization"]["reference_speaker_seconds"] == 3
    assert result["diarization"]["der"] == 0
    assert result["text_score"]["cer"] == 0


def test_empty_reference_has_null_rates_but_preserves_insertions_and_false_alarms():
    result = compare_segments([], [segment("X", 0, 2, "多余")])
    assert result["text_score"]["cer"] is None
    assert result["text_score"]["edit_distance"] == 2
    assert result["diarization"]["der"] is None
    assert result["diarization"]["false_alarm_seconds"] == 2
    assert compare_segments([], [])["speaker_mapping"] == {}


def test_normalization_preserves_fillers_and_can_ignore_explicit_annotation_tags():
    assert normalize_text("嗯，ＡbＣ１２３！ 哈哈。") == "嗯abc123哈哈"
    assert normalize_text("[噪声]啊<笑声>", re.compile(r"\[[^\]]*\]|<[^>]*>")) == "啊"
    assert normalize_text("[噪声]啊") == "噪声啊"


def slow_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        next_row = [i]
        for j, cb in enumerate(b, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (ca != cb)))
        row = next_row
    return row[-1]


def test_bit_vector_distance_matches_independent_dynamic_programming():
    rng = random.Random(20261105)
    for _ in range(300):
        a = "".join(rng.choices("你好abc", k=rng.randrange(35)))
        b = "".join(rng.choices("你好abc", k=rng.randrange(35)))
        assert edit_distance(a, b) == slow_distance(a, b)
    assert edit_distance("a" * 5000, "b" * 5000) == 5000


def test_loads_bom_full_output_and_sorts_without_requiring_same_segmentation(tmp_path):
    path = save(
        tmp_path / "full.json", {"segments": [segment("A", 2, 3, "后"), segment("A", 0, 2, "前")]}
    )
    assert [s["text"] for s in load_segments(path)] == ["前", "后"]


@pytest.mark.parametrize(
    "data",
    [
        {},
        [{"speaker": "A"}],
        [segment("", 0, 1, "好")],
        [segment("A", True, 1, "好")],
        [segment("A", -1, 1, "好")],
        [segment("A", 1, 0, "好")],
        [segment("A", 0, float("nan"), "好")],
    ],
)
def test_invalid_input_does_not_produce_a_zero_error_score(tmp_path, data):
    with pytest.raises((TypeError, ValueError)):
        load_segments(save(tmp_path / "bad.json", data))


def test_far_field_matching_does_not_accidentally_match_near_field_annotations(tmp_path):
    reference, hypothesis = tmp_path / "reference", tmp_path / "hypothesis"
    far = save(reference / "far" / "R8001_M8004.json", [])
    near = save(reference / "near" / "R8001_M8004_N_SPK8013.json", [])
    hyp = save(hypothesis / "R8001_M8004_MS801.wav.json", [])
    pairs = match_files(reference, hypothesis)
    pair = next(p for p in pairs if p.get("hypothesis_file") == str(hyp))
    assert pair["reference_file"] == str(far)
    assert any(
        p.get("reference_file") == str(near) and p.get("status") == "missing_hypothesis"
        for p in pairs
    )


def test_discovery_prefers_canonical_but_accepts_full_only_and_skips_metadata(tmp_path):
    canonical = save(tmp_path / "a.wav.json", [])
    save(tmp_path / "a.wav.full.json", {"segments": []})
    full_only = save(tmp_path / "b.wav.full.json", {"segments": []})
    save(tmp_path / "batch-summary.json", {})
    save(tmp_path / "comparison-report.json", {})
    assert set(discover_json(tmp_path)) == {canonical, full_only}


def test_duplicate_reference_files_are_reported_instead_of_guessed(tmp_path):
    refs, hyps = tmp_path / "r", tmp_path / "h"
    save(refs / "one" / "meeting.json", [])
    save(refs / "two" / "meeting.json", [])
    save(hyps / "meeting.wav.json", [])
    assert match_files(refs, hyps)[0]["status"] == "ambiguous_reference"


def test_summary_uses_total_characters_and_duration_not_mean_of_file_rates():
    one = compare_segments([segment("A", 0, 1, "好")], [segment("X", 0, 1, "坏")])
    nine = compare_segments(
        [segment("A", 0, 9, "一二三四五六七八九")], [segment("X", 0, 9, "一二三四五六七八九")]
    )
    summary = summarize(
        [
            dict(one, status="compared"),
            dict(nine, status="compared"),
            {"status": "missing_hypothesis"},
        ]
    )
    assert summary["text_score"]["cer"] == 0.1
    assert summary["compared_files"] == 2


def test_cli_writes_reports_and_continues_past_missing_or_invalid_files(tmp_path):
    refs, hyps, out = tmp_path / "r", tmp_path / "h", tmp_path / "out"
    save(refs / "R8001_M8004.json", [segment("SPK8013", 0, 2, "你好开始")])
    save(
        hyps / "R8001_M8004_MS801.wav.json",
        [segment("X", 0, 1, "你好"), segment("X", 1, 2, "开始")],
    )
    save(refs / "missing.json", [])
    save(refs / "bad.json", [])
    save(hyps / "bad.wav.json", {"wrong": []})
    assert (
        main(["--reference", str(refs), "--hypothesis", str(hyps), "--output-dir", str(out)]) == 1
    )
    report = json.loads((out / "comparison-report.json").read_text(encoding="utf-8"))
    assert report["summary"]["status_counts"] == {
        "compared": 1,
        "error": 1,
        "missing_hypothesis": 1,
    }
    assert report["summary"]["text_score"]["cer"] == 0
    assert "参考：你好开始" in (out / "comparison-report.txt").read_text(encoding="utf-8")
    with (out / "comparison-files.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3


def test_cli_can_explicitly_compare_two_files_with_unrelated_names(tmp_path):
    ref = save(tmp_path / "reference.json", [segment("A", 0, 1, "你好")])
    hyp = save(tmp_path / "output.json", {"segments": [segment("B", 0, 1, "你好")]})
    assert (
        main(
            [
                "--reference",
                str(ref),
                "--hypothesis",
                str(hyp),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
