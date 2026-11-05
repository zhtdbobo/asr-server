"""Compare reference and ASR segments without requiring equal labels or boundaries.

Standard library only. DER uses a maximum-overlap, one-to-one speaker mapping,
zero collar and includes overlap. CER concatenates each mapped speaker's text.
These are diagnostics of the supplied segments, not the official AliMeeting scorer.
"""

import argparse
import csv
import io
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

AUDIO_SUFFIXES = {
    ".wav",
    ".wave",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".amr",
    ".aif",
    ".aiff",
    ".caf",
    ".mka",
}
MEETING_NAME = re.compile(r"^(R\d+_M\d+)(?:_MS\d+)?$", re.IGNORECASE)
REPORT_NAMES = {"batch-summary.json", "comparison-report.json"}


def load_segments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    records = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise TypeError("需要片段数组，或包含 segments 数组的完整结果对象")
    segments = []
    for index, item in enumerate(records, 1):
        if not isinstance(item, dict):
            raise TypeError(f"第 {index} 个片段不是对象")
        speaker, text = item.get("speaker"), item.get("text")
        start, end = item.get("start"), item.get("end")
        if not isinstance(speaker, str) or not speaker.strip() or not isinstance(text, str):
            raise ValueError(f"第 {index} 个片段需要非空 speaker 和字符串 text")
        if (
            any(
                isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
                for t in (start, end)
            )
            or start < 0
            or end < start
        ):
            raise ValueError(f"第 {index} 个片段的 start/end 无效，时间单位应为秒")
        segments.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return sorted(segments, key=lambda s: (s["start"], s["end"], s["speaker"]))


def normalize_text(text: str, ignore_pattern=None) -> str:
    if ignore_pattern:
        text = ignore_pattern.sub("", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] in "LNM")


def edit_distance(reference: str, hypothesis: str) -> int:
    """Exact Levenshtein distance with bit vectors, suitable for long meetings."""
    prefix = 0
    while prefix < min(len(reference), len(hypothesis)) and reference[prefix] == hypothesis[prefix]:
        prefix += 1
    reference, hypothesis = reference[prefix:], hypothesis[prefix:]
    suffix = 0
    while (
        suffix < min(len(reference), len(hypothesis))
        and reference[-suffix - 1] == hypothesis[-suffix - 1]
    ):
        suffix += 1
    if suffix:
        reference, hypothesis = reference[:-suffix], hypothesis[:-suffix]
    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    size = len(reference)
    if not size:
        return len(hypothesis)
    masks = defaultdict(int)
    for index, char in enumerate(reference):
        masks[char] |= 1 << index
    mask = (1 << size) - 1
    high = 1 << (size - 1)
    positive, negative, distance = mask, 0, size
    for char in hypothesis:
        equal = masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        distance += bool(plus & high) - bool(minus & high)
        plus = ((plus << 1) | 1) & mask
        minus = (minus << 1) & mask
        positive = (minus | ~(vertical | plus)) & mask
        negative = plus & vertical
    return distance


def timeline_spans(reference, hypothesis):
    """Sweep boundaries; duplicate/overlapping turns of one speaker count once."""
    events = defaultdict(list)
    for side, segments in enumerate((reference, hypothesis)):
        for segment in segments:
            if segment["end"] > segment["start"]:
                events[segment["start"]].append((side, segment["speaker"], 1))
                events[segment["end"]].append((side, segment["speaker"], -1))
    times = sorted(events)
    active = [Counter(), Counter()]
    for index, start in enumerate(times[:-1]):
        for side, speaker, change in events[start]:
            active[side][speaker] += change
            if not active[side][speaker]:
                del active[side][speaker]
        yield start, times[index + 1], set(active[0]), set(active[1])


def maximum_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Exact Hungarian assignment; pad missing speakers with zero-weight dummies."""
    rows, columns = len(weights), len(weights[0]) if weights else 0
    size = max(rows, columns)
    u, v = [0.0] * (size + 1), [0.0] * (size + 1)
    matched, previous = [0] * (size + 1), [0] * (size + 1)
    for row in range(1, size + 1):
        matched[0] = row
        column = 0
        minimum, used = [math.inf] * (size + 1), [False] * (size + 1)
        while True:
            used[column] = True
            current_row = matched[column]
            delta, next_column = math.inf, 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                weight = (
                    weights[current_row - 1][candidate - 1]
                    if current_row <= rows and candidate <= columns
                    else 0
                )
                cost = -weight - u[current_row] - v[candidate]
                if cost < minimum[candidate]:
                    minimum[candidate], previous[candidate] = cost, column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[matched[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if not matched[column]:
                break
        while column:
            matched[column] = matched[previous[column]]
            column = previous[column]
    return [
        (matched[c] - 1, c - 1)
        for c in range(1, size + 1)
        if 0 < matched[c] <= rows and c <= columns
    ]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def compare_segments(reference, hypothesis, ignore_pattern=None):
    reference_speakers = sorted({s["speaker"] for s in reference})
    hypothesis_speakers = sorted({s["speaker"] for s in hypothesis})
    spans = list(timeline_spans(reference, hypothesis))
    overlaps = defaultdict(float)
    for start, end, refs, hyps in spans:
        for hyp in hyps:
            for ref in refs:
                overlaps[hyp, ref] += end - start
    weights = [[overlaps[hyp, ref] for ref in reference_speakers] for hyp in hypothesis_speakers]
    mapping = {
        hypothesis_speakers[h]: reference_speakers[r]
        for h, r in maximum_assignment(weights)
        if weights[h][r] > 0
    }
    diarization = {
        "reference_speaker_seconds": 0.0,
        "miss_seconds": 0.0,
        "false_alarm_seconds": 0.0,
        "confusion_seconds": 0.0,
        "reference_overlap_seconds": 0.0,
    }
    for start, end, refs, hyps in spans:
        duration = end - start
        correct = len(refs & {mapping[h] for h in hyps if h in mapping})
        diarization["reference_speaker_seconds"] += duration * len(refs)
        diarization["miss_seconds"] += duration * max(len(refs) - len(hyps), 0)
        diarization["false_alarm_seconds"] += duration * max(len(hyps) - len(refs), 0)
        diarization["confusion_seconds"] += duration * (min(len(refs), len(hyps)) - correct)
        if len(refs) > 1:
            diarization["reference_overlap_seconds"] += duration
    diarization["error_seconds"] = sum(
        diarization[key] for key in ("miss_seconds", "false_alarm_seconds", "confusion_seconds")
    )
    diarization["der"] = ratio(
        diarization["error_seconds"], diarization["reference_speaker_seconds"]
    )

    texts = [defaultdict(list), defaultdict(list)]
    for side, segments in enumerate((reference, hypothesis)):
        for segment in segments:
            texts[side][segment["speaker"]].append(segment["text"])
    inverse = {ref: hyp for hyp, ref in mapping.items()}
    pairs = [(ref, inverse.get(ref)) for ref in reference_speakers]
    pairs.extend((None, hyp) for hyp in hypothesis_speakers if hyp not in mapping)
    speakers = []
    for ref, hyp in pairs:
        ref_text = "".join(texts[0].get(ref, []))
        hyp_text = "".join(texts[1].get(hyp, []))
        a, b = normalize_text(ref_text, ignore_pattern), normalize_text(hyp_text, ignore_pattern)
        errors = edit_distance(a, b)
        speakers.append(
            {
                "reference_speaker": ref,
                "hypothesis_speaker": hyp,
                "reference_text": ref_text,
                "hypothesis_text": hyp_text,
                "reference_characters": len(a),
                "hypothesis_characters": len(b),
                "edit_distance": errors,
                "cer": ratio(errors, len(a)),
            }
        )
    text_score = {
        key: sum(s[key] for s in speakers)
        for key in ("reference_characters", "hypothesis_characters", "edit_distance")
    }
    text_score["cer"] = ratio(text_score["edit_distance"], text_score["reference_characters"])
    warnings = []
    if diarization["reference_overlap_seconds"]:
        warnings.append("参考包含重叠讲话；DER 包含重叠区域，文本按各说话人分别拼接。")
    if len(mapping) < max(len(reference_speakers), len(hypothesis_speakers)):
        warnings.append("存在未匹配说话人，其文字计入删除或插入错误；请检查人数与时间轴。")
    if not text_score["reference_characters"]:
        warnings.append("归一化后的参考文字为空，CER 为 null，仍保留错误数量。")
    if not diarization["reference_speaker_seconds"]:
        warnings.append("参考语音时长为零，DER 为 null，仍保留误检时长。")
    return {
        "reference_segments": len(reference),
        "hypothesis_segments": len(hypothesis),
        "reference_speakers": reference_speakers,
        "hypothesis_speakers": hypothesis_speakers,
        "speaker_mapping": mapping,
        "text_score": text_score,
        "diarization": diarization,
        "speakers": speakers,
        "warnings": warnings,
    }


def recording_name(path: Path) -> str:
    name = path.name
    name = name[:-10] if name.lower().endswith(".full.json") else name[:-5]
    if Path(name).suffix.lower() in AUDIO_SUFFIXES:
        name = name[: -len(Path(name).suffix)]
    return name.casefold()


def meeting_name(path: Path) -> str:
    name = recording_name(path)
    match = MEETING_NAME.fullmatch(name)
    return match[1].casefold() if match else name


def discover_json(root: Path) -> list[Path]:
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    return [
        p
        for p in paths
        if p.is_file()
        and p.suffix.lower() == ".json"
        and p.name.casefold() not in REPORT_NAMES
        and not (
            p.name.lower().endswith(".full.json") and p.with_name(p.name[:-10] + ".json").is_file()
        )
    ]


def match_files(reference: Path, hypothesis: Path):
    if not reference.exists() or not hypothesis.exists():
        raise ValueError("参考或识别路径不存在，请检查 --reference 和 --hypothesis")
    if reference.is_file() != hypothesis.is_file():
        raise ValueError("请同时传两个 JSON 文件，或同时传两个目录")
    if reference.is_file():
        return [{"reference_file": str(reference), "hypothesis_file": str(hypothesis)}]
    references, hypotheses = discover_json(reference), discover_json(hypothesis)
    if not references or not hypotheses:
        raise ValueError("参考或识别目录中没有可比较的 JSON 文件")
    exact, meetings = defaultdict(list), defaultdict(list)
    for ref in references:
        exact[recording_name(ref)].append(ref)
        meetings[meeting_name(ref)].append(ref)
    pairs, used = [], set()
    for hyp in hypotheses:
        candidates = exact.get(recording_name(hyp)) or meetings.get(meeting_name(hyp), [])
        item = {"hypothesis_file": str(hyp)}
        if not candidates:
            item.update(status="missing_reference", error="没有匹配的参考 JSON")
        elif len(candidates) > 1:
            item.update(
                status="ambiguous_reference",
                error="多个参考文件匹配，请缩小参考目录",
                candidates=[str(p) for p in candidates],
            )
        else:
            item["reference_file"] = str(candidates[0])
            used.add(candidates[0])
        pairs.append(item)
    pairs.extend(
        {"reference_file": str(ref), "status": "missing_hypothesis", "error": "没有匹配的识别结果"}
        for ref in references
        if ref not in used
    )
    return pairs


def summarize(files):
    scored = [item for item in files if item["status"] == "compared"]
    text = {
        key: sum(item["text_score"][key] for item in scored)
        for key in ("reference_characters", "hypothesis_characters", "edit_distance")
    }
    text["cer"] = ratio(text["edit_distance"], text["reference_characters"])
    timing = {
        key: sum(item["diarization"][key] for item in scored)
        for key in (
            "reference_speaker_seconds",
            "miss_seconds",
            "false_alarm_seconds",
            "confusion_seconds",
            "error_seconds",
            "reference_overlap_seconds",
        )
    }
    timing["der"] = ratio(timing["error_seconds"], timing["reference_speaker_seconds"])
    return {
        "status_counts": dict(Counter(item["status"] for item in files)),
        "compared_files": len(scored),
        "text_score": text,
        "diarization": timing,
    }


def percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def write_report(report, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    lines = [
        "识别结果与参考标注对比",
        f"状态统计：{summary['status_counts']}",
        f"按时间映射的分说话人 CER：{percent(summary['text_score']['cer'])}",
        f"片段时间线 DER：{percent(summary['diarization']['der'])}",
        "汇总仅包含成功配对的文件；错误率可以超过 100%。",
        "CER 忽略标点/空白，统一全半角和大小写；按映射说话人拼接后比较。",
        "DER 使用输入 segments、零边界容差、包含重叠；不是原始 Community-1 的独立评分。",
    ]
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(
        [
            "reference_file",
            "hypothesis_file",
            "status",
            "cer",
            "der",
            "reference_characters",
            "edit_distance",
            "miss_seconds",
            "false_alarm_seconds",
            "confusion_seconds",
            "error",
        ]
    )
    for item in report["files"]:
        text, timing = item.get("text_score", {}), item.get("diarization", {})
        writer.writerow(
            [
                item.get("reference_file", ""),
                item.get("hypothesis_file", ""),
                item["status"],
                text.get("cer"),
                timing.get("der"),
                text.get("reference_characters"),
                text.get("edit_distance"),
                timing.get("miss_seconds"),
                timing.get("false_alarm_seconds"),
                timing.get("confusion_seconds"),
                item.get("error", ""),
            ]
        )
        lines.extend(
            [
                "",
                f"文件：{item.get('hypothesis_file', item.get('reference_file'))}",
                f"状态：{item['status']}",
            ]
        )
        if item["status"] != "compared":
            lines.append(item["error"])
            continue
        lines.append(f"CER：{percent(text['cer'])}；DER：{percent(timing['der'])}")
        lines.extend(item["warnings"])
        for speaker in item["speakers"]:
            lines.extend(
                [
                    (
                        f"参考 {speaker['reference_speaker']} ← 识别 {speaker['hypothesis_speaker']}；"
                        f"CER {percent(speaker['cer'])}，编辑错误 {speaker['edit_distance']}"
                    ),
                    "参考：" + speaker["reference_text"],
                    "识别：" + speaker["hypothesis_text"],
                ]
            )
    (directory / "comparison-files.csv").write_text(csv_buffer.getvalue(), encoding="utf-8-sig")
    (directory / "comparison-report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="对比转写与 reference：自动匹配文件、映射说话人，允许不同分段"
    )
    parser.add_argument("--reference", type=Path, default=Path("outputs/reference"))
    parser.add_argument("--hypothesis", type=Path, required=True, help="识别 JSON 文件或目录")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    parser.add_argument(
        "--ignore-pattern", help="计算 CER 前移除匹配文字的正则；默认保留语气词和标记正文"
    )
    args = parser.parse_args(argv)
    try:
        ignore = re.compile(args.ignore_pattern) if args.ignore_pattern else None
        files = match_files(args.reference.resolve(), args.hypothesis.resolve())
        for item in files:
            if "status" not in item:
                try:
                    item.update(
                        compare_segments(
                            load_segments(Path(item["reference_file"])),
                            load_segments(Path(item["hypothesis_file"])),
                            ignore,
                        )
                    )
                    item["status"] = "compared"
                except (OSError, TypeError, ValueError) as exc:
                    item.update(status="error", error=str(exc))
            if item["status"] == "compared":
                print(
                    f"[完成] {Path(item['hypothesis_file']).name}："
                    f"CER {percent(item['text_score']['cer'])}，DER {percent(item['diarization']['der'])}",
                    flush=True,
                )
            else:
                print(
                    f"[{item['status']}] {item.get('hypothesis_file', item.get('reference_file'))}：{item['error']}",
                    flush=True,
                )
        report = {
            "reference": str(args.reference.resolve()),
            "hypothesis": str(args.hypothesis.resolve()),
            "settings": {
                "speaker_mapping": "maximum_time_overlap_one_to_one",
                "text_metric": "time_mapped_speaker_cer",
                "normalization": "NFKC, casefold, letters/numbers/marks only",
                "ignore_pattern": args.ignore_pattern,
                "der_collar_seconds": 0,
                "der_include_overlap": True,
                "der_source": "segments",
            },
            "summary": summarize(files),
            "files": files,
        }
        write_report(report, args.output_dir.resolve())
    except (OSError, ValueError, re.error) as exc:
        parser.exit(1, f"对比失败：{exc}\n")
    summary = report["summary"]
    print(
        f"成功对比 {summary['compared_files']} 个文件；"
        f"CER {percent(summary['text_score']['cer'])}；DER {percent(summary['diarization']['der'])}\n"
        f"报告：{args.output_dir.resolve()}"
    )
    return int(any(item["status"] != "compared" for item in files))


if __name__ == "__main__":
    sys.exit(main())
