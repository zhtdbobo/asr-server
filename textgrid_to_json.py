"""Convert AliMeeting/Praat text TextGrids to reference segments, without dependencies."""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from uuid import uuid4

# Praat reads free-standing values and ignores labels (including item [1]:).
# Keep quoted text together, including newlines and doubled quotes; ! comments
# must be ignored only outside strings. This handles both full and short formats.
# https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html
_TOKEN = re.compile(
    r'(?<!\S)(?:"(?:[^"]|"")*"|<exists>|<absent>|'
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?!\S)|![^\r\n]*"
)
_SPEAKER = re.compile(r"(?<![A-Za-z0-9])SPK\d+(?![A-Za-z0-9])", re.IGNORECASE)


class _Reader:
    def __init__(self, text: str):
        self.tokens = iter(
            match.group() for match in _TOKEN.finditer(text) if not match.group().startswith("!")
        )

    def take(self) -> str:
        try:
            return next(self.tokens)
        except StopIteration as exc:
            raise ValueError("TextGrid 内容不完整，请检查层数、片段数或引号") from exc

    def string(self) -> str:
        token = self.take()
        if not token.startswith('"'):
            raise ValueError(f"应为带双引号的字符串，实际为 {token!r}")
        return token[1:-1].replace('""', '"')

    def number(self) -> float:
        token = self.take()
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"应为数值，实际为 {token!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"时间戳或数量必须是有限数值: {token!r}")
        return value

    def count(self) -> int:
        value = self.number()
        if value < 0 or not value.is_integer():
            raise ValueError(f"层数或片段数必须是非负整数: {value}")
        return int(value)

    def bounds(self) -> tuple[float, float]:
        start, end = self.number(), self.number()
        if end < start:
            raise ValueError(f"结束时间小于开始时间: {start} -> {end}")
        return start, end

    def finish(self):
        if next(self.tokens, None) is not None:
            raise ValueError("TextGrid 中有多余数据，请检查声明的层数或片段数")


def parse_textgrid(text: str, source_stem: str = "") -> list[dict]:
    """Extract nonempty intervals; keep speaker IDs, original seconds and overlaps."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    reader = _Reader(text.lstrip("\ufeff"))
    if reader.string() not in {"ooTextFile", "ooTextFile short"}:
        raise ValueError("仅支持文本 TextGrid；二进制文件请先用 Praat 另存为文本格式")
    if reader.string() != "TextGrid":
        raise ValueError("文件对象类型不是 TextGrid")
    reader.bounds()
    flag = reader.take()
    if flag == "<absent>":
        reader.finish()
        return []
    if flag != "<exists>":
        raise ValueError(f"缺少 tiers? <exists>，实际为 {flag!r}")

    segments = []
    for _ in range(reader.count()):
        kind, tier_name = reader.string(), reader.string().strip()
        reader.bounds()
        count = reader.count()
        if kind == "TextTier":
            # Point annotations have no end time and are not speech segments.
            for _ in range(count):
                reader.number()
                reader.string()
            continue
        if kind != "IntervalTier":
            raise ValueError(f"不支持的标注层类型: {kind!r}")

        # AliMeeting uses e.g. R0003_M0046_F_SPK0093 or F_SPK0093 as
        # tier names. Near-field files also encode the speaker in their name.
        match = _SPEAKER.search(tier_name) or _SPEAKER.search(source_stem)
        speaker = match.group().upper() if match else tier_name
        if not speaker:
            raise ValueError("区间层没有说话人名称，且文件名中没有 SPK 编号")
        for index in range(count):
            start, end = reader.bounds()
            mark = reader.string().strip()
            if end <= start:
                raise ValueError(f"{tier_name!r} 第 {index + 1} 个区间时长必须大于 0")
            if mark:
                segments.append({"speaker": speaker, "start": start, "end": end, "text": mark})
    reader.finish()
    return sorted(
        segments, key=lambda segment: (segment["start"], segment["end"], segment["speaker"])
    )


def read_textgrid(path: Path, encoding: str | None = None) -> list[dict]:
    data = path.read_bytes()
    if data.startswith(b"ooBinaryFile"):
        raise ValueError("二进制 TextGrid 请先用 Praat 另存为文本格式")
    if encoding is None:
        encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError("无法解码文件；可用 --encoding gb18030 等显式指定源文件编码") from exc
    return parse_textgrid(text, source_stem=path.stem)


def convert_file(source: Path, target: Path, encoding: str | None = None) -> int:
    if source.resolve() == target.resolve():
        raise ValueError("JSON 输出路径不能与源 TextGrid 相同")
    segments = read_textgrid(source, encoding)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(segments, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return len(segments)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="将 TextGrid 转为 speaker/start/end/text JSON，支持单文件或递归处理目录"
    )
    parser.add_argument("input", type=Path, help="TextGrid 文件或数据集目录")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path, help="单文件模式的 JSON 输出路径")
    output.add_argument("--output-dir", type=Path, help="输出根目录，默认 outputs/reference")
    parser.add_argument("--encoding", help="手动指定源编码，默认自动识别 UTF-8/UTF-16 BOM")
    args = parser.parse_args(argv)
    source = args.input.resolve()
    if source.is_file():
        if source.suffix.lower() != ".textgrid":
            parser.error("输入文件扩展名必须是 .TextGrid")
        files = [source]
        root = source.parent
    elif source.is_dir():
        if args.output:
            parser.error("目录输入请使用 --output-dir，不能使用 --output")
        root = source
        files = sorted(
            (
                path
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() == ".textgrid"
            ),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
        if not files:
            parser.error("目录及其子目录中没有找到 .TextGrid 文件")
    else:
        parser.error(f"输入文件或目录不存在: {source}")
    if args.output and args.output.suffix.lower() != ".json":
        parser.error("--output 的扩展名必须是 .json")

    output_dir = args.output_dir or Path("outputs/reference")
    failed = 0
    for path in files:
        target = args.output or output_dir / path.relative_to(root).with_suffix(".json")
        try:
            count = convert_file(path, target, args.encoding)
        except (ValueError, OSError, LookupError) as exc:
            failed += 1
            print(f"失败 {path}: {exc}", file=sys.stderr)
            continue
        print(f"成功 {path.name}: {count} 段 -> {target.resolve()}")
    print(f"转换结束：共 {len(files)} 个，成功 {len(files) - failed} 个，失败 {failed} 个。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
