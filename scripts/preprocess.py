#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
访谈文本预处理脚本（interview-insight skill）

功能：
  1. 自动识别并分离访谈逐字稿中的角色（访谈者/受访者、问/答、Q/A 等），支持显式指定标签
  2. 提取目标角色（默认受访者/用户）的全部发言
  3. 清洗语气词、停顿标记、重复标点与冗余空白（保留语义，不做改写）
  4. 统计高频词（优先 jieba，未安装时降级为 n-gram 统计）
  5. 单独统计"空白语段"（未回答、极短回应）——沉默也是数据，不得当作噪声丢弃
  6. 支持标准化工作目录结构（--workdir）
  7. 多份访谈自动加受访者前缀（[U1-S03]），保证"语段 → 受访者"可反查

用法：
  python3 preprocess.py --input 00_raw/访谈A.txt --workdir interview_analysis --segments
  python3 preprocess.py --input 00_raw/ --workdir interview_analysis --segments
  python3 preprocess.py --input 00_raw/U1.txt --role interviewer --outputdir ./out
  python3 preprocess.py --input 00_raw/*.txt --workdir out --participant-labels "参与者A,参与者B"
  python3 preprocess.py --input 00_raw/U1.txt --workdir out --keep-fillers   # 保留语气词，供逐字引用

输出：
  <outputdir>/cleaned_text.txt   清洗后的目标角色发言（带 [S编号]，便于溯源）
  <outputdir>/word_stats.json    高频词、语段统计、受访者映射与空白语段记录

说明：
  清洗会移除语气词与停顿标记，因此 cleaned_text 不再是逐字字面转写。
  报告中的"原话"引自本文件；如需严格逐字引用，请加 --keep-fillers 重跑，或回原始逐字稿取。

编号规则：
  单份访谈            → [S01] [S02] ...
  两份及以上访谈      → [U1-S01] [U2-S01] ...（受访者号 + 该受访者内的语段号）
  "U几" 与源文件的对应关系记录在 word_stats.json 的 respondent_map 中。
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

# 角色标记词表：行首出现这些词（紧跟中英文冒号）即判定为对应角色
SPEAKER_LABELS = {
    "interviewer": ["访谈者", "访谈员", "采访者", "主持人", "研究员", "问", "Q", "I", "Interviewer"],
    "user": ["受访者", "被访者", "用户", "参与者", "嘉宾", "答", "A", "R", "Interviewee", "Participant"],
}

# 清洗时移除的填充词/语气词（仅移除独立出现的，避免破坏句内语义）
FILLERS = [
    "嗯嗯嗯", "嗯嗯", "嗯哼", "呃呃", "啊啊啊",
    "那个那个", "就是就是", "然后然后", "这个这个",
]
# 句首/独立语气词（按词边界处理）
STANDALONE_FILLERS = ["嗯", "呃", "额", "唉", "哦", "噢", "喔", "哎", "呀", "啊"]

# 高频中文停用词（词频统计时过滤）
STOPWORDS = set(
    "的 了 和 是 就 都 而 及 与 这 那 你 我 他 她 它 们 着 过 个 在 有 "
    "也 还 很 太 吧 吗 呢 啊 嗯 哦 哈 嘛 呀 啦 呗 喽 啥 咋 么 之 其 或 "
    "一个 一些 什么 怎么 为什么 可以 可能 应该 觉得 感觉 自己 我们 你们 "
    "他们 她们 就是 然后 这个 那个 这样 那样 这里 那里 因为 所以 但是 "
    "不过 而且 并且 以及 还是 或者 如果 虽然 比如 其实 现在 时候 东西".split()
)

# 标准工作目录结构
WORKDIR_TREE = [
    "00_raw",
    "01_segments",
    "02_codes",
    "03_axial",
    "04_themes",
    "05_report",
]

# 极短回应判定：清洗后长度不超过该值，视为可能的空白/敷衍语段
SHORT_SEGMENT_MAXLEN = 4

# 标点类（用于合并连续标点）
PUNCT_CLASS = "，。！？、,.!?"


def build_patterns(extra_labels, role_key):
    """在默认标签基础上合并用户显式指定的标签，返回行首角色标记正则。

    注意：必须建构为「单一括号包裹的交替组」，且自定义标签在前、默认标签在后，
    按长度降序排列以避免短标签吞掉长标签。
    """
    base = SPEAKER_LABELS[role_key]
    extra = [x.strip() for x in (extra_labels or "").replace("，", ",").split(",") if x.strip()]
    # 自定义标签优先，同时去重（保持自定义顺序在前）
    merged = extra + [x for x in base if x not in extra]
    # 长标签优先，防止 "参与者" 抢先匹配 "参与者A"
    merged = sorted(dict.fromkeys(merged), key=len, reverse=True)
    alt = "|".join(re.escape(x) for x in merged)
    return r"^(?:" + alt + r")\s*[:：]"


def parse_speakers(raw_text, interviewer_re, user_re):
    """按行首角色标记拆分对话，返回 ([(role, content), ...], 是否匹配到任何标记)。"""
    segments = []
    current_role = None
    current_lines = []
    matched_any = False

    def flush():
        if current_role is not None and current_lines:
            segments.append((current_role, "\n".join(current_lines).strip()))

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue
        if interviewer_re.match(stripped):
            matched_any = True
            flush()
            current_role = "interviewer"
            current_lines = [interviewer_re.sub("", stripped).strip()]
        elif user_re.match(stripped):
            matched_any = True
            flush()
            current_role = "user"
            current_lines = [user_re.sub("", stripped).strip()]
        else:
            # 无角色标记的续行，归入当前角色
            if current_role is None:
                current_role = "user"
            current_lines.append(stripped)
    flush()

    if not matched_any:
        # 全文没有角色标记：整体视为用户发言
        return [("user", raw_text.strip())], False
    return segments, True


def extract_role(segments, role):
    """提取目标角色的发言段落列表。"""
    target = (role or "user").lower()
    wanted = "interviewer" if target in ("interviewer", "q", "i") else "user"
    return [content for r, content in segments if r == wanted and content.strip()]


def clean_text(text, strip_fillers=True):
    """清洗单段发言：去填充词与停顿标记、规范标点与空白，不改写语义。

    strip_fillers=False 时保留语气词（用于需要逐字引用的场景），
    但标点与空白的规范化仍然执行——它不改变用词。
    """
    if strip_fillers:
        for f in FILLERS:
            text = text.replace(f, "")
        for f in STANDALONE_FILLERS:
            # 后接标点、省略号、空白或行尾的独立语气词
            text = re.sub(
                r"(?<![\u4e00-\u9fa5a-zA-Z])" + re.escape(f) + r"(?=[" + PUNCT_CLASS + r"~…\s]|$)",
                "", text,
            )
    # 省略号规范化：3 个以上合并为标准的「……」，不并入下面的标点合并（否则会被压成单个 …）
    text = re.sub(r"…{3,}", "……", text)
    # 连续标点合并为一个（取最后一个）
    text = re.sub(r"([" + PUNCT_CLASS + r"]){2,}", lambda m: m.group(1), text)
    text = re.sub(r"~{2,}", "~", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # 行首残留的标点、省略号与空白
    text = re.sub(r"^[…" + PUNCT_CLASS + r"\s]+", "", text, flags=re.MULTILINE)
    return text.strip()


def has_content(text):
    """判断清洗后的语段是否含实义内容（至少一个中文/字母/数字）。

    纯标点残留（如 '。'）与纯空白视为无实义，应从正文剔除；
    短但有实义的回应（如 '还行'）保留，同时记录为极短回应供参考。
    """
    return bool(re.search(r"[\u4e00-\u9fa5a-zA-Z0-9]", text))


def tokenize(text):
    """优先 jieba 分词；不可用时降级为 2~3 字 n-gram。"""
    try:
        import jieba  # type: ignore
        words = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]
        return [w for w in words if w not in STOPWORDS and re.search(r"[\u4e00-\u9fa5a-zA-Z]", w)], "jieba"
    except ImportError:
        chunks = re.split(r"[，。！？、；：,.!?;:\s]", text)
        grams = Counter()
        for chunk in chunks:
            chunk = re.sub(r"[^\u4e00-\u9fa5]", "", chunk)
            for n in (3, 2):
                for i in range(len(chunk) - n + 1):
                    gram = chunk[i: i + n]
                    if gram not in STOPWORDS:
                        grams[gram] += 1
        return grams, "ngram-fallback"


def word_frequency(text, topn):
    tokens, method = tokenize(text)
    freq = Counter(tokens) if method == "jieba" else tokens
    return freq.most_common(topn), method


def ensure_workdir(workdir):
    """创建标准工作目录结构。"""
    for sub in WORKDIR_TREE:
        os.makedirs(os.path.join(workdir, sub), exist_ok=True)
    return workdir


def collect_inputs(inputs):
    """展开输入：支持传入多个路径与 shell 已展开的文件列表。"""
    files = []
    for item in inputs:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                p = os.path.join(item, name)
                if os.path.isfile(p) and name.lower().endswith((".txt", ".md")):
                    files.append(p)
        else:
            files.append(item)
    return files


def main():
    ap = argparse.ArgumentParser(description="访谈逐字稿预处理：角色分离、清洗、词频统计、空白语段统计")
    ap.add_argument("--input", "-i", required=True, nargs="+",
                    help="访谈逐字稿路径（.txt/.md，UTF-8），可传多个或一个目录")
    ap.add_argument("--workdir", "-w", default=None,
                    help="标准工作目录（自动创建 00_raw~05_report 结构）")
    ap.add_argument("--outputdir", "-o", default=None,
                    help="输出目录；默认 <workdir>/01_segments，未指定 workdir 时为当前目录")
    ap.add_argument("--role", "-r", default="user", help="要分析的角色：user（默认）或 interviewer")
    ap.add_argument("--topn", "-n", type=int, default=30, help="高频词输出数量（默认30）")
    ap.add_argument("--segments", action="store_true", help="清洗结果按段落分块并编号")
    ap.add_argument("--interviewer-labels", default=None,
                    help="追加访谈者标签（逗号分隔），与内置标签合并。比让脚本猜测可靠")
    ap.add_argument("--participant-labels", default=None,
                    help="追加受访者标签（逗号分隔），与内置标签合并")
    ap.add_argument("--keep-fillers", action="store_true",
                    help="保留语气词与停顿标记（默认移除）。需要逐字引用原话时使用；"
                         "注意保留后 cleaned_text 更接近逐字稿，但可读性下降")
    args = ap.parse_args()

    # 目录解析
    if args.workdir:
        ensure_workdir(args.workdir)
    outdir = args.outputdir or (os.path.join(args.workdir, "01_segments") if args.workdir else ".")
    os.makedirs(outdir, exist_ok=True)

    files = collect_inputs(args.input)
    if not files:
        sys.exit("未找到可处理的输入文件（支持 .txt/.md）")

    interviewer_re = re.compile(build_patterns(args.interviewer_labels, "interviewer"), re.IGNORECASE)
    user_re = re.compile(build_patterns(args.participant_labels, "user"), re.IGNORECASE)
    strip_fillers = not args.keep_fillers

    # 按输入顺序为每个文件分配受访者编号（跳过的文件同样占号，避免错位）
    respondent_map = []
    indexed_segments = []          # [(respondent_id, content), ...]
    empty_segments = []
    no_marker_files = []

    for idx, path in enumerate(files):
        rid = f"U{idx + 1}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"[WARN] 跳过 {path}：{e}", file=sys.stderr)
            respondent_map.append({"id": rid, "file": path, "status": "skipped", "reason": str(e)})
            continue

        segments, matched = parse_speakers(raw, interviewer_re, user_re)
        if not matched:
            no_marker_files.append(path)

        role_texts = extract_role(segments, args.role)
        if not role_texts:
            print(f"[WARN] {path} 中未识别到角色「{args.role}」的发言", file=sys.stderr)
            respondent_map.append({"id": rid, "file": path, "status": "skipped",
                                   "reason": f"未识别到角色「{args.role}」的发言"})
            continue

        cleaned = [clean_text(t, strip_fillers=strip_fillers) for t in role_texts]
        # 空白/极短语段单独记录：无实义内容的剔除出正文，短但有实义的保留
        kept = []
        for t in cleaned:
            if not has_content(t):
                empty_segments.append({"respondent": rid, "file": path, "content": t,
                                       "reason": "清洗后无实义内容（纯标点或空白）"})
                continue
            if len(t) <= SHORT_SEGMENT_MAXLEN:
                empty_segments.append({"respondent": rid, "file": path, "content": t,
                                       "reason": "极短回应"})
            kept.append(t)

        for t in kept:
            indexed_segments.append((rid, t))
        respondent_map.append({"id": rid, "file": path, "status": "ok", "segment_count": len(kept)})

    if not indexed_segments:
        sys.exit("没有可输出的有效语段，请检查逐字稿角色标记或改用 --role interviewer")

    ok_respondents = [r for r in respondent_map if r["status"] == "ok"]
    multi = len(ok_respondents) > 1

    # 编号：单份 [S01]；多份 [U1-S01]（受访者号 + 该受访者内序号）
    counters = {}
    lines = []
    for rid, content in indexed_segments:
        counters[rid] = counters.get(rid, 0) + 1
        n = counters[rid]
        label = f"{rid}-S{n:02d}" if multi else f"S{n:02d}"
        lines.append((rid, label, content))

    # 为每份访谈记录语段编号区间，便于「语段 → 受访者」反查
    ranges = {}
    for rid, label, _ in lines:
        ranges.setdefault(rid, []).append(label)
    for entry in respondent_map:
        rng = ranges.get(entry["id"])
        if rng:
            entry["segment_labels"] = f"{rng[0]}–{rng[-1]}" if len(rng) > 1 else rng[0]

    all_cleaned = [c for _, _, c in lines]
    if args.segments:
        body = "\n\n".join(f"[{label}] {content}" for _, label, content in lines)
    else:
        body = "\n\n".join(all_cleaned)

    top_words, method = word_frequency("\n".join(all_cleaned), args.topn)

    stats = {
        "source_files": files,
        "analyzed_role": args.role,
        "file_count": len(files),
        "respondent_count": len(ok_respondents),
        "segment_count": len(all_cleaned),
        "total_chars": sum(len(t) for t in all_cleaned),
        "tokenize_method": method,
        "fillers_stripped": strip_fillers,
        "index_scheme": "respondent-prefixed" if multi else "flat",
        "respondent_map": respondent_map,
        "files_without_role_marker": no_marker_files,
        "empty_or_short_segments": empty_segments,
        "top_words": [{"word": w, "count": c} for w, c in top_words],
        "note": (
            "原话引自本清洗文本，非逐字转写；如需逐字引用请加 --keep-fillers 重跑，或回 00_raw/ 取原文。"
            if strip_fillers else
            "本次已保留语气词（--keep-fillers），文本更接近逐字转写；仍建议核对 00_raw/ 以确认无转写误差。"
        ),
    }

    cleaned_path = os.path.join(outdir, "cleaned_text.txt")
    stats_path = os.path.join(outdir, "word_stats.json")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(body + "\n")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] 预处理完成（分词方式：{method}）")
    print(f"   文件数：{len(files)}｜受访者：{len(ok_respondents)}｜语段数：{len(all_cleaned)}｜总字数：{stats['total_chars']}")
    print(f"   编号方式：{'[U1-S01] 形式（多份访谈，可反查受访者）' if multi else '[S01] 形式（单份访谈）'}")
    print(f"   清洗文本：{cleaned_path}")
    print(f"   统计文件：{stats_path}")
    print("   Top10 高频词：" + " / ".join(f"{w}({c})" for w, c in top_words[:10]))
    if method == "ngram-fallback":
        print("   [提示] 未检测到 jieba，已降级为 n-gram 统计，词频噪音较大（会出现跨词碎片）。")
        print("          建议安装：pip install jieba，以获得可用的分词结果。")
    if no_marker_files:
        print(f"   [提示] {len(no_marker_files)} 个文件未识别到角色标记，已整体视为用户发言：")
        for p in no_marker_files:
            print(f"          - {p}")
        print("          注意：未识别到标记时，整个文件作为单一语段处理，编号粒度较粗。")
        print("          建议用 --interviewer-labels / --participant-labels 追加标签，可获得逐句编号。")
    if empty_segments:
        print(f"   [提示] 记录到 {len(empty_segments)} 条空白/极短语段（已写入 word_stats.json）。")
        print("          沉默也是数据，不要当作噪声丢弃。")
    if not strip_fillers:
        print("   [提示] 已保留语气词（--keep-fillers）：文本更接近逐字，但引用时仍须核对 00_raw/。")


if __name__ == "__main__":
    main()
