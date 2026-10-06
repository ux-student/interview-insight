#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
访谈文本预处理脚本（interview-insight skill）
功能：
  1. 自动识别并分离访谈逐字稿中的角色（访谈者/受访者、问/答、Q/A 等）
  2. 提取目标角色（默认受访者/用户）的全部发言
  3. 清洗语气词、重复标点与冗余空白（保留语义，不做改写）
  4. 统计高频词（优先 jieba，未安装时降级为 n-gram 统计），输出候选关键词
用法：
  python3 preprocess.py --input transcript.txt --outputdir ./out
  python3 preprocess.py --input transcript.txt --role user --segments
输出：
  <outputdir>/cleaned_text.txt   清洗后的目标角色发言（带段落编号，便于溯源）
  <outputdir>/word_stats.json    高频词/候选关键词统计
"""
import argparse
import json
import re
import sys
from collections import Counter

# 角色标记：key=角色类别，value=行首标记正则（冒号支持中英文）
SPEAKER_PATTERNS = {
    "interviewer": r"^(?:访谈者|访谈员|采访者|主持人|研究员|问|Q|I|Interviewer)\s*[:：]",
    "user": r"^(?:受访者|被访者|用户|参与者|嘉宾|答|A|R|Interviewee|Participant)\s*[:：]",
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


def parse_speakers(raw_text):
    """按行首角色标记拆分对话，返回 [(role, content), ...]；无标记时整体归为 user。"""
    interviewer_re = re.compile(SPEAKER_PATTERNS["interviewer"], re.IGNORECASE)
    user_re = re.compile(SPEAKER_PATTERNS["user"], re.IGNORECASE)

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
        return [("user", raw_text.strip())]
    return segments


def extract_role(segments, role):
    """提取目标角色的发言段落列表。"""
    target = role.lower()
    if target in ("user", "interviewee", "a", "r", "respondent"):
        wanted = "user"
    elif target in ("interviewer", "q", "i"):
        wanted = "interviewer"
    else:
        wanted = "user"
    return [content for r, content in segments if r == wanted and content.strip()]


def clean_text(text):
    """清洗单段发言：去填充词、规范标点与空白，不改写语义。"""
    # 连续重复的填充短语
    for f in FILLERS:
        text = text.replace(f, "")
    # 去掉独立成句或句首的语气词（后接标点/停顿）
    for f in STANDALONE_FILLERS:
        text = re.sub(r"(?<![\u4e00-\u9fa5a-zA-Z])" + re.escape(f) + r"(?=[，。！？、,.!?~ ]|$)", "", text)
    # 重复标点归一
    text = re.sub(r"([，。！？、,.!?]){2,}", lambda m: m.group(1), text)
    text = re.sub(r"~{2,}", "~", text)
    # 省略号保留
    # 多余空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # 清理标点后遗留的句首停顿符
    text = re.sub(r"^[，、,\s]+", "", text, flags=re.MULTILINE)
    return text.strip()


def tokenize(text):
    """优先 jieba 分词；不可用时降级为 2~3 字 n-gram。"""
    try:
        import jieba  # type: ignore
        words = [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]
        return [w for w in words if w not in STOPWORDS and re.search(r"[\u4e00-\u9fa5a-zA-Z]", w)], "jieba"
    except ImportError:
        # 降级：按标点/停顿切句后提取 2-3 字中文片段做 n-gram
        chunks = re.split(r"[，。！？、；：,.!?;:\s]", text)
        grams = Counter()
        for chunk in chunks:
            chunk = re.sub(r"[^\u4e00-\u9fa5]", "", chunk)
            for n in (3, 2):
                for i in range(len(chunk) - n + 1):
                    gram = chunk[i : i + n]
                    if gram not in STOPWORDS:
                        grams[gram] += 1
        return grams, "ngram-fallback"


def word_frequency(text, topn):
    tokens, method = tokenize(text)
    if method == "jieba":
        freq = Counter(tokens)
    else:
        freq = tokens  # 已是 Counter
    return freq.most_common(topn), method


def main():
    ap = argparse.ArgumentParser(description="访谈逐字稿预处理：角色分离、清洗、词频统计")
    ap.add_argument("--input", "-i", required=True, help="访谈逐字稿文件路径（.txt/.md，UTF-8）")
    ap.add_argument("--outputdir", "-o", default=".", help="输出目录（默认当前目录）")
    ap.add_argument("--role", "-r", default="user", help="要分析的角色：user（默认）或 interviewer")
    ap.add_argument("--topn", "-n", type=int, default=30, help="高频词输出数量（默认30）")
    ap.add_argument("--segments", action="store_true", help="清洗结果按原始段落分块并编号")
    args = ap.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = f.read()
    except (OSError, UnicodeDecodeError) as e:
        sys.exit(f"读取输入文件失败：{e}")

    segments = parse_speakers(raw)
    role_texts = extract_role(segments, args.role)
    if not role_texts:
        sys.exit(f"未识别到角色「{args.role}」的发言，请检查逐字稿角色标记或改用 --role interviewer")

    cleaned = [clean_text(t) for t in role_texts]
    cleaned = [t for t in cleaned if t]

    if args.segments:
        body = "\n\n".join(f"[S{i+1:02d}] {t}" for i, t in enumerate(cleaned))
    else:
        body = "\n\n".join(cleaned)

    top_words, method = word_frequency("\n".join(cleaned), args.topn)
    stats = {
        "source_file": args.input,
        "analyzed_role": args.role,
        "segment_count": len(cleaned),
        "total_chars": sum(len(t) for t in cleaned),
        "tokenize_method": method,
        "top_words": [{"word": w, "count": c} for w, c in top_words],
    }

    import os
    os.makedirs(args.outputdir, exist_ok=True)
    cleaned_path = os.path.join(args.outputdir, "cleaned_text.txt")
    stats_path = os.path.join(args.outputdir, "word_stats.json")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(body + "\n")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] 预处理完成（分词方式：{method}）")
    print(f"   段落数：{len(cleaned)}｜总字数：{stats['total_chars']}")
    print(f"   清洗文本：{cleaned_path}")
    print(f"   词频统计：{stats_path}")
    print("   Top10 高频词：" + " / ".join(f"{w}({c})" for w, c in top_words[:10]))


if __name__ == "__main__":
    main()
