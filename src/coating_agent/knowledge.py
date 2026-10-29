from importlib.resources import files
import json
import re
from rank_bm25 import BM25Okapi

CARDS = json.loads(files("coating_agent").joinpath("cards.json").read_text(encoding="utf-8"))
for card in CARDS:
    card.update({"subsystem": "供浆", "version": "1.0", "reviewed": "2026-10-27", "limitation": "不可作为现场操作依据"})

def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())
    return words + ["".join(words[i:i+2]) for i in range(len(words)-1)]


# Cards are immutable for the lifetime of the local process; build the index once.
INDEX = BM25Okapi([tokens(c["title"] + c["text"]) for c in CARDS])


def search(query: str, k: int = 3) -> list[dict]:
    if not isinstance(query, str) or not query.strip() or not isinstance(k, int) or not 1 <= k <= 10:
        return []
    # Avoid presenting generic character overlap as evidence for unrelated topics.
    if not any(term in query for term in ("供浆", "涂布", "极片", "泵", "流量", "传感器", "阻力", "数据缺失", "数据 缺失", "参数优化")):
        return []
    scores = INDEX.get_scores(tokens(query))
    order = sorted(range(len(CARDS)), key=lambda i: scores[i], reverse=True)
    return [dict(CARDS[i]) for i in order[:k] if scores[i] > 0]
