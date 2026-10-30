"""Europe PMC REST 查詢。"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from source_http import get_json

EPMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_PAGE_SIZE = 25
EPMC_MAX_RESULTS = 50
ABSTRACT_MAX_CHARS = 1200


def _parse_article(r: Dict[str, Any]) -> Dict[str, Any]:
    src, pid = r.get("source"), r.get("id")
    abstract = r.get("abstractText")
    if abstract and len(abstract) > ABSTRACT_MAX_CHARS:
        abstract = abstract[:ABSTRACT_MAX_CHARS] + "…"
    return {
        "source_db": src,
        "id": pid,
        "pmid": r.get("pmid"),
        "pmcid": r.get("pmcid"),
        "doi": r.get("doi"),
        "title": r.get("title"),
        "authors": r.get("authorString"),
        "journal": r.get("journalTitle"),
        "year": r.get("pubYear"),
        "pub_date": r.get("firstPublicationDate"),
        "cited_by": r.get("citedByCount"),
        "open_access": r.get("isOpenAccess"),
        "abstract": abstract,
        "url": f"https://europepmc.org/article/{src}/{pid}" if src and pid else None,
    }


def search_articles(
    query: str,
    max_results: int = EPMC_MAX_RESULTS,
    include_abstract: bool = False,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "query": query,
        "format": "json",
        "pageSize": min(EPMC_PAGE_SIZE, max_results),
        "resultType": "core" if include_abstract else "lite",
    }

    items: List[Dict[str, Any]] = []
    total: Optional[int] = None
    pages = 0
    cursor = "*"
    while len(items) < max_results:
        params["cursorMark"] = cursor
        data = get_json(EPMC_BASE_URL, params)
        pages += 1
        if total is None:
            total = data.get("hitCount")
        results = (data.get("resultList") or {}).get("result") or []
        items.extend(_parse_article(r) for r in results)
        next_cursor = data.get("nextCursorMark")
        if not results or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    items = items[:max_results]

    return {
        "source": "Europe PMC",
        "query": {k: v for k, v in params.items() if k != "cursorMark"},
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "total_count": total,
        "returned_count": len(items),
        "pages_fetched": pages,
        "truncated": bool(total is not None and total > len(items)),
        "items": items,
    }


def _as_list(value) -> list:
    if not value:
        return []
    return [value] if isinstance(value, str) else [v for v in value if v]


def _phrase_clause(terms) -> str:
    """每個詞都比對標題與摘要，詞與詞之間用 OR。"""
    return "(" + " OR ".join(
        f'TITLE:"{t}" OR ABSTRACT:"{t}"' for t in _as_list(terms)
    ) + ")"


def search_target_disease(target_symbol: str, disease_name, **kwargs) -> Dict[str, Any]:
    """基因與疾病都必須出現在標題或摘要，只收 PubMed 來源。disease_name 可為字串或同義詞清單。"""
    query = (
        f'(TITLE:"{target_symbol}" OR ABSTRACT:"{target_symbol}") '
        f"AND {_phrase_clause(disease_name)} "
        f"AND SRC:MED"
    )
    return search_articles(query, **kwargs)


NEGATIVE_TERMS = [
    "lack of efficacy",
    "no clinical benefit",
    "failed",
    "discontinued",
    "terminated",
    "toxicity",
    "adverse events",
    "dose-limiting",
]


def search_negative_literature(subject: str, disease=None, **kwargs) -> Dict[str, Any]:
    """標的或藥物出現在標題／摘要，且摘要含負面結果關鍵字。disease 可為字串或同義詞清單。"""
    negative = " OR ".join(f'ABSTRACT:"{t}"' for t in NEGATIVE_TERMS)
    query = (
        f'(TITLE:"{subject}" OR ABSTRACT:"{subject}") '
        f"AND ({negative}) AND SRC:MED"
    )
    if disease:
        query += f" AND {_phrase_clause(disease)}"
    return search_articles(query, **kwargs)