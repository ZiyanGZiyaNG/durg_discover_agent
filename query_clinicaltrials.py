"""ClinicalTrials.gov API v2 查詢。"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from source_http import get_json

CT_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
CT_PAGE_SIZE = 50
CT_MAX_RESULTS = 100


def _parse_study(study: Dict[str, Any]) -> Dict[str, Any]:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    cond = p.get("conditionsModule", {})
    arms = p.get("armsInterventionsModule", {})
    sponsor = p.get("sponsorCollaboratorsModule", {})
    nct_id = ident.get("nctId")
    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle"),
        "status": status.get("overallStatus"),
        "start_date": (status.get("startDateStruct") or {}).get("date"),
        "completion_date": (status.get("completionDateStruct") or {}).get("date"),
        "why_stopped": status.get("whyStopped"),
        "phases": design.get("phases") or [],
        "study_type": design.get("studyType"),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "conditions": cond.get("conditions") or [],
        "interventions": [
            {
                "name": i.get("name"),
                "type": i.get("type"),
                "arm_group_labels": i.get("armGroupLabels") or [],
            }
            for i in (arms.get("interventions") or [])
        ],
        "arm_groups": [
            {"label": g.get("label"), "type": g.get("type")}
            for g in (arms.get("armGroups") or [])
        ],
        "sponsor": (sponsor.get("leadSponsor") or {}).get("name"),
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else None,
    }


def search_trials(
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    term: Optional[str] = None,
    statuses: Optional[List[str]] = None,
    max_results: int = CT_MAX_RESULTS,
) -> Dict[str, Any]:
    if not any([condition, intervention, term]):
        raise ValueError("condition / intervention / term 至少要給一個")

    params: Dict[str, Any] = {
        "format": "json",
        "countTotal": "true",
        "pageSize": min(CT_PAGE_SIZE, max_results),
    }
    if condition:
        params["query.cond"] = condition
    if intervention:
        params["query.intr"] = intervention
    if term:
        params["query.term"] = term
    if statuses:
        params["filter.overallStatus"] = ",".join(statuses)

    items: List[Dict[str, Any]] = []
    total: Optional[int] = None
    pages = 0
    while len(items) < max_results:
        data = get_json(CT_BASE_URL, params)
        pages += 1
        if total is None:
            total = data.get("totalCount")
        items.extend(_parse_study(s) for s in data.get("studies", []))
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    items = items[:max_results]

    return {
        "source": "ClinicalTrials.gov",
        "api_version": "v2",
        "query": {k: v for k, v in params.items() if k != "pageToken"},
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "total_count": total,
        "returned_count": len(items),
        "pages_fetched": pages,
        "truncated": bool(total is not None and total > len(items)),
        "items": items,
    }


def search_trials_for_pair(target_symbol: str, disease_name: str, **kwargs) -> Dict[str, Any]:
    """disease–target 配對：疾病放 condition，基因符號放 term。"""
    return search_trials(condition=disease_name, term=target_symbol, **kwargs)