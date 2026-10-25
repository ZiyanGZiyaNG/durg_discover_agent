"""Open Targets GraphQL queries with timeouts and predictable errors."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_TIMEOUT_SECONDS = 30

SEARCH_QUERY = """
query Search($queryString: String!, $entityNames: [String!]) {
  search(queryString: $queryString, entityNames: $entityNames) {
    hits { id name }
  }
}
"""

DISEASE_IDENTITY_QUERY = """
query DiseaseIdentity($efoId: String!) {
  disease(efoId: $efoId) { id name }
}
"""

TARGET_IDENTITY_QUERY = """
query TargetIdentity($ensemblId: String!) {
  target(ensemblId: $ensemblId) { id approvedSymbol approvedName }
}
"""

DISEASE_TARGET_ASSOCIATION_QUERY = """
query DiseaseAssociatedTargets($efoId: String!, $pageSize: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: 0, size: $pageSize}) {
      count
      rows {
        target { id approvedSymbol approvedName }
        score
        datasourceScores { id score }
      }
    }
  }
}
"""

DISEASE_TARGET_PAIR_QUERY = """
query DiseaseTargetPair($efoId: String!, $ensemblId: String!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(Bs: [$ensemblId], page: {index: 0, size: 1}) {
      count
      rows {
        target { id approvedSymbol approvedName }
        score
        datasourceScores { id score }
      }
    }
  }
}
"""

KNOWN_DRUGS_QUERY = """
query KnownDrugs($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    drugAndClinicalCandidates {
      count
      rows {
        drug {
          id name drugType maximumClinicalStage
          mechanismsOfAction { rows { actionType mechanismOfAction } }
        }
        maxClinicalStage
        diseases { disease { id name } }
      }
    }
  }
}
"""

TRACTABILITY_QUERY = """
query Tractability($ensemblId: String!) {
    target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        tractability { label modality value}
    }
}
"""

TYPE_INTROSPECTION_QUERY = """
query TypeFields($typeName: String!) {
  __type(name: $typeName) {
    kind
    name
    fields {
      name
      args { name }
      type {
        kind name
        ofType { kind name ofType { kind name ofType { kind name } } }
      }
    }
  }
}
"""


class OpenTargetsError(RuntimeError):
    """Raised when Open Targets cannot return a usable response."""


def _session() -> requests.Session:
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    try:
        response = _session().post(
            ENDPOINT,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get("errors"):
            message = "; ".join(
                error.get("message", "GraphQL error")
                for error in payload["errors"]
            )
            raise OpenTargetsError(
                f"Open Targets GraphQL error (HTTP {response.status_code}): {message}"
            )
        response.raise_for_status()
        if not isinstance(payload, dict):
            raise OpenTargetsError(
                "Open Targets returned a non-JSON response "
                f"(HTTP {response.status_code}): {response.text[:500]}"
            )
    except OpenTargetsError:
        raise
    except requests.HTTPError as exc:
        detail = exc.response.text[:1000] if exc.response is not None else ""
        raise OpenTargetsError(
            f"Open Targets request failed: {exc}; response: {detail}"
        ) from exc
    except (requests.RequestException, ValueError) as exc:
        raise OpenTargetsError(f"Open Targets request failed: {exc}") from exc
    return payload.get("data") or {}


def _named_type(type_ref: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Unwrap GraphQL NON_NULL/LIST wrappers and return the named kind/type."""
    current = type_ref or {}
    while current.get("ofType"):
        current = current["ofType"]
    return current.get("kind"), current.get("name")


@lru_cache(maxsize=64)
def _introspect_type(type_name: str) -> dict[str, Any]:
    return _graphql(TYPE_INTROSPECTION_QUERY, {"typeName": type_name}).get("__type") or {}


def _selection_for_type(
    type_name: str,
    *,
    depth: int = 2,
    visited: frozenset[str] = frozenset(),
) -> list[str]:
    """Build a schema-aware selection of all fields that require no arguments.

    Scalar/enum fields are retained at every level. Nested objects are expanded
    to a bounded depth so current API fields are collected without hard-coding a
    release-specific evidence schema or following recursive entity links forever.
    """
    if type_name in visited:
        return []
    type_info = _introspect_type(type_name)
    selections: list[str] = []
    for field in type_info.get("fields") or []:
        if field.get("args"):
            continue
        kind, child_name = _named_type(field.get("type"))
        name = field.get("name")
        if not name or not child_name:
            continue
        if kind in {"SCALAR", "ENUM"}:
            selections.append(name)
        elif kind == "OBJECT" and depth > 0:
            children = _selection_for_type(
                child_name,
                depth=depth - 1,
                visited=visited | {type_name},
            )
            if children:
                selections.append(f"{name} {{ {' '.join(children)} }}")
        # Interfaces and unions require type-specific fragments. They are listed
        # in evidence_fields as unavailable rather than guessed here.
    return selections


def _flatten_selection_names(selections: list[str]) -> list[str]:
    return [selection.split(" ", 1)[0] for selection in selections]


def get_datasource_evidence_records(
    efo_id: str,
    ensembl_id: str,
    datasource_id: str,
    *,
    max_records: int = 500,
) -> dict[str, Any]:
    """Return raw underlying evidence for one datasource and pair.

    The evidence field selection is generated from the live Open Targets
    GraphQL schema. This preserves release-specific fields, including new
    clinical report attributes, instead of asking an LLM to reconstruct them.
    """
    if max_records < 1:
        raise ValueError("max_records must be at least 1")
    nested_selections = _selection_for_type("Evidence", depth=2)
    scalar_selections = _selection_for_type("Evidence", depth=0)
    if not scalar_selections:
        raise OpenTargetsError("Open Targets Evidence schema has no queryable fields.")

    def query_rows(selections: list[str]) -> dict[str, Any]:
        query = f"""
    query DatasourceEvidence(
      $efoId: String!,
      $ensemblId: String!,
      $datasourceIds: [String!]!,
      $size: Int!
    ) {{
      disease(efoId: $efoId) {{
        evidences(
          ensemblIds: [$ensemblId],
          datasourceIds: $datasourceIds,
          size: $size
        ) {{
          count
          rows {{ {' '.join(selections)} }}
        }}
      }}
    }}
    """
        return _graphql(
            query,
            {
                "efoId": efo_id,
                "ensemblId": ensembl_id,
                "datasourceIds": [datasource_id],
                "size": max_records,
            },
        )

    warnings: list[str] = []
    selections = nested_selections or scalar_selections
    try:
        data = query_rows(selections)
    except OpenTargetsError as nested_error:
        if selections == scalar_selections:
            raise
        warnings.append(
            "The API rejected the expanded nested evidence selection; Agent 1 "
            "retried with every top-level scalar/enum field. Original error: "
            f"{nested_error}"
        )
        selections = scalar_selections
        data = query_rows(selections)
    evidences = ((data.get("disease") or {}).get("evidences") or {})
    records = evidences.get("rows") or []
    total = int(evidences.get("count") or 0)
    fetched_bytes = len(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    estimated = None
    if records:
        estimated = round(fetched_bytes / len(records) * total)
    return {
        "datasource_id": datasource_id,
        "total_evidence_count": total,
        "fetched_evidence_count": len(records),
        "truncated": len(records) < total,
        "estimated_complete_size_bytes": estimated,
        "evidence_fields": _flatten_selection_names(selections),
        "collection_warnings": warnings,
        "records": records,
    }


def _search(name: str, entity: str) -> dict[str, str] | None:
    data = _graphql(SEARCH_QUERY, {"queryString": name, "entityNames": [entity]})
    hits = (data.get("search") or {}).get("hits") or []
    return hits[0] if hits else None


class EntityResolutionError(OpenTargetsError):
    """Raised when a free-text entity query cannot be selected uniquely."""

    def __init__(self, query: str, entity: str, candidates: list[dict[str, str]]):
        self.query = query
        self.entity = entity
        self.candidates = candidates
        choices = ", ".join(f"{item['id']} ({item['name']})" for item in candidates)
        super().__init__(
            f"Ambiguous {entity} query {query!r}. Supply --{entity}-id. "
            f"Candidates: {choices or 'none'}"
        )


def _normalized_entity_name(value: str) -> str:
    value = value.casefold().replace("’", "'").replace("'s", "")
    return "".join(character for character in value if character.isalnum())


def search_entity_candidates(name: str, entity: str, limit: int = 5) -> list[dict[str, str]]:
    if entity not in {"disease", "target"}:
        raise ValueError("entity must be 'disease' or 'target'")
    data = _graphql(SEARCH_QUERY, {"queryString": name, "entityNames": [entity]})
    hits = (data.get("search") or {}).get("hits") or []
    return [{"id": hit["id"], "name": hit["name"]} for hit in hits[:limit]]


def resolve_entity(
    name: str, entity: str, entity_id: str | None = None
) -> dict[str, str]:
    """Resolve a disease or target before any LLM planning occurs."""
    if entity not in {"disease", "target"}:
        raise ValueError("entity must be 'disease' or 'target'")
    if entity_id:
        if entity == "disease":
            record = (_graphql(DISEASE_IDENTITY_QUERY, {"efoId": entity_id}).get("disease"))
            if not record:
                raise OpenTargetsError(f"Disease ID not found: {entity_id}")
            return {"id": record["id"], "name": record["name"], "symbol": ""}
        record = (_graphql(TARGET_IDENTITY_QUERY, {"ensemblId": entity_id}).get("target"))
        if not record:
            raise OpenTargetsError(f"Target ID not found: {entity_id}")
        return {
            "id": record["id"],
            "name": record.get("approvedName") or record.get("approvedSymbol") or "",
            "symbol": record.get("approvedSymbol") or "",
        }
    candidates = search_entity_candidates(name, entity)
    normalized_query = _normalized_entity_name(name)
    exact = [
        candidate
        for candidate in candidates
        if _normalized_entity_name(candidate["name"]) == normalized_query
    ]
    if len(exact) != 1:
        if not candidates:
            raise OpenTargetsError(f"{entity.title()} not found: {name}")
        raise EntityResolutionError(name, entity, candidates)
    return {**exact[0], "symbol": name if entity == "target" else ""}


def get_efo_id(disease_name: str) -> str | None:
    hit = _search(disease_name, "disease")
    return hit["id"] if hit else None


def get_ensembl_id(gene_symbol: str) -> str | None:
    hit = _search(gene_symbol, "target")
    return hit["id"] if hit else None


def get_disease_associated_targets(
    efo_id: str, page_size: int = 10
) -> list[dict[str, Any]]:
    data = _graphql(
        DISEASE_TARGET_ASSOCIATION_QUERY,
        {"efoId": efo_id, "pageSize": page_size},
    )
    disease = data.get("disease")
    if not disease:
        return []
    return (disease.get("associatedTargets") or {}).get("rows") or []


def discover_disease_targets(
    disease_name: str, top_n: int = 10, disease_id: str | None = None
) -> dict[str, Any]:
    """Resolve a disease and return its top targets by Open Targets score."""
    if not 1 <= top_n <= 100:
        raise ValueError("top_n must be between 1 and 100")
    disease_hit = (
        resolve_entity(disease_name, "disease", disease_id)
        if disease_id
        else _search(disease_name, "disease")
    )
    if not disease_hit:
        raise OpenTargetsError(f"Disease not found: {disease_name}")
    data = _graphql(
        DISEASE_TARGET_ASSOCIATION_QUERY,
        {"efoId": disease_hit["id"], "pageSize": top_n},
    )
    disease = data.get("disease")
    if not disease:
        raise OpenTargetsError(f"Disease record unavailable: {disease_hit['id']}")
    association = disease.get("associatedTargets") or {}
    rows = association.get("rows") or []
    rows = sorted(
        rows,
        key=lambda row: (
            -(row.get("score") or 0.0),
            (row.get("target") or {}).get("id") or "",
        ),
    )[:top_n]
    targets = []
    for rank, row in enumerate(rows, start=1):
        target = row.get("target") or {}
        targets.append(
            {
                "rank": rank,
                "target_id": target.get("id") or "",
                "target_symbol": target.get("approvedSymbol") or "",
                "target_name": target.get("approvedName") or "",
                "overall_association_score": row.get("score") or 0.0,
                "datasource_scores": [
                    {"datasource_id": item["id"], "score": item["score"]}
                    for item in (row.get("datasourceScores") or [])
                ],
            }
        )
    total = int(association.get("count") or len(targets))
    return {
        "disease_query": disease_name,
        "disease_id": disease["id"],
        "disease_name": disease["name"],
        "requested_target_count": top_n,
        "returned_target_count": len(targets),
        "total_associated_target_count": total,
        "targets": targets,
        "limitations": [
            f"Only the top {top_n} targets were selected for v1 discovery.",
            "No underlying evidence records were downloaded in discovery mode.",
            "The project has not yet defined its own target-selection score or threshold.",
        ],
    }


def get_disease_target_evidence_by_ids(
    disease_id: str,
    target_id: str,
    disease_query: str = "",
    target_query: str = "",
    page_size: int = 100,
    max_evidence_records: int = 500,
) -> dict[str, Any]:
    """Query one already-resolved exact pair; performs no name search."""

    disease_name = disease_query or disease_id
    target_name = target_query or target_id
    disease_hit = {"id": disease_id, "name": disease_name}
    target_hit = {"id": target_id, "name": target_name}

    data = _graphql(
        DISEASE_TARGET_PAIR_QUERY,
        {"efoId": disease_hit["id"], "ensemblId": target_hit["id"]},
    )
    disease = data.get("disease")
    if not disease:
        raise OpenTargetsError(f"Disease record unavailable: {disease_hit['id']}")
    rows = (disease.get("associatedTargets") or {}).get("rows") or []

    match = next(
        (row for row in rows if (row.get("target") or {}).get("id") == target_hit["id"]),
        None,
    )
    target = (match or {}).get("target") or {}
    limitations = [
        "The verified disease ID and target ID were queried as an exact pair.",
        "Datasource scores are Open Targets association scores, not clinical confidence scores.",
    ]
    if not match:
        limitations.append("Open Targets returned no association row for the exact pair.")

    datasource_scores = [
        {"datasource_id": item["id"], "score": item["score"]}
        for item in ((match or {}).get("datasourceScores") or [])
    ]
    datasource_evidence = []
    clinical_score = next(
        (
            item["score"]
            for item in datasource_scores
            if item["datasource_id"] == "clinical_precedence"
        ),
        None,
    )
    if clinical_score is not None:
        clinical = get_datasource_evidence_records(
            disease["id"],
            target_hit["id"],
            "clinical_precedence",
            max_records=max_evidence_records,
        )
        clinical["aggregated_score"] = clinical_score
        datasource_evidence.append(clinical)
        if clinical["truncated"]:
            limitations.append(
                "Clinical precedence evidence was truncated at "
                f"{max_evidence_records} records; the output includes an estimated "
                "complete JSON size."
            )

    return {
        "disease_query": disease_name,
        "target_query": target_name,
        "disease_id": disease["id"],
        "disease_name": disease["name"],
        "target_id": target_hit["id"],
        "target_symbol": target.get("approvedSymbol") or target_name.upper(),
        "target_name": target.get("approvedName") or target_hit["name"],
        "overall_association_score": match.get("score") if match else None,
        "datasource_scores": datasource_scores,
        "datasource_evidence": datasource_evidence,
        "rank_within_fetched_targets": None,
        "fetched_target_count": len(rows),
        "limitations": limitations,
    }


def get_disease_target_evidence(
    disease_name: str,
    target_name: str,
    page_size: int = 100,
    max_evidence_records: int = 500,
) -> dict[str, Any]:
    """Resolve IDs by name, then query that exact pair."""

    disease_hit = _search(disease_name, "disease")
    if not disease_hit:
        raise OpenTargetsError(f"Disease not found: {disease_name}")
    target_hit = _search(target_name, "target")
    if not target_hit:
        raise OpenTargetsError(f"Target not found: {target_name}")
    return get_disease_target_evidence_by_ids(
        disease_hit["id"],
        target_hit["id"],
        disease_query=disease_name,
        target_query=target_name,
        page_size=page_size,
        max_evidence_records=max_evidence_records,
    )


def get_target_drug_evidence_by_ids(ensembl_id: str, efo_id: str) -> dict[str, Any]:
    data = _graphql(KNOWN_DRUGS_QUERY, {"ensemblId": ensembl_id})
    candidates = (data.get("target") or {}).get("drugAndClinicalCandidates") or {}
    rows = candidates.get("rows") or []

    def format_row(row: dict[str, Any], matched: bool) -> dict[str, Any]:
        drug = row["drug"]
        return {
            "drug_id": drug["id"],
            "drug_name": drug["name"],
            "drug_type": drug.get("drugType"),
            "max_clinical_stage": row.get("maxClinicalStage"),
            "mechanisms_of_action": [
                item["mechanismOfAction"]
                for item in (drug.get("mechanismsOfAction") or {}).get("rows", [])
            ],
            "disease_indications": list(
                dict.fromkeys(
                    d["disease"]["name"]
                    for d in row.get("diseases", [])
                    if d.get("disease")
                )
            ),
            "exact_disease_match": matched,
        }

    drugs = []
    for row in rows:
        disease_ids = {
            d["disease"]["id"] for d in row.get("diseases", []) if d.get("disease")
        }
        drugs.append(format_row(row, efo_id in disease_ids))
    return {
        "ensembl_id": ensembl_id,
        "efo_id": efo_id,
        "total_drug_count": int(candidates.get("count") or len(rows)),
        "fetched_drug_count": len(rows),
        "drugs": drugs,
    }


def get_known_drugs_associated_with_target(
    ensembl_id: str, efo_id: str
) -> list[dict[str, Any]]:
    drugs = get_target_drug_evidence_by_ids(ensembl_id, efo_id)["drugs"]
    exact = [d for d in drugs if d["exact_disease_match"]]
    return exact or drugs


def get_target_tractability_by_id(ensembl_id: str) -> dict[str, Any]:
    data = _graphql(TRACTABILITY_QUERY, {"ensemblId": ensembl_id})
    target = data.get("target")
    if not target:
        raise OpenTargetsError(f"Target record unavailable: {ensembl_id}")
    assessments = [
        {"modality": item["modality"], "label": item["label"], "value": item["value"]}
        for item in (target.get("tractability") or [])
    ]
    return {
        "target_id": target["id"],
        "target_symbol": target.get("approvedSymbol") or "",
        "assessment_count": len(assessments),
        "assessments": assessments,
        "limitations": [
            "Tractability is an in silico druggability assessment, "
            "not evidence of efficacy or of an existing drug."
        ]
    }