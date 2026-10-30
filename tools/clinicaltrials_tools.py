from langchain_core.tools import tool

from query_clinicaltrials import search_trials, search_trials_for_pair


@tool
def search_clinical_trials_tool(
    condition: str = "",
    intervention: str = "",
    term: str = "",
    max_results: int = 20,
) -> dict:
    """Search ClinicalTrials.gov by condition, intervention or free-text term."""
    return search_trials(
        condition=condition or None,
        intervention=intervention or None,
        term=term or None,
        max_results=max_results,
    )


@tool
def get_clinical_trials_for_pair_tool(
    target_symbol: str, disease_name: str, max_results: int = 20
) -> dict:
    """Return ClinicalTrials.gov trials for one disease-target pair."""
    return search_trials_for_pair(
        target_symbol, disease_name, max_results=max_results
    )