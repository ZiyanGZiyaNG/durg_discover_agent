from langchain_core.tools import tool

from query_europepmc import search_articles, search_target_disease


@tool
def search_literature_tool(query: str, max_results: int = 20) -> dict:
    """Search Europe PMC literature with an Europe PMC query string."""
    return search_articles(query, max_results=max_results)


@tool
def get_literature_for_pair_tool(
    target_symbol: str, disease_name: str, max_results: int = 20
) -> dict:
    """Return PubMed literature mentioning one disease-target pair via Europe PMC."""
    return search_target_disease(
        target_symbol, disease_name, max_results=max_results
    )