# -*- coding: utf-8 -*-

"""
graph_agent.py

LangGraph workflow for PropGPT query intelligence + retrieval + generation.

Key fixes / upgrades vs your version:
- Zero hard-coded mapping logic in the graph: we only use mappings coming from set_mappings_for_type() / COLUMN_MAPPING.
- Prompts now receive Allowed Entities / Selected Keys / Selected Columns as JSON (handled in prompts.py you updated).
- Fixes a critical bug: output_tokens was computed using cleaned_text before it was defined.
- More robust relevance check (context length + doc count) while still lightweight.
- Safer LLM fallback import path and safer defaults.
"""

import logging
import operator
from pathlib import Path
from typing import TypedDict, List, Annotated, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage

from .core_utils import (
    load_and_clean_data,
    create_documents,
    get_embeddings,
    build_cache_key,
    build_vector_store,
    build_bm25_retriever,
    hybrid_retrieve,
    clean_response,
    set_mappings_for_type,
    get_category_keys,
    get_columns_for_keys,
    flatten_columns,
    count_tokens,
)

from .config import EXCEL_FILE, PICKLE_FILE
from .prompts import build_location_prompt, build_city_prompt, build_project_prompt
from .agents import planner_identify_mapping_keys, agent_pick_relevant_columns

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    query: str
    items: List[str]
    comparison_type: str
    llm: Any
    years: List[int]
    chat_history: List[Dict[str, str]]

    detected_requirements: List[str]
    candidate_keys: List[str]
    candidate_columns: List[str]
    selected_keys: List[str]
    selected_columns: List[str]

    context_docs: List[Any]
    context_text: str

    final_response: str
    iteration_count: int
    input_tokens: int
    output_tokens: int
    messages: Annotated[List[BaseMessage], operator.add]
    relevance_passed: bool


def detect_requirements(query: str) -> List[str]:
    q = (query or "").lower()
    requirements: List[str] = []

    patterns = {
        "supply": ["supply", "available", "inventory", "total units", "unsold", "supplied"],
        "demand": ["demand", "sold", "purchased", "transactions", "absorbed", "consumed", "sales", "consumption"],
        "price": ["price", "rate", "per sqft", "cost", "valuation", "agreement price", "avg price"],
        "demography": ["demographic", "demography", "buyer", "pincode", "pin code", "age range", "profile"],
        "comparison": ["compare", "comparison", "versus", "vs", "difference"],
    }

    for req, words in patterns.items():
        if any(w in q for w in words):
            requirements.append(req)

    if not requirements:
        requirements.append("all")

    return sorted(list(set(requirements)))


def load_mappings_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Node: Load Mappings ---")

    query = state.get("query", "") or ""
    comparison_type = state.get("comparison_type", "Location") or "Location"

    set_mappings_for_type(comparison_type)

    from .core_utils import COLUMN_MAPPING  # loaded/updated by set_mappings_for_type()
    candidate_keys = sorted(list(COLUMN_MAPPING.keys())) if COLUMN_MAPPING else []

    detected_requirements = detect_requirements(query)
    logger.info("Detected requirements: %s", detected_requirements)

    return {
        "candidate_keys": candidate_keys,
        "detected_requirements": detected_requirements,
        "iteration_count": 0,
        "selected_keys": [],
        "selected_columns": [],
        "context_docs": [],
        "context_text": "",
        "relevance_passed": False,
    }


def smart_filter_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Node: Smart Filter (Iteration %s) ---", state.get("iteration_count", 0))

    query = state.get("query", "") or ""
    detected_requirements = state.get("detected_requirements", ["all"]) or ["all"]
    candidate_keys = state.get("candidate_keys", []) or []

    llm = state.get("llm")
    if not llm:
        from .config import get_llm  # keep local; do not hardcode any model here
        llm = get_llm()

    # Get the mapping dict for this comparison type (needed by new agents.py)
    # Get the mapping dict for this comparison type (needed by new agents.py)
    from .core_utils import COLUMN_MAPPING, CATEGORY_MAPPING
    mapping_dict = COLUMN_MAPPING if COLUMN_MAPPING else {}
    category_mapping = CATEGORY_MAPPING if CATEGORY_MAPPING else {}

    candidate_keys = list(mapping_dict.keys())
    relevant_candidates = candidate_keys


    # CRITICAL FIX: Pass mapping_dict to enable proper column-aware scoring
    mapping_keys = planner_identify_mapping_keys(
        llm=llm,
        query=query,
        candidate_keys=relevant_candidates,
        mapping_dict=mapping_dict,
        category_mapping=category_mapping,
        max_keys=10,
        use_llm_rerank=True
    )
    mapping_keys = (mapping_keys or [])[:10]

    columns_by_key = get_columns_for_keys(mapping_keys)
    all_candidate_columns = flatten_columns(columns_by_key)

    # CRITICAL FIX: Pass mapping_dict to enable proper column selection
    selected_columns = agent_pick_relevant_columns(
        llm=llm,
        query=query,
        selected_keys=mapping_keys,
        candidate_columns=all_candidate_columns,
        mapping_dict=mapping_dict,
        max_cols=20,
        use_llm_rerank=True
    )
    selected_columns = selected_columns or []

    logger.info("Planner selected %s keys; Column agent picked %s columns (from %s candidates)",
                len(mapping_keys), len(selected_columns), len(all_candidate_columns))

    return {
        "selected_keys": mapping_keys,
        "selected_columns": selected_columns,
        "iteration_count": int(state.get("iteration_count", 0)) + 1,
    }


def retrieval_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Node: Retrieval ---")

    query = state.get("query", "") or ""
    items = state.get("items", []) or []
    comparison_type = state.get("comparison_type", "Location") or "Location"
    selected_keys = state.get("selected_keys", []) or []
    years = state.get("years", [2020, 2021, 2022, 2023, 2024]) or [2020, 2021, 2022, 2023, 2024]

    if not items or not selected_keys:
        logger.warning("No items or keys selected for retrieval")
        return {"context_text": "No data available.", "context_docs": []}

    try:
        from django.conf import settings
        base_dir = Path(getattr(settings, "DATA_DIR", ""))

        excel_path = base_dir / EXCEL_FILE
        pickle_path = base_dir / PICKLE_FILE

        df, defaults, id_col = load_and_clean_data(
            excel_path, pickle_path, comparison_type,
            items=items, years=years
        )

        if df is None or df.empty:
            return {"context_text": "No data found for selected items.", "context_docs": []}

        selected_columns_list = state.get("selected_columns", []) or []
        
        # Original: columns_by_key = get_columns_for_keys(selected_keys)
        # New Logic: Filter full mapping to ONLY include columns in 'selected_columns'
        columns_by_key = {}
        full_mapping = get_columns_for_keys(selected_keys)
        
        if not selected_columns_list:
             # Fallback: if no columns selected (shouldn't happen), use full mapping
             columns_by_key = full_mapping
        else:
            selected_set = set(selected_columns_list)
            for k, cols in full_mapping.items():
                filtered = [c for c in cols if c in selected_set]
                if filtered:
                    columns_by_key[k] = filtered
        
        if not columns_by_key:
             logger.warning("No columns left after filtering by selected_columns. Fallback to full.")
             columns_by_key = full_mapping


        documents = create_documents(
            df=df,
            item_ids=items,
            defaults=defaults,
            columns_by_key=columns_by_key,
            years=years,
            comparison_type=comparison_type,
            id_col=id_col,
        )

        if not documents:
            return {"context_text": "No documents could be created.", "context_docs": []}

        embeddings = get_embeddings()
        all_cols = flatten_columns(columns_by_key)
        cache_key = build_cache_key(items, selected_keys, all_cols)

        vector_store = build_vector_store(documents, embeddings, cache_key)
        bm25_retriever = build_bm25_retriever(documents)

        context_docs = hybrid_retrieve(query, selected_keys, vector_store, bm25_retriever)
        context_text = "\n\n".join((doc.page_content or "").strip() for doc in (context_docs or []) if doc)

        if not context_text.strip():
            context_text = "No relevant evidence retrieved for the selected keys."

        return {"context_docs": context_docs or [], "context_text": context_text}

    except Exception as e:
        logger.exception("Error in retrieval node: %s", e)
        return {"context_text": f"Error during data retrieval: {str(e)}", "context_docs": []}


def check_relevance_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Node: Check Relevance ---")

    context = (state.get("context_text", "") or "").strip()
    docs = state.get("context_docs", []) or []
    iteration = int(state.get("iteration_count", 0))

    passed = False
    if len(context) >= 250 and len(docs) >= 1:
        passed = True
    if iteration >= 3:
        passed = True

    logger.info("Relevance passed: %s (context_len=%s, docs=%s, iteration=%s)",
                passed, len(context), len(docs), iteration)

    return {"relevance_passed": passed}


def generate_response_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Node: Generate Response ---")

    llm = state.get("llm")
    if not llm:
        from .config import get_llm
        llm = get_llm()

    query = state.get("query", "") or ""
    items = state.get("items", []) or []
    selected_keys = state.get("selected_keys", []) or []
    selected_columns = state.get("selected_columns", []) or []
    context = state.get("context_text", "") or ""
    comparison_type = (state.get("comparison_type", "Location") or "Location").lower()
    chat_history = state.get("chat_history", []) or []
    detected_requirements = state.get("detected_requirements", []) or []

    if comparison_type == "location":
        prompt_func = build_location_prompt
    elif comparison_type == "city":
        prompt_func = build_city_prompt
    else:
        prompt_func = build_project_prompt

    formatted_prompt = prompt_func(
        question=query,
        items=items,
        mapping_keys=selected_keys,
        selected_columns=selected_columns,
        context=context,
        category_summary=", ".join(detected_requirements),
        chat_history=chat_history,
    )

    llm_response = llm.invoke(formatted_prompt)

    raw_text = getattr(llm_response, "content", None) or str(llm_response)
    cleaned_text = clean_response(raw_text)

    usage = getattr(llm_response, "usage_metadata", {}) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)

    if input_tokens <= 0:
        input_tokens = count_tokens(formatted_prompt)
    if output_tokens <= 0:
        output_tokens = count_tokens(cleaned_text)

    return {
        "final_response": cleaned_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def router(state: AgentState) -> str:
    if state.get("relevance_passed", False):
        return "generate"
    if int(state.get("iteration_count", 0)) >= 3:
        return "generate"
    return "refine"


def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("load_mappings", load_mappings_node)
    workflow.add_node("smart_filter", smart_filter_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("generate_response", generate_response_node)

    workflow.set_entry_point("load_mappings")

    workflow.add_edge("load_mappings", "smart_filter")
    workflow.add_edge("smart_filter", "retrieval")
    workflow.add_edge("retrieval", "check_relevance")

    workflow.add_conditional_edges(
        "check_relevance",
        router,
        {"refine": "smart_filter", "generate": "generate_response"},
    )

    workflow.add_edge("generate_response", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
