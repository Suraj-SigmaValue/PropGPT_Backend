import json
import logging
import re
import operator
from typing import TypedDict, List, Annotated, Dict, Any, Literal, Tuple
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# We import from .core_utils to reuse shared logic
from .core_utils import (
    load_and_clean_data,
    create_documents,
    get_embeddings,
    get_response_cache,
    build_cache_key,
    build_vector_store,
    build_bm25_retriever,
    hybrid_retrieve,
    clean_response,
    normalize_colname,
    set_mappings_for_type,
    get_category_keys,
    get_columns_for_keys,
    flatten_columns,
    count_tokens
)

from .config import EXCEL_FILE, PICKLE_FILE, SHEET_CONFIG
from .prompts import (
    build_location_prompt,
    build_city_prompt,
    build_project_prompt
)

from .agents import (
    planner_identify_mapping_keys,
    agent_pick_relevant_columns
)

# Configure logging
logger = logging.getLogger(__name__)

# --- State Definition ---
class AgentState(TypedDict):
    # Inputs
    query: str
    items: List[str]
    comparison_type: str
    llm: Any
    years: List[int]
    chat_history: List[Dict[str, str]]
    
    # Internal State
    detected_requirements: List[str]
    candidate_keys: List[str]
    candidate_columns: List[str]
    mapping_keys: List[str]  # The keys used for retrieval
    selected_keys: Annotated[List[str], operator.add]
    selected_columns: Annotated[List[str], operator.add]
    context_docs: List[Any]
    context_text: str
    
    # Outputs
    final_response: str
    iteration_count: int
    input_tokens: int
    output_tokens: int
    messages: Annotated[List[BaseMessage], operator.add]
    relevance_passed: bool

# --- Helper Functions ---

def detect_requirements(query: str) -> List[str]:
    """Detect categories/requirements from query."""
    query_lower = query.lower()
    requirements = []
    
    patterns = {
        'supply': ['supply', 'available', 'total units', 'unsold', 'supplied', 'fsi', 'floor space index'],
        'demand': ['demand', 'sold', 'purchased', 'transactions', 'absorbed', 'consumed', 'sales', 'consumption'],
        'price': ['price', 'rate', 'per sqft', 'cost', 'valuation', 'agreement price', 'avg price'],
        'demography': ['demographic', 'demography', 'buyer', 'pincode', 'age range', 'profile'],
        'comparison': ['compare', 'comparison', 'versus', 'vs', 'difference'],
    }
    
    for req, words in patterns.items():
        if any(word in query_lower for word in words):
            requirements.append(req)
            
    if not requirements:
        requirements.append('all')
        
    return list(set(requirements))

# --- Nodes ---

def load_mappings_node(state: AgentState):
    """Load mappings and detect requirements."""
    logger.info("--- Node: Load Mappings ---")
    
    query = state.get("query", "")
    comparison_type = state.get("comparison_type", "Location")
    
    # Set global mappings for utilities
    set_mappings_for_type(comparison_type)
    
    from .core_utils import COLUMN_MAPPING
    candidate_keys = sorted(list(COLUMN_MAPPING.keys())) if COLUMN_MAPPING else []
    
    detected_requirements = detect_requirements(query)
    logger.info(f"Detected requirements: {detected_requirements}")
    
    return {
        "candidate_keys": candidate_keys,
        "detected_requirements": detected_requirements,
        "iteration_count": 0,
        "selected_keys": [],
        "selected_columns": []
    }

def smart_filter_node(state: AgentState):
    """Filter keys based on detected requirements and query relevance."""
    logger.info(f"--- Node: Smart Filter (Iteration {state.get('iteration_count', 0)}) ---")
    
    query = state.get("query", "")
    detected_requirements = state.get("detected_requirements", ["all"])
    candidate_keys = state.get("candidate_keys", [])
    current_selected = state.get("selected_keys", [])
    
    # Use planner to identify mapping keys
    # Instead of just taking detected category keys, we let the planner select from candidate_keys
    # based on the query.
    
    llm = state.get("llm")
    if not llm:
        from core.config import get_llm
        llm = get_llm()

    # Get candidate keys relevant to the detected requirements first
    relevant_candidates = []
    for req in detected_requirements:
        cat_keys = get_category_keys(req)
        relevant_candidates.extend([k for k in cat_keys if k in candidate_keys])
    
    # Deduplicate relevant candidates
    relevant_candidates = list(dict.fromkeys(relevant_candidates))
    
    if not relevant_candidates:
        relevant_candidates = candidate_keys

    # 1. Select Mapping Keys using Planner Agent
    mapping_keys = planner_identify_mapping_keys(llm, query, relevant_candidates)
    
    # Safety limit (align with user's prompt update: Hard limit 7-10 keys)
    mapping_keys = mapping_keys[:10]
    
    # 2. Get all possible columns for these keys
    columns_by_key = get_columns_for_keys(mapping_keys)
    all_candidate_columns = flatten_columns(columns_by_key)
    
    # 3. Use Column Agent to pick strictly relevant columns
    selected_columns = agent_pick_relevant_columns(llm, query, mapping_keys, all_candidate_columns)
    
    logger.info(f"Planner selected {len(mapping_keys)} keys")
    logger.info(f"Column agent picked {len(selected_columns)} columns out of {len(all_candidate_columns)}")
    
    return {
        "selected_keys": mapping_keys,
        "selected_columns": selected_columns,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def retrieval_node(state: AgentState):
    """Retrieve data and build context."""
    logger.info("--- Node: Retrieval ---")
    
    query = state.get("query", "")
    items = state.get("items", [])
    comparison_type = state.get("comparison_type", "Location")
    selected_keys = state.get("selected_keys", [])
    years = state.get("years", [2020, 2021, 2022, 2023, 2024])
    
    if not items or not selected_keys:
        logger.warning("No items or keys selected for retrieval")
        return {"context_text": "No data available.", "context_docs": []}
    
    try:
        # Load and clean data
        from django.conf import settings
        base_dir = Path(settings.DATA_DIR)
        excel_path = base_dir / EXCEL_FILE
        pickle_path = base_dir / PICKLE_FILE
        
        df, defaults, id_col = load_and_clean_data(
            excel_path, pickle_path, comparison_type,
            items=items, years=years
        )
        
        if df is None or df.empty:
            return {"context_text": "No data found for selected items.", "context_docs": []}
            
        # Get columns for keys
        columns_by_key = get_columns_for_keys(selected_keys)
        
        # Create documents
        documents = create_documents(
            df, items, defaults, columns_by_key,
            years=years, comparison_type=comparison_type, id_col=id_col
        )
        
        if not documents:
            return {"context_text": "No documents could be created.", "context_docs": []}
            
        # Retrieval
        embeddings = get_embeddings()
        all_cols = flatten_columns(columns_by_key)
        cache_key = build_cache_key(items, selected_keys, all_cols)
        
        vector_store = build_vector_store(documents, embeddings, cache_key)
        bm25_retriever = build_bm25_retriever(documents)
        
        context_docs = hybrid_retrieve(query, selected_keys, vector_store, bm25_retriever)
        context_text = "\n\n".join(doc.page_content.strip() for doc in context_docs)
        
        return {
            "context_docs": context_docs,
            "context_text": context_text
        }
        
    except Exception as e:
        logger.exception(f"Error in retrieval node: {e}")
        return {"context_text": f"Error during data retrieval: {str(e)}", "context_docs": []}

def check_relevance_node(state: AgentState):
    """Check if the retrieved context is sufficient."""
    logger.info("--- Node: Check Relevance ---")
    
    query = state.get("query", "")
    context = state.get("context_text", "")
    iteration = state.get("iteration_count", 0)
    
    # Minimal check: do we have context and is it relevant?
    relevance_passed = len(context) > 100 or iteration >= 3
    
    logger.info(f"Relevance passed: {relevance_passed}")
    
    return {"relevance_passed": relevance_passed}

def generate_response_node(state: AgentState):
    """Generate the final LLM response."""
    logger.info("--- Node: Generate Response ---")
    
    llm = state["llm"]
    query = state["query"]
    items = state["items"]
    selected_keys = state["selected_keys"]
    selected_columns = state["selected_columns"]
    context = state["context_text"]
    comparison_type = state["comparison_type"]
    chat_history = state.get("chat_history", [])
    
    # Select appropriate prompt
    if comparison_type.lower() == "location":
        prompt_func = build_location_prompt
    elif comparison_type.lower() == "city":
        prompt_func = build_city_prompt
    else:
        prompt_func = build_project_prompt
        
    formatted_prompt = prompt_func(
        question=query,
        items=items,
        mapping_keys=selected_keys,
        selected_columns=selected_columns,
        context=context,
        category_summary=", ".join(state.get("detected_requirements", [])),
        chat_history=chat_history
    )
    
    llm_response = llm.invoke(formatted_prompt)
    
    # Extract token usage if available
    usage = getattr(llm_response, 'usage_metadata', {})
    input_tokens = usage.get('input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    
    # Fallback to manual counting if usage is 0
    if input_tokens == 0:
        input_tokens = count_tokens(formatted_prompt)
    if output_tokens == 0:
        output_tokens = count_tokens(cleaned_text)
    
    raw_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
    cleaned_text = clean_response(raw_text)
    
    return {
        "final_response": cleaned_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

# --- Router ---

def router(state: AgentState):
    """Determine next step."""
    if state.get("relevance_passed", False):
        return "generate"
    elif state.get("iteration_count", 0) >= 3:
        return "generate"
    else:
        return "refine"

# --- Graph Definition ---

def create_graph():
    """Create the iterative requirement-driven graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_mappings", load_mappings_node)
    workflow.add_node("smart_filter", smart_filter_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # Set entry point
    workflow.set_entry_point("load_mappings")
    
    # Define edges
    workflow.add_edge("load_mappings", "smart_filter")
    workflow.add_edge("smart_filter", "retrieval")
    workflow.add_edge("retrieval", "check_relevance")
    
    # Conditional edge
    workflow.add_conditional_edges(
        "check_relevance",
        router,
        {
            "refine": "smart_filter",
            "generate": "generate_response"
        }
    )
    
    workflow.add_edge("generate_response", END)
    
    # Memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app