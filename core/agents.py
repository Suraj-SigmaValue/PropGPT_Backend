"""
Query Intelligence Agents for PropGPT.

Contains two agents:
1. Planner Agent: Selects relevant mapping keys based on user query
2. Column Agent: Selects relevant columns based on query and mapping keys
"""

import json
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def planner_identify_mapping_keys(llm, query: str, candidate_keys: List[str]) -> List[str]:
    """
    Planner Agent: Identifies and selects the most relevant mapping keys for the user query.
    
    Args:
        llm: Language model instance (LangChain LLM)
        query: User's analysis query
        candidate_keys: List of available mapping keys to choose from
    
    Returns:
        List of selected mapping keys (at most 8, or fewer if fewer are relevant)
    """
    if not candidate_keys:
        return []

    # -------------------------
    # Local helper functions for query analysis
    # -------------------------
    def _classify_query_type(q: str) -> str:
        q = (q or "").lower()
        bhk_tokens = ["configuration", "config mix", "1bhk", "2bhk", "3bhk", "4bhk", "bhk"]
        if any(tok in q for tok in bhk_tokens):
            return "bhk_config"
        return "generic"

    def _classify_metric(q: str) -> str:
        q = (q or "").lower()
        supply_tokens = ["supply", "available", "total units", "unsold", "supplied", "fsi", "floor space index"]
        demand_tokens = ["demand", "sold", "purchased", "bought", "transactions", "sales count", "absorbed", "consumed", "consumption"]
        has_supply = any(tok in q for tok in supply_tokens)
        has_demand = any(tok in q for tok in demand_tokens)
        if has_supply and has_demand: return "both"
        if has_supply: return "supply"
        if has_demand: return "demand"
        return "unknown"

    def _is_demography_query(q: str) -> bool:
        q = (q or "").lower()
        demo_tokens = ["demography", "demographic", "buyer profile", "pincode", "pin code", "age profile"]
        return any(tok in q for tok in demo_tokens)

    def _detect_property_mentions(q: str) -> dict:
        q = (q or "").lower()
        return {
            "has_property_type": any(tok in q for tok in ["flat", "shop", "office", "property type", "property"]),
            "has_bhk_type": any(tok in q for tok in ["1bhk", "2bhk", "3bhk", "4bhk", "bhk", "bedroom"])
        }

    q_low = (query or "").lower()
    q_type = _classify_query_type(query)
    metric_type = _classify_metric(query)
    is_demo = _is_demography_query(query)
    property_mentions = _detect_property_mentions(query)

    sys_instr = (
        "You are a planning assistant that selects the most relevant mapping keys for answering a "
        "real-estate analytics question. Return ONLY a JSON list of mapping keys from CANDIDATE_KEYS."
    )
    prompt = f"""
    User Query: {query}

CANDIDATE_KEYS (Allowed Selection Only):
{json.dumps(candidate_keys, indent=2)}

You are a **Query-Intent Classification & Mapping-Key Selection Agent** operating under **strict analytical constraints**.

Your **ONLY responsibility** is to return the **minimum correct set of mapping keys** required to answer the user query.

Your output **must always be a JSON array of mapping keys only**.

---

## 🧠 CORE THINKING FRAMEWORK (INTERNAL – MUST FOLLOW)

### 🔹 LAYER 1 — PRIMARY MEASURE IDENTIFICATION
Classify the metric into ONE primary category based on the query focus:
1. **SUPPLY**: Queries about available inventory, launched items, or total supplied (e.g., supplied, total units, capacity, available
2. **DEMAND**: Queries about sold/consumed/absorbed items (e.g., units sold, area consumed, transactions).
3. **DEMOGRAPHY**: Queries about buyer profiles (e.g., pincode, age range).
4. **PRICE**: Queries about pricing, rates, or sales values (e.g., average price, price per sq ft).
5. **OTHER**: General or mixed (e.g., location details, totals without supply/demand split).

If the query compares categories (e.g., supply vs demand), select keys from both but minimize overlap.


### 🔹 LAYER 2 — DIMENSIONAL PRIORITY (CRITICAL)
If the query mentions breakdowns:
- By config (e.g., "BHK", "bedroom"): Select segmented keys (e.g., "BHK wise ...").
- By type (e.g., "property type", "flat", "shop"): Select segmented keys (e.g., "Property type wise ...").
- By demography (e.g., "pincode", "age"): Select segmented keys (e.g., "Buyer Pincode ...", "Age Range ...").
- Always prefer segmented over aggregate if breakdown is implied.

### 🔹 LAYER 3 — COMPARISON TYPE ADAPTATION
Adapt selection to the analysis level (inferred from query or keys):
- For locations/cities: Include broad aggregates and shares (e.g., "broad property types Share (%)").
- For projects: Include project-specific details (e.g., "Type of Project", "Total Phases of Project").
- Ensure keys are versatile across levels; select minimum that cover the query without assuming type.

### 🔹 LAYER 4 — CONTRADICTION & DRIFT CHECK (SELF-LEARNING LOOP)

Before finalizing output, perform this internal validation:

1. Did I accidentally switch **Supply ↔ Demand** due to filters?
2. Did I mix **Metadata with Transactions**?
3. Did I select **more keys than strictly required**?
4. Can **one key** answer the question instead of many?

If **YES** to any → correct internally before output.

⚠️ **GENERAL MAPPING GUIDANCE** (NEUTRAL)
- Match keys to query intent via token overlap and category.
- For demand: Use keys with "sold", "consumed", "transactions".
- For demography: Use keys with "pincode", "age range", "buyer".
- For price: Use keys with "price", "rate", "sales (INR)".
- Avoid bias: No default to specific keys; evaluate all candidates equally.

Classify the metric into **ONE and ONLY ONE** of the following:

| Metric Type        | Examples                                                 | Mapping Category |
| ------------------ | -------------------------------------------------------- | ---------------- |
| **SUPPLY**         | supplied, total, planned, inventory, capacity, available | Supply           |
| **DEMAND**         | sold, purchased, booked, absorbed, transactions          | Demand           |
| **PRICE / VALUE**  | price, rate, sales value                                 | Pricing          |
| **METADATA**       | name, city, developer, date                              | Metadata         |
| **INFRASTRUCTURE** | towers, buildings, floors                                | Infrastructure   |
| **COMPOSITION**    | percentage, share, mix                                   | Composition      |
| **BUYER PROFILE**  | buyer pincode, age, origin                               | Demographic      |



## 🧩 RULES
- SELECT ONLY FROM CANDIDATE_KEYS.
- MINIMUM KEYS: Use the fewest that fully answer the query (aim 3-5; MAX 6-8).
- NEVER MIX unrelated categories (e.g., supply + price unless query compares).
- OUTPUT ONLY: JSON array, e.g., ["Key1", "Key2"].
- No commentary, explanations, or extra text.


### ❌ NEVER MIX:

* Supply + Demand
* Metadata + Sales
* Infrastructure + Units (unless explicitly asked)
"""

    try:
        raw_resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw_text = getattr(raw_resp, "content", None) or str(raw_resp)
        start, end = raw_text.find("["), raw_text.rfind("]") + 1
        if start == -1 or end <= 0:
            raise ValueError("Planner did not return JSON array")
        parsed = json.loads(raw_text[start:end])
        filtered = [key for key in parsed if key in candidate_keys]

        # --- PROCESS-LEVEL DIMENSIONAL ENFORCEMENT ---
        # Ensure that if a dimension is in query, at least one segmented key is selected
        dimensions_to_check = []
        if property_mentions["has_bhk_type"]: dimensions_to_check.append("bhk wise")
        if "pincode" in q_low or "buyer" in q_low: dimensions_to_check.append("pincode")
        if "age" in q_low: dimensions_to_check.append("age range")
        
        for dim in dimensions_to_check:
            # Check if any currently selected key covers this dimension
            if not any(dim in k.lower() for k in filtered):
                # Search candidate pool for a key matching BOTH dimension AND metric
                potential_keys = [k for k in candidate_keys if dim in k.lower()]
                # Metric keywords to narrow down
                metric_keywords = ["sold", "units", "carpet area", "sales"] if metric_type in ("demand", "both", "unknown") else ["total units", "supplied", "unsold"]
                
                scored_candidates = []
                for k in potential_keys:
                    k_l = k.lower()
                    m_score = sum(1 for kw in metric_keywords if kw in k_l)
                    scored_candidates.append((k, m_score))
                
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                if scored_candidates and scored_candidates[0][0] not in filtered:
                    filtered.append(scored_candidates[0][0])

        # Always include Location metadata if query asks for location/village names
        if "location" in q_low and "Location" in candidate_keys and "Location" not in filtered:
            filtered.insert(0, "Location")

        # Fallback to general classification rules if LLM returned nothing
        if not filtered:
             raise Exception("LLM returned no valid keys")

        return filtered[:8]

    except Exception as exc:
        logger.error("[planner_identify_mapping_keys] GLOBAL FALLBACK: %s", exc)
        q_low_f = (query or "").lower()
        
        # --- DYNAMIC SCORING FALLBACK ---
        intent_dimensions = []
        if any(tok in q_low_f for tok in ["bhk", "bedroom", "units"]): intent_dimensions.append("bhk wise")
        if any(tok in q_low_f for tok in ["pincode", "pin code", "buyer"]): intent_dimensions.append("pincode")
        if "age" in q_low_f: intent_dimensions.append("age range")
        if any(tok in q_low_f for tok in ["property", "residential", "commercial", "flat", "shop"]): intent_dimensions.append("property type")

        metric = _classify_metric(query)
        q_tokens = set(re.findall(r"[\w>]+", q_low_f))
        
        scored_keys = []
        for key in candidate_keys:
            k_low = key.lower()
            score = 0
            # Dimension Scoring
            for dim in intent_dimensions:
                if dim in k_low: score += 10
            # Metric Matching
            if metric == "supply" and any(tok in k_low for tok in ["total units", "supply", "carpet area supplied"]): score += 5
            if metric == "demand" and any(tok in k_low for tok in ["sold", "consumed", "consumption", "sales"]): score += 5
            # Contradiction Penalty
            if metric == "supply" and any(tok in k_low for tok in ["sold", "consumed", "sales"]): score -= 30
            if metric == "demand" and any(tok in k_low for tok in ["total units", "supplied", "unsold"]): score -= 30
            # Token overlap
            for tok in q_tokens:
                if len(tok) > 3 and tok in k_low: score += 1
            scored_keys.append((key, score))

        scored_keys.sort(key=lambda x: x[1], reverse=True)
        filtered = [k[0] for k in scored_keys if k[1] > 0]
        
        if "location" in q_low_f and "Location" in candidate_keys and "Location" not in filtered:
            filtered.insert(0, "Location")

        return filtered[:8] if filtered else candidate_keys[: min(6, len(candidate_keys))]


def agent_pick_relevant_columns(llm, query: str, selected_keys: List[str], candidate_columns: List[str]) -> List[str]:
    """
    Column Agent: Selects relevant columns from candidates based on query and mapping keys.
    """
    if not candidate_columns:
        return []

    sys_instr = (
        "You select strictly relevant dataframe column names for the user's analytics query. "
        "Return ONLY a JSON list of exact column names from CANDIDATE_COLUMNS—no extra text."
    )
    prompt = f"""User Query: {query}
Selected Mapping Keys: {json.dumps(selected_keys, indent=2)}
CANDIDATE_COLUMNS: {json.dumps(candidate_columns, indent=2)}

RULES:
1. Select at least ONE column per mapping key.
2. If dimensional breakdown (BHK, Pincode) is requested, prioritize specific columns for those segments.
3. Max 25-40 columns for detailed breakdowns.
4. Return ONLY a JSON array.
"""
    try:
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        s, e = raw.find("["), raw.rfind("]") + 1
        if s == -1 or e <= 0: raise ValueError("Invalid JSON")
        picked = json.loads(raw[s:e])
        picked = [c for c in picked if c in candidate_columns]
        
        if not picked and selected_keys:
            # Simple heuristic fallback
            for key in selected_keys:
                for col in candidate_columns:
                    if key.lower() in col.lower(): picked.append(col)
        
        return list(dict.fromkeys(picked)) or candidate_columns[: min(15, len(candidate_columns))]

    except Exception as exc:
        logger.warning("[agent_pick_relevant_columns] fallback: %s", exc)
        fallback = []
        for key in selected_keys:
            for col in candidate_columns:
                if key.lower() in col.lower(): fallback.append(col)
        return fallback[:25] or candidate_columns[: min(15, len(candidate_columns))]


def agent_correction_mapping(llm, query: str, old_keys: List[str], candidate_keys: List[str]) -> Dict[str, Any]:
    """
    Correction Agent for Thumbs Down feedback.
    """
    sys_instr = "You are a correction assistant. User disliked previous answer. Provide BETTER mapping keys."
    prompt = f"Query: {query}\nRejected: {json.dumps(old_keys)}\nCandidates: {json.dumps(candidate_keys)}\nReturn JSON: {{\"reasoning_trace\": \"...\", \"corrected_keys\": [...]}}"
    
    try:
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        s, e = raw.find("{"), raw.rfind("}") + 1
        return json.loads(raw[s:e])
    except:
        return {"reasoning_trace": "Fallback due to error", "corrected_keys": candidate_keys[:3]}


def universal_validate_selection(query: str, selected_keys: List[str], selected_columns: List[str]) -> Dict[str, Any]:
    """Simple validation."""
    errors = []
    warnings = []
    if not selected_keys: errors.append("No keys selected")
    if not selected_columns: errors.append("No columns selected")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggestions": [],
        "query_categories": {}
    }

def validate_selected_columns(query: str, selected_keys: List[str], selected_columns: List[str]) -> Dict[str, Any]:
    return universal_validate_selection(query, selected_keys, selected_columns)