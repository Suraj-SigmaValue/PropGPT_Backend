"""
Query Intelligence Agents for PropGPT.

Contains two agents:
1. Planner Agent: Selects relevant mapping keys based on user query
2. Column Agent: Selects relevant columns based on query and mapping keys
"""

import json
import re
import logging
from typing import List
from typing import Dict, Any

logger = logging.getLogger(__name__)


def planner_identify_mapping_keys(llm, query: str, candidate_keys: List[str]) -> List[str]:
    """
    Planner Agent: Identifies and selects the most relevant mapping keys for the user query.
    
    Args:
        llm: Language model instance (LangChain LLM)
        query: User's analysis query
        candidate_keys: List of available mapping keys to choose from
    
    Returns:
        List of selected mapping keys (at most 6, or fewer if fewer are relevant)
    """
    if not candidate_keys:
        return []

    # -------------------------
    # Local helper: classify query by type (BHK config etc.)
    # -------------------------
    def _classify_query_type(q: str) -> str:
        q = (q or "").lower()

        # BHK / configuration demand-style queries
        bhk_tokens = [
            "configuration", "configurations", "config mix",
            "smaller units", "larger units",
            "smaller configuration", "larger configuration",
            "small size", "big size", "bigger units",
            "1bhk", "2bhk", "3bhk", "4bhk", "bhk",
        ]
        if any(tok in q for tok in bhk_tokens):
            return "bhk_config"

        return "generic"

    # -------------------------
    # Local helper: classify metric as supply / demand / both
    # -------------------------
    def _classify_metric(q: str) -> str:
        q = (q or "").lower()

        supply_tokens = [
            "supply", "available", "inventory", "stock",
            "total units", "unsold", "launched", "supplied",
        ]
        demand_tokens = [
            "demand", "sold", "purchased", "bought",
            "transactions", "sales count", "absorbed", "consumed",
        ]

        has_supply = any(tok in q for tok in supply_tokens)
        has_demand = any(tok in q for tok in demand_tokens)

        if has_supply and has_demand:
            return "both"
        if has_supply:
            return "supply"
        if has_demand:
            return "demand"
        return "unknown"

    # -------------------------
    # Local helper: detect demography / pincode analysis queries
    # -------------------------
    def _is_demography_query(q: str) -> bool:
        q = (q or "").lower()
        demo_tokens = [
            "demography", "demographic", "demographics",
            "buyer profile", "buyer mix", "age profile",
            "pincode", "pin code", "area-wise buyers", "buyer location",
            "top 10 buyer", "top buyer area", "top buyer pincode",
        ]
        return any(tok in q for tok in demo_tokens)

    # -------------------------
    # Local helper: detect property type and BHK mentions
    # -------------------------
    def _detect_property_mentions(q: str) -> dict:
        q = (q or "").lower()
        return {
            "has_property_type": any(tok in q for tok in ["flat", "shop", "office", "property type", "property"]),
            "has_bhk_type": any(tok in q for tok in ["1bhk", "2bhk", "3bhk", "4bhk", "bhk", "bedroom", "configuration"])
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

You are a precise Query-Classification & Mapping-Key Selection Agent.
Your ONLY task is to identify the most relevant mapping keys required to answer the user query.

You must NOT generate explanations, insights, or summaries.
Your output must be a JSON array of mapping keys ONLY.

────────────────────────────────────
STEP 1: IDENTIFY QUERY INTENT
────────────────────────────────────

Determine what the user is asking for by classifying the query into ONE primary intent.
Use keyword meaning, not surface words.

────────────
A. PROJECT METADATA / PROFILE
────────────
Questions related to static project information such as:
- Project name, location, city
- Developer / organization / individual
- Project type
- Commencement or completion dates
- Number of phases
- RERA or registration-level attributes

Select ONLY from:
- "Project Name"
- "Project Location"
- "Project City"
- "Type of Project"
- "Project commencement date"
- "Project Completion date"
- "Total Phases of Project"
- "Name of the organization/ Indvidual"

────────────
B. INFRASTRUCTURE / PHYSICAL CAPACITY
────────────
Questions about physical construction attributes:
- Number of buildings or towers
- Floor count (min/max/total)

Select ONLY from:
- "No of Buildings or Towers in Project"
- "total floors"

────────────
C. COMPOSITION / SHARE / MIX
────────────
Questions asking for percentage distribution or composition:
- Residential vs commercial share
- Property type mix
- Percentage contribution

Select ONLY:
- "broad property types Share (%)"

────────────
D. DEMOGRAPHIC / BUYER PROFILE
────────────
Questions related to buyer characteristics:
- Buyer pincodes
- Top buyer locations
- Buyer distribution by property type or BHK
- Age range or buyer profile

Select ONLY from:
- "Top 10 Buyer Pincode units sold"
- "Property type wise Top 10 Buyer Pincode Percentage(%)"
- "BHK wise Top 10 Buyer Pincode Unit Sold Percentage(%)"
- Relevant age-range keys (if applicable)

────────────
E. SUPPLY (PLANNED / TOTAL / INVENTORY)
────────────
Questions about units that exist or are planned, NOT sold:
- Total units
- Units planned
- Available inventory
- Supply or capacity

Select ONLY from:
- "total units"
- "Property Type wise total units"
- "BHK wise total units"

⚠️ NEVER select sold or transaction-related keys for supply queries.

────────────
F. DEMAND / SOLD UNITS
────────────
Questions about sales or transactions:
- Units sold
- Flats sold
- Shops sold
- Demand, purchases, transactions

Select ONLY from:
- "Units sold"
- "Property type wise Units Sold"
- "BHK types wise units sold"

────────────
G. PRICING / VALUE / RATES
────────────
Questions related to pricing or monetary value:
- Price per sq ft
- Average pricing
- Total sales value

Select ONLY from:
- "Property Type Wise Average Price per Sq. Ft. (Carpet Area Basis)"
- "Total sales (INR)"

────────────────────────────────────
STEP 2: APPLY CRITICAL SELECTION RULES
────────────────────────────────────

1. SUPPLY vs DEMAND (STRICT)
- Supply indicators: "total", "planned", "available", "inventory", "capacity"
  → Select ONLY supply keys
- Demand indicators: "sold", "purchased", "transactions", "bought"
  → Select ONLY demand keys
- Ambiguous query like:
  "How many flats are there?"
  → DEFAULT to DEMAND (sold units)

2. METADATA vs SALES DATA
- Project identity, dates, developer, phases
  → Metadata keys ONLY
- Never mix metadata with unit sold keys unless explicitly requested

3. INFRASTRUCTURE vs UNITS
- Buildings / towers / floors → Infrastructure
- Units planned or total → Supply
- Units sold → Demand

4. KEY MINIMIZATION
- Select ONLY the minimum required keys
- Do NOT include redundant or loosely related keys
- Prefer fewer keys if sufficient

5. MAXIMUM MAPPING KEY LIMIT (HARD RULE)
- You MUST select a maximum of 4–5 mapping keys.
- Selecting more than 10 keys is NOT allowed.
- If more than 10 keys seem relevant:
  → Prioritize the most critical keys that directly answer the query.
  → Drop secondary or descriptive keys.
- If fewer than 8 keys are sufficient, select fewer.


────────────────────────────────────
STEP 3: VALIDATION CHECK
────────────────────────────────────

Before finalizing:
- Confirm selected keys directly answer the query
- Confirm no conflicting categories are mixed
- Confirm no unnecessary keys are included

────────────────────────────────────
STEP 4: OUTPUT FORMAT (STRICT)
────────────────────────────────────

Return ONLY a valid JSON array of selected mapping keys.
No explanations.
No markdown.
No extra text.

Example Output:
[
  "Project Name",
  "Project Location"
]
    """
    try:
        raw_resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw_text = getattr(raw_resp, "content", None) or str(raw_resp)
        start, end = raw_text.find("["), raw_text.rfind("]") + 1
        if start == -1 or end <= 0:
            raise ValueError("Planner did not return JSON array")
        parsed = json.loads(raw_text[start:end])
        if not isinstance(parsed, list):
            raise ValueError("Planner output is not a list")
        filtered = [key for key in parsed if key in candidate_keys]

        # -------------------------
        # Deterministic hard rules (do NOT rely only on LLM)
        # -------------------------

        # 1) If query is about "location", ensure "Location" mapping key is present
        if "location" in q_low and "Location" in candidate_keys and "Location" not in filtered:
            filtered.insert(0, "Location")

        # 2) If query is BHK/configuration-style, ensure BHK demand key is present
        if q_type == "bhk_config":
            if "BHK types wise units sold" in candidate_keys and "BHK types wise units sold" not in filtered:
                filtered.append("BHK types wise units sold")

        # 3) SUPPLY vs DEMAND ENFORCEMENT with granular property/BHK detection
        if metric_type in ("supply", "both"):
            # Priority order for supply keys based on query specificity
            supply_key_priority = []
            
            if property_mentions["has_bhk_type"]:
                # BHK-specific supply query
                supply_key_priority.extend([
                    "BHK wise total units",
                    "BHK wise total carpet area in sqft",
                    "Property Type wise total units", 
                    "total units"
                ])
            elif property_mentions["has_property_type"]:
                # Property-type-specific supply query
                supply_key_priority.extend([
                    "Property Type wise total units",
                    "Property type wise Total Carpet Area (in sq ft)",
                    "total units"
                ])
            else:
                # General supply query
                supply_key_priority.extend([
                    "total units",
                    "Total Carpet Area (In sq ft)",
                    "Property Type wise total units",
                    "BHK wise total units"
                ])
            
            # Add priority supply keys that are available
            for key in supply_key_priority:
                if key in candidate_keys and key not in filtered:
                    filtered.append(key)

            # CRITICAL FIX: For supply-only queries, REMOVE demand/sold keys
            if metric_type == "supply":
                # Remove any demand-related keys
                filtered = [k for k in filtered if not (
                    "units sold" in k.lower() or 
                    "sold" in k.lower() or 
                    "total sales" in k.lower() or
                    "demand" in k.lower()
                )]
            else:
                # For "both", push sold/sales keys to the end
                demand_like = []
                non_demand_like = []
                for k in filtered:
                    if ("units sold" in k.lower()) or ("sold" in k.lower()) or ("total sales" in k.lower()):
                        demand_like.append(k)
                    else:
                        non_demand_like.append(k)
                filtered = non_demand_like + demand_like

        # 4) DEMOGRAPHY / PINCODE ENFORCEMENT
        if is_demo:
            demo_key = "Top 10 Buyer Pincode units sold"
            if demo_key in candidate_keys and demo_key not in filtered:
                filtered.append(demo_key)

        # Limit to max 6 keys, keep existing behaviour if list is empty
        if not filtered:
            return candidate_keys[: min(6, len(candidate_keys))]
        return filtered[:6]

    except Exception as exc:
        logger.warning("[planner_identify_mapping_keys] fallback due to: %s", exc)
        q_low = (query or "").lower()
        query_tokens = set(re.findall(r"[\w>]+", q_low))
        heuristic = [
            key for key in candidate_keys
            if any(token in key.lower() for token in query_tokens)
        ]

        filtered = heuristic or candidate_keys[: min(6, len(candidate_keys))]

        # Same deterministic guarantees in fallback

        if "location" in q_low and "Location" in candidate_keys and "Location" not in filtered:
            filtered.insert(0, "Location")

        if _classify_query_type(query) == "bhk_config":
            if "BHK types wise units sold" in candidate_keys and "BHK types wise units sold" not in filtered:
                filtered.append("BHK types wise units sold")

        # Enhanced supply handling in fallback
        metric_type_fallback = _classify_metric(query)
        property_mentions_fallback = _detect_property_mentions(query)
        
        if metric_type_fallback in ("supply", "both"):
            supply_key_priority = []
            
            if property_mentions_fallback["has_bhk_type"]:
                supply_key_priority.extend([
                    "BHK wise total units",
                    "BHK wise total carpet area in sqft",
                    "Property Type wise total units", 
                    "total units"
                ])
            elif property_mentions_fallback["has_property_type"]:
                supply_key_priority.extend([
                    "Property Type wise total units",
                    "Property type wise Total Carpet Area (in sq ft)",
                    "total units"
                ])
            else:
                supply_key_priority.extend([
                    "total units",
                    "Total Carpet Area (In sq ft)",
                    "Property Type wise total units",
                    "BHK wise total units"
                ])
            
            for key in supply_key_priority:
                if key in candidate_keys and key not in filtered:
                    filtered.append(key)

            # CRITICAL FIX: For supply-only queries, REMOVE demand/sold keys (fallback)
            if metric_type_fallback == "supply":
                # Remove any demand-related keys
                filtered = [k for k in filtered if not (
                    "units sold" in k.lower() or 
                    "sold" in k.lower() or 
                    "total sales" in k.lower() or
                    "demand" in k.lower()
                )]
            else:
                # For "both", push sold/sales keys to the end
                demand_like = []
                non_demand_like = []
                for k in filtered:
                    if ("units sold" in k.lower()) or ("sold" in k.lower()) or ("total sales" in k.lower()):
                        demand_like.append(k)
                    else:
                        non_demand_like.append(k)
                filtered = non_demand_like + demand_like

        if _is_demography_query(query):
            demo_key = "Top 10 Buyer Pincode units sold"
            if demo_key in candidate_keys and demo_key not in filtered:
                filtered.append(demo_key)

        return filtered[:6]


def agent_pick_relevant_columns(llm, query: str, selected_keys: List[str], candidate_columns: List[str]) -> List[str]:
    """
    Column Agent: Selects the most relevant columns from candidates based on user query and selected keys.
    
    Args:
        llm: Language model instance (LangChain LLM)
        query: User's analysis query
        selected_keys: List of selected mapping keys (from planner agent)
        candidate_columns: List of available columns to choose from
    
    Returns:
        List of selected column names (typically 7-20 relevant columns)
    """
    if not candidate_columns:
        return []

    sys_instr = (
        "You select strictly relevant dataframe column names for the user's analytics query. "
        "Return ONLY a JSON list of exact column names from CANDIDATE_COLUMNS—no extra text."
    )
    prompt = f"""User Query:
{query}

Selected Mapping Keys (Context Only — Do Not Output):
{json.dumps(selected_keys, indent=2)}

CANDIDATE_COLUMNS (Allowed Selection Only):
{json.dumps(candidate_columns, indent=2)}

You are a strict Column-Selection Agent.
Your ONLY task is to choose the most relevant columns required to answer the user query,
based on the already selected mapping keys.

You must NOT generate explanations, summaries, or analysis.
You must ONLY return a JSON array of column names.

────────────────────────────────────
CORE SELECTION RULES (STRICT)
────────────────────────────────────

1. RELEVANCE ONLY
- Select ONLY columns that directly help answer the user query.
- Avoid generic, metadata-only, duplicate, or noisy columns.
- If a column does not materially improve the answer, DO NOT select it.

2. MAPPING KEY COVERAGE (MANDATORY)
- You MUST select at least ONE column for EACH mapping key listed in "Selected Mapping Keys".
- No mapping key may be ignored.
- If a mapping key has multiple relevant columns, select the most descriptive and informative ones.

3. COLUMN COUNT LIMIT
- Select the smallest possible set of columns that is still sufficient.
- Preferred range: 7–20 columns total.
- If fewer than 10 columns are sufficient, select fewer.
- Selecting unnecessary columns is considered an error.

4. DEMAND-SPECIFIC RULE
- If the user query refers to "Demand" AND no specific demand subtype is mentioned:
  → Default to selecting columns related to:
    "Property type wise Units Sold"
- Do NOT mix supply-related columns unless explicitly requested.

5. CONFLICT AVOIDANCE
- Do NOT mix:
  - Supply columns with Demand queries
  - Pricing columns with Unit count queries
  - Metadata columns with transactional analysis
- Every selected column must align with BOTH:
  - The user query intent
  - The selected mapping keys

────────────────────────────────────
VALIDATION CHECK (INTERNAL)
────────────────────────────────────

Before finalizing the output:
- Confirm each selected column exists in CANDIDATE_COLUMNS
- Confirm each mapping key has at least one selected column
- Confirm no redundant or overlapping columns are included
- Confirm column count is minimal and sufficient

────────────────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────────────────

Return ONLY a valid JSON array of column names.
No markdown.
No explanations.
No extra text.

Example Output:
[
  "Property Type",
  "Units Sold",
  "BHK Type"
]

    """
    try:
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        s, e = raw.find("["), raw.rfind("]") + 1
        if s == -1 or e <= 0:
            raise ValueError("Agent did not return a JSON list.")
        picked = json.loads(raw[s:e])
        if not isinstance(picked, list):
            raise ValueError("Agent output is not a list.")
        picked = [c for c in picked if c in candidate_columns]
        picked = list(dict.fromkeys(picked))
        
        # If agent returned empty, fallback to simple matching based on keys
        if not picked and selected_keys:
            logger.info("Agent returned 0 columns, using fallback based on keys.")
            fallback = []
            for key in selected_keys:
                for col in candidate_columns:
                    # Loose matching: if key is part of column name
                    if key.lower() in col.lower():
                        fallback.append(col)
            picked = fallback[:20]

        return picked or candidate_columns[: min(15, len(candidate_columns))]

    except Exception as exc:
        logger.warning("[agent_pick_relevant_columns] fallback due to: %s", exc)
        # Fallback: select columns that loosely match the selected keys
        fallback = []
        if selected_keys:
            for key in selected_keys:
                for col in candidate_columns:
                    if key.lower() in col.lower():
                        fallback.append(col)
        
        return fallback[:20] or candidate_columns[: min(15, len(candidate_columns))]


def agent_correction_mapping(llm, query: str, old_keys: List[str], candidate_keys: List[str]) -> List[str]:
    """
    Correction Agent: Proposes NEW mapping keys assuming the old ones were incorrect (Thumbs Down).
    
    Args:
        llm: Language model instance
        query: User's original query
        old_keys: The keys used in the rejected response
        candidate_keys: All available keys
        
    Returns:
        New list of mapping keys
    """
    if not candidate_keys:
        return []

    sys_instr = (
        "You are a correction assistant. The user provided negative feedback (Thumbs Down) for a previous answer. "
        "The previous answer used a specific set of mapping keys which the user ostensibly found incorrect or insufficient. "
        "Your task: Re-analyze the query and select BETTER mapping keys from CANDIDATE_KEYS. "
        "Avoid simply repeating the exact same set if possible, unless you are strictly convinced they are the only correct ones "
        "(in which case, maybe add a missing key). "
        "Return ONLY a JSON list of mapping keys."
    )
    
    prompt = f"""
    ### SYSTEM ROLE: REINFORCEMENT LEARNING CORRECTION AGENT

You are a specialized Mapping-Key Correction Agent operating inside a reinforcement learning loop.
You are invoked ONLY because the previous mapping decision received a NEGATIVE REWARD (User Thumbs Down).

Your task is to diagnose the failure, change strategy, and produce a corrected mapping
that is logically distinct and more aligned with the user’s intent.

────────────────────────────────────
CURRENT STATE
────────────────────────────────────

1. User Query:
"{query}"

2. Rejected Mapping Policy (Incorrect Selection):
{json.dumps(old_keys, indent=2)}

3. Action Space (Allowed Candidate Keys Only):
{json.dumps(candidate_keys, indent=2)}

────────────────────────────────────
OPTIMIZATION OBJECTIVE
────────────────────────────────────

Your objective is to maximize user reward by selecting a NEW and CORRECT mapping strategy.

You must apply Reflexion:
- Identify the logical failure in the previous mapping
- Explicitly change the selection approach
- Avoid repeating the same assumptions

This is NOT a retry — it is a correction.

────────────────────────────────────
STEP 1: DIAGNOSTIC / CRITIQUE PHASE
────────────────────────────────────

Analyze WHY the rejected keys failed.

Check for common failure modes:
- Supply vs Demand inversion (planned vs sold)
- Wrong abstraction level (aggregate vs granular)
- Metadata vs transactional confusion
- Misinterpreted intent (profile vs performance)
- Over-selection or under-selection of keys
- Selection of descriptive keys instead of analytical keys

Assume:
- The user’s negative feedback implies the mapping was logically incorrect or irrelevant
- The issue is with mapping choice, not user phrasing

────────────────────────────────────
STEP 2: STRATEGY SHIFT / CORRECTION PHASE
────────────────────────────────────

Select a DIFFERENT set of mapping keys that better satisfy the query.

Mandatory constraints:
- You MUST NOT output the same set of keys as the rejected policy
- At least one key must be different
- Prefer fewer, higher-signal keys over broad coverage
- Select ONLY from the provided Candidate Keys

Heuristics:
- If the previous mapping was too granular → move more abstract
- If it was too abstract → move more concrete
- If it focused on structure → move to performance
- If it focused on counts → move to value or distribution
- If the query implies higher-level reasoning (e.g., “Anti-Gravity”):
  → Prefer computed, aggregated, or parent-category keys

────────────────────────────────────
KEY SELECTION RULES (STRICT)
────────────────────────────────────

- Select ONLY keys that directly correct the diagnosed failure
- Do NOT add exploratory or “just in case” keys
- Keep the set minimal and purposeful
- Do NOT exceed logical necessity

────────────────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────────────────

Return ONLY valid JSON.
No markdown.
No extra text.

{{
  "reasoning_trace": "Concise explanation of why the previous keys failed and how the new keys correct that failure.",
  "corrected_keys": ["key1", "key2"]
}}

    """
    
    try:
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        s, e = raw.find("["), raw.rfind("]") + 1
        if s == -1 or e <= 0:
            # Fallback: Just return the old keys if parsing fails, or try heuristic
            return old_keys
        
        parsed = json.loads(raw[s:e])
        if not isinstance(parsed, list):
            return old_keys
            
        filtered = [k for k in parsed if k in candidate_keys]
        if not filtered:
             # If agent went rogue and returned invalid keys, fallback to old keys or top candidates
             return candidate_keys[:3]
             
        return filtered
        
    except Exception as e:
        logger.warning(f"Correction agent failed: {e}")
        return old_keys


# Add these validation functions at the end of agents.py
def universal_validate_selection(query: str, selected_keys: List[str], selected_columns: List[str]) -> Dict[str, Any]:
    """
    Simple validation function.
    """
    errors = []
    warnings = []
    suggestions = []
    
    # Check if we have keys and columns
    if not selected_keys:
        errors.append("No mapping keys selected")
        suggestions.append("Please select at least one mapping key")
    
    if not selected_columns:
        errors.append("No columns selected")
        suggestions.append("Please select relevant data columns")
    
    # Simple semantic check based on query
    query_lower = query.lower()
    
    # Check for demographic queries
    if any(term in query_lower for term in ["demographic", "pincode", "age", "buyer"]):
        has_demo = any(any(demo_term in key.lower() for demo_term in ["pincode", "age", "buyer", "demograph"]) 
                      for key in selected_keys)
        if not has_demo:
            warnings.append("Query mentions demographics but no demographic keys selected")
    
    # Check for supply queries
    if any(term in query_lower for term in ["supply", "available", "inventory", "total units"]):
        has_supply = any(any(supply_term in key.lower() for supply_term in ["total units", "available", "supply"])
                        for key in selected_keys)
        if not has_supply:
            warnings.append("Query mentions supply but no supply keys selected")
    
    # Check for demand queries
    if any(term in query_lower for term in ["sold", "demand", "transactions", "sales"]):
        has_demand = any(any(demand_term in key.lower() for demand_term in ["sold", "demand", "transactions"])
                        for key in selected_keys)
        if not has_demand:
            warnings.append("Query mentions demand but no demand keys selected")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "query_categories": {}  # Simple placeholder
    }


def validate_selected_columns(query: str, selected_keys: List[str], selected_columns: List[str]) -> Dict[str, Any]:
    """Wrapper for backward compatibility."""
    return universal_validate_selection(query, selected_keys, selected_columns)


# Add missing import at the top
from typing import Dict, Any