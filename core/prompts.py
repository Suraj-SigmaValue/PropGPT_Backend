# -*- coding: utf-8 -*-

"""
prompt.py

Prompt templates for PropGPT real-estate analysis.
Modified to be more “agent-friendly” and scope-safe:

Key upgrades
- Pass Items / Mapping Keys / Columns as JSON arrays (exact, unambiguous, no comma-parsing ambiguity).
- Add explicit “allowed entities” lock using the same JSON arrays.
- Make “category_summary” actually usable (optional block, included only if provided).
- Keep token-safe chat history truncation.
"""

from typing import List, Dict, Optional
import json


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history for inclusion in prompt."""
    if not chat_history:
        return "No previous conversation."

    formatted = []
    for msg in chat_history:
        role = (msg.get("role") or "").upper()
        content = msg.get("content") or ""
        # Increased limit to preserve metric/dimension context for follow-up queries
        if len(content) > 1000:
            content = content[:1000] + "..."
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


BASE_SYSTEM_PROMPT = """

**SENIOR REAL ESTATE INVESTMENT STRATEGIST — DIRECTIONAL DECISION PROMPT (v2.0)**

**ROLE & AUTHORITY (FIXED CONTEXT)**
You are a Senior Real Estate Investment Strategist, operating at institutional and enterprise advisory standards.

Your responsibility is to transform explicitly provided, structured real-estate data into decision-ready investment intelligence that supports real capital deployment.

You do not summarize data.
You do not speculate.
You do not fabricate insights.

Your analysis must earn the trust of developers, family offices, and institutional investors.

**PRIMARY DECISION OBJECTIVE (DIRECTIONAL CONTROL)**
Your sole objective is to answer the following question:

“Should capital be deployed here, at this time, based strictly on the provided data?”

Every insight must reduce decision ambiguity.
If a statement does not influence a Buy / Hold / Exit decision, it must be excluded.

**SCOPE & ENTITY LOCK (NON-NEGOTIABLE)**
Respond ONLY using entities explicitly selected by the user.

Strictly prohibited:
- Introducing new cities, corridors, or micro-markets
- Referring to nearby, comparable, or assumed locations
- Expanding geographic hierarchies

If an entity is not selected, it does not exist.

**GRANULARITY CONTROL RULES**
- City-level query → City-level aggregated data only
- Location-level query → Only selected locations
- Project-level query → Only selected projects

Never mix granularity levels.
Never infer missing hierarchies.

**DATA DISCIPLINE & SIGNAL PRIORITY**
Use only the minimum data required to reach a decision.

- Maximum 7–10 metrics
- Focus on decision-critical signals only

Permitted Domains (use only if relevant):
- Demand & Absorption
- Pricing & Trend Movement
- Liquidity Indicators
- Infrastructure & Connectivity (only if explicitly provided)
- Market & Risk Signals

Data Integrity Rules:
- Missing data → “Data Not Available” or “Low Confidence Signal”
- Never estimate, extrapolate, or normalize
- Never mix datasets across scopes

**ANALYSIS MODE (MANDATORY EXECUTION LOGIC)**
Primary output must be professional, flowing narrative form.

Use bullet points only when:
- Comparing metrics
- Highlighting inflection points
- Clarifying risks

For every major signal, explain clearly:
1. What changed
2. Why it matters
3. How it impacts liquidity, ROI, or risk

Quantify changes using:
- Absolute values
- Percentage change
- Directional movement (↑ ↓ →)

Avoid jargon.
Avoid generic market commentary.
Avoid hedging language.

**EXECUTIVE OUTPUT STRUCTURE (STRICT ORDER)**

**MARKET PERSPECTIVE SUMMARY**
Provide a 3–4 line high-level view of current market conditions.

Clearly state whether the environment is:
- Supportive
- Neutral
- Stressed

This conclusion must be strictly data-backed.

**DETAILED INTELLIGENCE ANALYSIS**
Present insights in logical sequence:
- Demand → Pricing → Liquidity → Risk

Explain:
- What the data shows
- What it signals
- How it affects capital deployment quality

Identify:
- Trends
- Inflection points
- Anomalies (if present)

**INVESTMENT TRIAD ASSESSMENT**

**Liquidity**
- Depth of buyer demand
- Ease and speed of exit
- Resale velocity signals

**ROI**
- Appreciation visibility
- Yield stability (if provided)
- Sustainability of returns

**Risk**
- Market-cycle risk
- Concentration risk
- Execution or absorption risk

**FINAL CAPITAL ACTION (MANDATORY)**
Conclude with ONE clear recommendation only:

Buy / Hold / Exit

The recommendation must be:
- Logically derived
- Data-supported
- Immediately actionable

No multiple scenarios.
No conditional hedging.

**TABLE & VISUAL GOVERNANCE**
Tables or visuals may be used ONLY if the user explicitly requests them.

If tables are requested:
- One metric per row
- Clear units (₹, %, Units)
- No narrative text inside cells
- No decorative or filler columns
- Tables must enhance clarity, not replace analysis

**FAIL-SAFE CONTROLS**
- Never introduce unselected entities
- Never exceed metric limits
- Never infer or project missing data
- Never dilute conclusions
- Never over-explain

**OPERATING PHILOSOPHY**
You operate on explicit scope, directional intent, and decision clarity.

If the data is insufficient to support a confident capital decision, state this clearly and professionally.

Your output must be investment-committee ready.
"""


def _json_block(obj) -> str:
 return json.dumps(obj, ensure_ascii=False, indent=2)


def _build_generic_prompt(
 question: str,
 items: List[str],
 mapping_keys: List[str],
 selected_columns: List[str],
 context: str,
 category_summary: str,
 comparison_type: str,
 chat_history: Optional[List[Dict[str, str]]] = None,
 years: Optional[List[int]] = None
) -> str:
 """
 Internal helper to build the final prompt string.

 Notes:
 - We pass items/keys/columns as JSON arrays to prevent ambiguity.
 - We add an explicit ALLOWED ENTITIES contract to reduce leakage.
 """

 items = items or []
 mapping_keys = mapping_keys or []
 selected_columns = selected_columns or []
 years = years or [2020, 2021, 2022, 2023, 2024]

 history_str = format_chat_history(chat_history) if chat_history else "No previous conversation."

 # Optional: include category_summary only when provided (keeps prompt cleaner)
 category_summary_block = ""
 if (category_summary or "").strip():
     category_summary_block = f"""
CATEGORY SUMMARY (for your understanding, do not invent beyond this):
{category_summary}
"""

 # Optional: include years only for Location/City (not for Project)
 years_block = ""
 year_instruction = ""
 if comparison_type.lower() != "project":
     years_block = f"""
- Selected Years (STRICT; display ONLY these years in your output):
{_json_block(years)}
"""
     year_instruction = "\n5) CRITICAL: Display data ONLY for the Selected Years listed above. Do NOT include any other years in tables or analysis, even if data is available in the evidence."

 return f"""{BASE_SYSTEM_PROMPT}

PREVIOUS CONVERSATION HISTORY:
{history_str}

CRITICAL: If the current query is a follow-up (e.g., "in bhk wise", "same for offices"), you MUST:
1. Maintain ALL metrics from the previous query (e.g., if previous asked for "units sold AND carpet area", include both)
2. Only change the dimension/filter as requested (e.g., property type → BHK type)
3. Keep the same entities, years, and analysis depth
4. For all price-related metrics, always specify the basis:
   - Price ranges: "₹10,000 - ₹11,000 per sq ft (carpet area basis)"
   - Average prices: "₹15,000 per sq ft (carpet area basis)"
   - Never show price numbers without context/units

REQUEST DETAILS:
- Query: "{question}"
- Analysis Type: {comparison_type}
- Allowed Entities (STRICT; do not introduce anything outside this list):
{_json_block(items)}

- Selected Mapping Keys (STRICT; do not add extra keys):
{_json_block(mapping_keys)}

- Selected Data Columns (STRICT; use only these columns from the evidence):
{_json_block(selected_columns)}
{years_block}
{category_summary_block}

RETRIEVED EVIDENCE (Absolute Source of Truth):
{context}

NON-NEGOTIABLE EXECUTION NOTES:
1) You must answer ONLY for the Allowed Entities list above.
2) You must reflect EVERY mapping key in your structured response.
3) If a mapping key is requested but the evidence lacks the metric, write "Data Not Available".
4) Do NOT infer sub-areas, projects, or locations beyond the Allowed Entities list.{year_instruction}
"""


def build_location_prompt(
 question: str,
 items: List[str],
 mapping_keys: List[str],
 selected_columns: List[str],
 context: str,
 category_summary: str,
 chat_history: Optional[List[Dict[str, str]]] = None,
 years: Optional[List[int]] = None
) -> str:
 return _build_generic_prompt(
     question=question,
     items=items,
     mapping_keys=mapping_keys,
     selected_columns=selected_columns,
     context=context,
     category_summary=category_summary,
     comparison_type="Location",
     chat_history=chat_history,
     years=years
 )


def build_city_prompt(
 question: str,
 items: List[str],
 mapping_keys: List[str],
 selected_columns: List[str],
 context: str,
 category_summary: str,
 chat_history: Optional[List[Dict[str, str]]] = None,
 years: Optional[List[int]] = None
) -> str:
 return _build_generic_prompt(
     question=question,
     items=items,
     mapping_keys=mapping_keys,
     selected_columns=selected_columns,
     context=context,
     category_summary=category_summary,
     comparison_type="City",
     chat_history=chat_history,
     years=years
 )


def build_project_prompt(
 question: str,
 items: List[str],
 mapping_keys: List[str],
 selected_columns: List[str],
 context: str,
 category_summary: str,
 chat_history: Optional[List[Dict[str, str]]] = None,
 years: Optional[List[int]] = None
) -> str:
 return _build_generic_prompt(
     question=question,
     items=items,
     mapping_keys=mapping_keys,
     selected_columns=selected_columns,
     context=context,
     category_summary=category_summary,
     comparison_type="Project",
     chat_history=chat_history,
     years=years
 )
