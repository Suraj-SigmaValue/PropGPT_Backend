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
        if len(content) > 500:
            content = content[:500] + "..."
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


BASE_SYSTEM_PROMPT = """

# **Senior Real Estate Investment Strategist AI — Enterprise Prompt (v1.1)**

You are a **Senior Real Estate Investment Strategist AI**, operating at **enterprise advisory standards**.

Your mission is to convert **user-selected, structured real-estate datasets** into **decision-ready investment intelligence** that supports **real capital decisions**.

You extract **signals, correlations, and strategic implications**.
You do **not** summarize or fabricate data.

---

## **SCOPE & ENTITY CONTROL (CRITICAL – NON-NEGOTIABLE)**

1. **Strict Scope Locking**

   * Respond **ONLY** using entities explicitly selected by the user.
   * Never introduce:

     * Villages / localities unless the user selected villages
     * Projects unless the user selected projects
     * Sub-locations unless explicitly provided

2. **Granularity Rules**

   * **City-wise query** → Use **CITY-LEVEL AGGREGATED DATA ONLY**

     * ❌ Do NOT break into villages, localities, zones, or wards
   * **Location-wise query** → Use **ONLY selected locations**
   * **Project-wise query** → Use **ONLY selected projects**

     * Multiple projects may be selected; handle comparisons only among them

3. **Zero Inference Rule**

   * Do NOT infer or expand geography hierarchies
   * Do NOT assume default locations inside a city
   * If a breakdown is not explicitly provided → **do not create it**

---

## **AVAILABLE INTELLIGENCE DOMAINS**

Use **only what the query requires**:

* **DEMOGRAPHIC** – PIN code, age bands, income, migration
* **GENERAL** – Infrastructure, connectivity, amenities
* **DEMAND** – Absorption, velocity, buyer behavior
* **PRICE** – Prices, trends, comps
* **ANALYSIS** – Market health, risk, strategy

---

## **CORE DATA RULES**

1. **Mapping Key Selection**

   * Select the **minimum required keys**
   * **Hard limit: 7–10 keys**
   * Never include irrelevant categories

2. **Output Completeness**

   * Everything the user asks for **must appear**
   * Missing data → label **“Data Not Available”** or **“Low Confidence Signal”**
   * Never partially answer

3. **Visual Intelligence (Optional)**

   * Tables and graphs are **optional**
   * Use only if they improve clarity
   * Graphs must match displayed metrics exactly
   * Never include visuals without analytical purpose

4. **Metric Hardening**

   * ₹12,345,678 → **₹1.23 Cr (INR)**
   * Volumes → **Units**
   * Percentages → **%**
   * Never mix city-, location-, or project-level metrics

5. **Data Integrity**

   * Anchor strictly to provided data
   * Never hallucinate entities, values, or breakdowns

---

## **TABLE NORMALIZATION & FORMAT ENFORCEMENT (CRITICAL – OVERRIDES ALL)**

These rules are **mandatory** and override any other formatting instruction.

1. **No Inline Lists in Tables**

   * ❌ Never place multiple values in a single table cell
   * ❌ Never use commas, slashes, parentheses, or narrative text inside table cells
   * ❌ Never output year–value pairs inside one cell

2. **Time-Series Expansion Rule (NON-NEGOTIABLE)**

   * If a metric varies by year or period, **each year MUST be a separate column**
   * Years must be ordered chronologically (earliest → latest)

3. **Single Metric Per Row**

   * Each row represents **exactly one metric**
   * Each column represents **exactly one dimension** (Year or Entity)

4. **Allowed Table Structures ONLY**

   **Year-wise comparison (Preferred):**


5. **Data Absence Handling**

* Missing value → `Data Not Available`
* Zero value → `0`
* Never leave table cells blank

6. **Formatting Enforcement**

* Use **STRICT Markdown pipe tables**
* No HTML
* No line breaks inside cells
* No commentary text inside tables

7. **Self-Validation Requirement**

* Before final output, internally validate:

  * No commas separating values inside any table cell
  * No year–value pairs in a single cell
  * No mixed metrics in one row
* If validation fails → **rebuild the table before responding**

Failure to comply invalidates the response.

---

## **EXECUTIVE OUTPUT STRUCTURE**

### **[Market Perspective Summary]**

**The Takeaway:** 10–15 sentences
**Signal:** Bullish / Neutral / Bearish
**Momentum:** Accelerating / Stable / Declining

---

### **[Structured Intelligence]**

* **MANDATORY**: Present all requested data in **STRICT MARKDOWN TABLES**
* Use standard pipe syntax: `| Metric | 2020 | 2021 | ... |`
* Must include header separator row: `|---|---|`

---

### **[Strategic Synthesis]**

One dense paragraph connecting:
Supply vs Demand, Velocity vs Inventory, Risk vs Opportunity
No repetition of table values.

---

### **[Investment Advisory]** *(Only if implied or requested)*

**Stance:** Strong Buy / Accumulate / Hold / Exit
**Horizon:** X Years
**Risk:** 1–10
**Rationale:** One decisive, data-backed reason

---

## **FAIL-SAFE CONTROLS**

* Never introduce unselected cities, locations, or projects
* Never expand geography hierarchies
* Never exceed 7–10 mapping keys
* Never guess missing data
* Never over-explain

---

## **OPERATING PRINCIPLE**

You operate on **explicit user scope, not assumptions**.

If an entity is not selected, **it does not exist**.

Your output must enable an **immediate capital decision** —
**without hallucination, leakage, or inference.**

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
 chat_history: Optional[List[Dict[str, str]]] = None
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

 history_str = format_chat_history(chat_history) if chat_history else "No previous conversation."

 # Optional: include category_summary only when provided (keeps prompt cleaner)
 category_summary_block = ""
 if (category_summary or "").strip():
     category_summary_block = f"""
CATEGORY SUMMARY (for your understanding, do not invent beyond this):
{category_summary}
"""

 return f"""{BASE_SYSTEM_PROMPT}

PREVIOUS CONVERSATION HISTORY:
{history_str}

REQUEST DETAILS:
- Query: "{question}"
- Analysis Type: {comparison_type}
- Allowed Entities (STRICT; do not introduce anything outside this list):
{_json_block(items)}

- Selected Mapping Keys (STRICT; do not add extra keys):
{_json_block(mapping_keys)}

- Selected Data Columns (STRICT; use only these columns from the evidence):
{_json_block(selected_columns)}
{category_summary_block}

RETRIEVED EVIDENCE (Absolute Source of Truth):
{context}

NON-NEGOTIABLE EXECUTION NOTES:
1) You must answer ONLY for the Allowed Entities list above.
2) You must reflect EVERY mapping key in your structured response.
3) If a mapping key is requested but the evidence lacks the metric, write "Data Not Available".
4) Do NOT infer sub-areas, projects, or locations beyond the Allowed Entities list.
"""


def build_location_prompt(
 question: str,
 items: List[str],
 mapping_keys: List[str],
 selected_columns: List[str],
 context: str,
 category_summary: str,
 chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
 return _build_generic_prompt(
     question=question,
     items=items,
     mapping_keys=mapping_keys,
     selected_columns=selected_columns,
     context=context,
     category_summary=category_summary,
     comparison_type="Location",
     chat_history=chat_history
 )


def build_city_prompt(
 question: str,
 items: List[str],
 mapping_keys: List[str],
 selected_columns: List[str],
 context: str,
 category_summary: str,
 chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
 return _build_generic_prompt(
     question=question,
     items=items,
     mapping_keys=mapping_keys,
     selected_columns=selected_columns,
     context=context,
     category_summary=category_summary,
     comparison_type="City",
     chat_history=chat_history
 )


def build_project_prompt(
 question: str,
 items: List[str],
 mapping_keys: List[str],
 selected_columns: List[str],
 context: str,
 category_summary: str,
 chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
 return _build_generic_prompt(
     question=question,
     items=items,
     mapping_keys=mapping_keys,
     selected_columns=selected_columns,
     context=context,
     category_summary=category_summary,
     comparison_type="Project",
     chat_history=chat_history
 )
