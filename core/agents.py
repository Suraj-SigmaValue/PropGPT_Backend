"""
agents.py

Query Intelligence Agents for PropGPT.

Goal
- Pick the most relevant "mapping keys" from mapping.py based on user query.
- Then pick the most relevant dataframe columns required to answer the query.

Design principles (per your ask)
- No hard-coded mapping-key names.
- No hard-coded category names.
- Works purely from what exists inside mapping.py (keys + their mapped column names).
- Deterministic ranking first (token / similarity / coverage), optional LLM rerank as a final tie-breaker.
- Clean, testable, and predictable.

How to use
1) Pass the correct mapping dict (e.g., COLUMN_MAPPING_City) as `mapping_dict`
2) Candidate keys are simply `list(mapping_dict.keys())`
3) Columns are selected using the chosen keys → mapping_dict[key] gives the canonical columns
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# -----------------------------
# Text utilities
# -----------------------------
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _tokens(s: str) -> List[str]:
    return _WORD_RE.findall(_norm(s))

def _unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))

def _char_ngrams(s: str, n: int = 3) -> List[str]:
    s = re.sub(r"\s+", " ", _norm(s))
    s = f" {s} "
    if len(s) < n:
        return [s]
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def _cosine_sparse(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    dot = 0.0
    for k, w in v1.items():
        dot += w * v2.get(k, 0.0)
    n1 = math.sqrt(sum(w*w for w in v1.values()))
    n2 = math.sqrt(sum(w*w for w in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


# -----------------------------
# Scoring model (no key hardcoding)
# -----------------------------
@dataclass(frozen=True)
class KeyProfile:
    key: str
    key_tokens: Tuple[str, ...]
    col_tokens: Tuple[str, ...]
    key_ngrams: Tuple[str, ...]
    col_ngrams: Tuple[str, ...]



# -----------------------------
# Category Synonyms
# -----------------------------
CATEGORY_KEYWORDS = {
    "demography": ["user", "users", "people", "buyer", "buyers", "origin", "location", "coming", "from", "where", "profile", "Demographic"],
    # Add other categories here if needed
}


def _build_profiles(mapping_dict: Dict[str, List[str]], category_mapping: Optional[Dict[str, List[str]]] = None) -> List[KeyProfile]:
    # Inverse map: key -> list of categories it belongs to
    key_to_cats = {}
    if category_mapping:
         for cat, keys in category_mapping.items():
            for k in keys:
                if k not in key_to_cats:
                    key_to_cats[k] = []
                key_to_cats[k].append(cat)

    profiles: List[KeyProfile] = []
    for k, cols in mapping_dict.items():
        # Enrich BOTH key and column profiles with category info + synonyms
        extra_tokens = []
        if k in key_to_cats:
            for cat in key_to_cats[k]:
                # 1. Add category name itself (e.g. "demography")
                extra_tokens.extend(_tokens(cat))
                
                # 2. Add synonyms (e.g. "user", "location" for demography)
                if cat in CATEGORY_KEYWORDS:
                     extra_tokens.extend(CATEGORY_KEYWORDS[cat])

        # Feature: Map "New Launch" -> Development Agreement (DA) keys
        # If the key contains "Development Agreement" or "(DA)" (handle typo "developement" too), matching "New Launch Project" queries
        k_lower = k.lower()
        if "development agreement" in k_lower or "developement agreement" in k_lower or "(da)" in k_lower:
            extra_tokens.extend(["new", "launch", "project"])

        # Feature: Enrich "Property Type" keys with specific types (office, flat, shop, etc)
        # This ensuring queries like "office sales" match "Property type wise..." keys instead of just "Total sales"
        if "property type" in k_lower:
            extra_tokens.extend(["flat", "office", "shop", "commercial", "residential", "apartment", "others"])

        # Key Tokens
        
        # Key Tokens
        kt_base = _tokens(k)
        kt = tuple(_unique(list(kt_base) + extra_tokens))

        # Col Tokens
        col_token_iter = (t for c in (cols or []) for t in _tokens(c))
        ct = tuple(_unique(list(col_token_iter) + extra_tokens))

        # Key Ngrams (include category name for trigram matching i.e "demographic" matching "demography")
        kng_base = _char_ngrams(k, 3)
        extra_ngrams = []
        for t in extra_tokens:
             extra_ngrams.extend(_char_ngrams(t, 3))
             
        kng = tuple(_unique(list(kng_base) + extra_ngrams))
        
        # Col Ngrams
        cng = tuple(_unique(ng for c in (cols or []) for ng in _char_ngrams(c, 3)))
        
        profiles.append(KeyProfile(
            key=k,
            key_tokens=kt,
            col_tokens=ct,
            key_ngrams=kng,
            col_ngrams=cng
        ))
    return profiles


def _idf_weights(docs: List[List[str]]) -> Dict[str, float]:
    """
    Basic IDF weighting over tokens (built from mapping keys + mapped columns).
    """
    N = len(docs)
    df: Dict[str, int] = {}
    for doc in docs:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
    idf: Dict[str, float] = {}
    for t, d in df.items():
        # smooth
        idf[t] = math.log((N + 1) / (d + 1)) + 1.0
    return idf


def _tfidf_vector(tokens: Sequence[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    out: Dict[str, float] = {}
    for t, f in tf.items():
        out[t] = (1.0 + math.log(f)) * idf.get(t, 1.0)
    return out


def _score_key(query: str, prof: KeyProfile, idf_tok: Dict[str, float], idf_ng: Dict[str, float]) -> float:
    """
    Score uses ONLY:
    - similarity between query and key text
    - similarity between query and mapped columns text
    - coverage bonus for matching both key and columns (robustness)
    """
    q_tok = _tokens(query)
    q_ng = _char_ngrams(query, 3)

    # Token TF-IDF cosine
    qv_tok = _tfidf_vector(q_tok, idf_tok)
    kv_tok = _tfidf_vector(list(prof.key_tokens), idf_tok)
    cv_tok = _tfidf_vector(list(prof.col_tokens), idf_tok)
    cos_key_tok = _cosine_sparse(qv_tok, kv_tok)
    cos_col_tok = _cosine_sparse(qv_tok, cv_tok)

    # Character trigram TF-IDF cosine (handles typos like "yer pincode")
    qv_ng = _tfidf_vector(q_ng, idf_ng)
    kv_ng = _tfidf_vector(list(prof.key_ngrams), idf_ng)
    cv_ng = _tfidf_vector(list(prof.col_ngrams), idf_ng)
    cos_key_ng = _cosine_sparse(qv_ng, kv_ng)
    cos_col_ng = _cosine_sparse(qv_ng, cv_ng)

    # Jaccard overlap (cheap but stabilizes ranking on short queries)
    jac_key = _jaccard(q_tok, prof.key_tokens)
    jac_col = _jaccard(q_tok, prof.col_tokens)

    # Coverage bonus: query overlaps both key-side and column-side signals
    overlap_key = len(set(q_tok) & set(prof.key_tokens))
    overlap_col = len(set(q_tok) & set(prof.col_tokens))
    coverage = 0.0
    if overlap_key > 0 and overlap_col > 0:
        coverage = 0.10
    elif overlap_key > 0 or overlap_col > 0:
        coverage = 0.05

    # Weighted blend (no hard-coded key names; only generic similarity)
    score = (
        0.35 * cos_key_tok +
        0.35 * cos_col_tok +
        0.10 * cos_key_ng +
        0.10 * cos_col_ng +
        0.05 * jac_key +
        0.05 * jac_col +
        coverage
    )
    return float(score)


def _safe_json_list(text: str) -> Optional[List[Any]]:
    if not text:
        return None
    s, e = text.find("["), text.rfind("]") + 1
    if s == -1 or e <= 0:
        return None
    try:
        parsed = json.loads(text[s:e])
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


# -----------------------------
# Planner Agent (keys)
# -----------------------------
def planner_identify_mapping_keys(
    llm,
    query: str,
    candidate_keys: List[str],
    mapping_dict: Optional[Dict[str, List[str]]] = None,
    category_mapping: Optional[Dict[str, List[str]]] = None,
    max_keys: int = 6,
    use_llm_rerank: bool = True
) -> List[str]:
    """
    Select the most relevant mapping keys for the query.

    IMPORTANT:
    - To be "100% from mapping.py", pass mapping_dict as the actual COLUMN_MAPPING_* dict.
    - If mapping_dict is not provided, we will do best-effort lexical scoring only on the keys.

    Returns:
      List[str] of selected keys, up to max_keys.
    """
    if not candidate_keys:
        return []

    # If mapping_dict not passed, fallback to a light key-only scoring
    if not mapping_dict:
        qtok = set(_tokens(query))
        scored = []
        for k in candidate_keys:
            ktok = set(_tokens(k))
            # Mix token overlap + trigram overlap to survive typos
            tok_score = len(qtok & ktok) / float(len(qtok | ktok) or 1)
            ng_score = _jaccard(_char_ngrams(query, 3), _char_ngrams(k, 3))
            scored.append((0.7 * tok_score + 0.3 * ng_score, k))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [k for _, k in scored[:max_keys]]

    # Deterministic rank using profiles built from mapping_dict
    profiles = _build_profiles(mapping_dict, category_mapping)

    # IDF built from both key tokens and column tokens (purely from mapping.py)
    docs_tok = [list(p.key_tokens) + list(p.col_tokens) for p in profiles]
    docs_ng  = [list(p.key_ngrams) + list(p.col_ngrams) for p in profiles]
    idf_tok = _idf_weights(docs_tok)
    idf_ng = _idf_weights(docs_ng)

    candidates_set = set(candidate_keys)
    scored: List[Tuple[float, str]] = []
    for p in profiles:
        if p.key not in candidates_set:
            continue
        s = _score_key(query, p, idf_tok, idf_ng)
        scored.append((s, p.key))

    scored.sort(reverse=True, key=lambda x: x[0])
    deterministic = [k for _, k in scored[:max_keys]]

    # Optional LLM rerank ONLY among top shortlist (keeps it stable + accurate)
    if not llm or not use_llm_rerank or len(deterministic) <= 1:
        return deterministic

    try:
        sys_instr = (
            "You are a mapping-key selector. Choose the best keys to answer the user query. "
            "Return ONLY a JSON array of keys chosen from SHORTLIST. No extra text."
        )
        prompt = f"""
User Query:
{query}

SHORTLIST (choose up to {max_keys}, only from this list):
{json.dumps(deterministic, indent=2)}

Context: Each key maps to one or more canonical dataset columns.
Rule: Pick only keys that directly help answer the query.
Output: JSON list only.
"""
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        picked = _safe_json_list(raw)
        if not picked:
            return deterministic
        picked = [k for k in picked if k in deterministic]
        picked = list(dict.fromkeys(picked))
        return picked[:max_keys] if picked else deterministic
    except Exception as exc:
        logger.warning("[planner_identify_mapping_keys] LLM rerank failed: %s", exc)
        return deterministic


# -----------------------------
# Column Agent (columns)
# -----------------------------
def agent_pick_relevant_columns(
    llm,
    query: str,
    selected_keys: List[str],
    candidate_columns: List[str],
    mapping_dict: Optional[Dict[str, List[str]]] = None,
    max_cols: int = 20,
    use_llm_rerank: bool = True
) -> List[str]:
    """
    Select the most relevant dataframe columns required to answer the query.

    If mapping_dict is provided:
      - We always include at least one mapped column per selected key (where possible),
        but we do not hardcode anything: we just read mapping_dict[key] columns.

    Then we rank remaining columns by similarity to query, and optionally let LLM rerank.
    """
    if not candidate_columns or not selected_keys:
        return []

    cand_set = set(candidate_columns)

    # 1) Start from canonical columns implied by selected keys (purely from mapping.py)
    chosen: List[str] = []
    if mapping_dict:
        for k in selected_keys:
            cols = mapping_dict.get(k, []) or []
            # keep only those actually present
            present = [c for c in cols if c in cand_set]
            # ensure at least one per key if available
            if present:
                # keep the most "descriptive": longest column name first (heuristic, key-agnostic)
                present.sort(key=lambda x: len(x), reverse=True)
                chosen.extend(present[: min(3, len(present))])  # cap per key to avoid bloat

    chosen = list(dict.fromkeys(chosen))

    # 2) Add extra relevant columns via similarity scoring (still key-agnostic)
    remaining = [c for c in candidate_columns if c not in set(chosen)]
    if remaining:
        # Build IDF over candidate columns text
        docs = [_tokens(c) for c in candidate_columns]
        idf = _idf_weights(docs)
        qv = _tfidf_vector(_tokens(query), idf)

        scored: List[Tuple[float, str]] = []
        for c in remaining:
            cv = _tfidf_vector(_tokens(c), idf)
            cos = _cosine_sparse(qv, cv)
            ng = _jaccard(_char_ngrams(query, 3), _char_ngrams(c, 3))
            jac = _jaccard(_tokens(query), _tokens(c))
            s = 0.70 * cos + 0.20 * ng + 0.10 * jac
            scored.append((s, c))

        scored.sort(reverse=True, key=lambda x: x[0])
        # fill up to max_cols
        for _, c in scored:
            if len(chosen) >= max_cols:
                break
            chosen.append(c)

    chosen = chosen[:max_cols]

    # 3) Optional LLM rerank (kept small + safe)
    if not llm or not use_llm_rerank or len(chosen) <= 3:
        return chosen

    try:
        sys_instr = (
            "You select strictly relevant dataframe column names for the query. "
            "Return ONLY a JSON list of exact column names from SHORTLIST. No extra text."
        )
        prompt = f"""
User Query:
{query}

Selected mapping keys (context):
{json.dumps(selected_keys, indent=2)}

SHORTLIST columns (pick what is needed, keep it small):
{json.dumps(chosen, indent=2)}

Rules:
- Output ONLY a JSON list.
- Keep only columns needed to answer the query.
"""
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        picked = _safe_json_list(raw)
        if not picked:
            return chosen
        picked = [c for c in picked if c in set(chosen)]
        picked = list(dict.fromkeys(picked))
        return picked[:max_cols] if picked else chosen
    except Exception as exc:
        logger.warning("[agent_pick_relevant_columns] LLM rerank failed: %s", exc)
        return chosen


# -----------------------------
# Correction Agent (keys)
# -----------------------------
def agent_correction_mapping(
    llm,
    query: str,
    old_keys: List[str],
    candidate_keys: List[str],
    mapping_dict: Optional[Dict[str, List[str]]] = None,
    max_keys: int = 6
) -> List[str]:
    """
    If user thumbs-down the mapping keys, propose a corrected set.
    No hardcoded key names; we just re-rank and then let LLM pick among alternatives.
    """
    if not candidate_keys:
        return []

    # Start with deterministic re-rank (try to move away from old keys)
    new_keys = planner_identify_mapping_keys(
        llm=None,
        query=query,
        candidate_keys=candidate_keys,
        mapping_dict=mapping_dict,
        max_keys=max_keys * 2,          # broader set for correction
        use_llm_rerank=False
    )

    # Remove old keys if possible (diversify)
    diversified = [k for k in new_keys if k not in set(old_keys)]
    shortlist = diversified[: max_keys * 2] if diversified else new_keys[: max_keys * 2]

    if not llm:
        return shortlist[:max_keys]

    try:
        sys_instr = (
            "You are a correction assistant for mapping keys. The previous keys were rejected. "
            "Choose a better set from SHORTLIST. Return ONLY a JSON list."
        )
        prompt = f"""
User query:
{query}

Rejected keys:
{json.dumps(old_keys, indent=2)}

SHORTLIST (pick up to {max_keys} keys from here only):
{json.dumps(shortlist, indent=2)}

Output: JSON list only.
"""
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        picked = _safe_json_list(raw)
        if not picked:
            return shortlist[:max_keys]
        picked = [k for k in picked if k in set(shortlist)]
        picked = list(dict.fromkeys(picked))
        return picked[:max_keys] if picked else shortlist[:max_keys]
    except Exception as exc:
        logger.warning("[agent_correction_mapping] failed: %s", exc)
        return shortlist[:max_keys]


# -----------------------------
# Validation
# -----------------------------
def validate_selected_columns(
    query: str,
    selected_keys: List[str],
    selected_columns: List[str],
    mapping_dict: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Validation without hardcoding key names:
    - Keys must exist
    - Columns must exist
    - If mapping_dict provided: each selected key should have at least one mapped column present (best-effort)
    """
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

    if not selected_keys:
        errors.append("No mapping keys selected.")
    if not selected_columns:
        errors.append("No columns selected.")

    if mapping_dict and selected_keys and selected_columns:
        col_set = set(selected_columns)
        missing_cover = []
        for k in selected_keys:
            mapped = mapping_dict.get(k, []) or []
            # if none of the mapped columns appear, it might be a mismatch between mapping_dict vs df columns
            if mapped and not any(c in col_set for c in mapped):
                missing_cover.append(k)
        if missing_cover:
            warnings.append("Some selected keys have no mapped columns present in selected_columns.")
            suggestions.append(f"Check DF column availability or mapping_dict for keys: {missing_cover}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "query_categories": {}
    }
    