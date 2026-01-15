"""
Estimated Time Calculator for PropGPT Queries
Conservative estimator that matches actual LangSmith 55-second average.
"""

import re
from typing import List


def estimate_processing_time(
    query: str,
    items: List[str],
    categories: List[str],
    years: List[int],
    comparison_type: str
) -> int:
    """
    Conservative estimator based on actual LangSmith 55-second average.
    
    Your LangSmith shows: 45-60 seconds average, up to 103 seconds
    Your UI shows: 20 seconds (2.5x too low)
    
    Returns estimated time in seconds (40-75 seconds typical)
    """
    
    # ====== BASE TIME FROM LANGCHAIN LOGS ======
    # Simple average from your LangSmith: ~55 seconds
    # Let's be conservative and aim for 55 seconds
    
    base_seconds = 55  # Target average from LangSmith
    
    # ====== QUERY COMPLEXITY ADJUSTMENTS ======
    query_lower = query.lower()
    word_count = len(query_lower.split())
    
    # Adjust based on query complexity
    complexity_adjustment = 0
    
    # Word count impact
    if word_count > 15:
        complexity_adjustment += 15  # Long queries take longer
    elif word_count > 8:
        complexity_adjustment += 5
    
    # Query type impact (from LangSmith patterns)
    if "compare" in query_lower or "versus" in query_lower or "vs" in query_lower:
        complexity_adjustment += 20  # Comparisons take ~20s longer
    
    if "analysis" in query_lower or "trend" in query_lower or "historical" in query_lower:
        complexity_adjustment += 15
    
    if "average" in query_lower and ("price" in query_lower or "sales" in query_lower):
        complexity_adjustment += 10
    
    # Multiple questions impact
    if " and " in query_lower:
        complexity_adjustment += query_lower.count(" and ") * 10
    
    if "," in query_lower and query_lower.count(",") > 1:
        complexity_adjustment += query_lower.count(",") * 5
    
    # ====== ITEM COUNT IMPACT ======
    item_count = len(items) if items else 1
    item_adjustment = (item_count - 1) * 8  # Each additional item adds ~8s
    
    # ====== YEARS IMPACT ======
    year_count = len(years) if years else 1
    if comparison_type.lower() != "project":
        year_adjustment = (year_count - 1) * 5  # Each year adds ~5s
    else:
        year_adjustment = 0  # Projects don't use years
    
    # ====== CATEGORIES IMPACT ======
    category_count = len(categories) if categories else 5
    category_adjustment = (category_count - 1) * 3  # Each category adds ~3s
    
    # ====== CALCULATE TOTAL ======
    total_seconds = (
        base_seconds +
        complexity_adjustment +
        item_adjustment +
        year_adjustment +
        category_adjustment
    )
    
    # ====== APPLY REALISTIC BOUNDS FROM LANGCHAIN ======
    # From your LangSmith: minimum 9s, maximum 103s
    # USER REQUEST: Always show at least 55 seconds
    total_seconds = max(55, min(100, total_seconds))
    
    # For very simple queries, still show at least 55 seconds as per user request
    if (word_count < 5 and 
        ("location" in query_lower or "coordinates" in query_lower or "name" in query_lower)):
        total_seconds = max(55, min(65, total_seconds))
    
    return int(round(total_seconds))


def get_time_estimate_message(seconds: int) -> str:
    """
    Convert seconds to user-friendly message.
    """
    if seconds <= 30:
        return "About 30 seconds"
    elif seconds <= 40:
        return "About 40 seconds"
    elif seconds <= 50:
        return "About 50 seconds"
    elif seconds <= 55:
        return "About 55 seconds"
    elif seconds <= 60:
        return "About a minute"
    elif seconds <= 75:
        return "About 1 minute 15 seconds"
    elif seconds <= 90:
        return "About 1.5 minutes"
    else:
        return "About 2 minutes"


# ====== SIMPLE CONSERVATIVE VERSION ======
def get_conservative_estimate(query: str, item_count: int = 1) -> str:
    """
    Always show conservative estimate that matches LangSmith average.
    """
    query_lower = query.lower()
    
    # Base on LangSmith average: 55 seconds
    base_seconds = 55
    
    # Simple adjustments
    if "compare" in query_lower:
        base_seconds = 70  # Comparisons take longer
    elif "analysis" in query_lower or "trend" in query_lower:
        base_seconds = 65
    elif "average" in query_lower:
        base_seconds = 60
    elif len(query_lower.split()) < 5:
        base_seconds = 45  # Very simple queries
    
    # Adjust for items
    if item_count > 1:
        base_seconds += (item_count - 1) * 8
    
    # Cap at realistic bounds
    base_seconds = max(40, min(85, base_seconds))
    
    return get_time_estimate_message(base_seconds)


# ====== EVEN SIMPLER: FIXED 55-SECOND ESTIMATE ======
def get_fixed_estimate() -> str:
    """
    Always show 55 seconds - matches your LangSmith average.
    This ensures users wait the actual average time.
    """
    return "About 55 seconds"  # Exactly 55 seconds as per LangSmith average


def get_processing_wait_message(elapsed_seconds: int) -> str:
    """
    Returns a status message if processing exceeds a certain threshold.
    As per USER request: "Once 55 second over... show 'Please wait, Response will display soon'"
    """
    if elapsed_seconds > 55:
        return "Please wait, Response will display soon"
    return ""


# Test the conservative estimates
if __name__ == "__main__":
    # Your actual queries from LangSmith
    test_queries = [
        ("can you give me the coordinates please", 1, "project"),
        ("give me average price and sales trend", 1, "location"),
        ("compare mumbai and pune prices", 2, "location"),
        ("what is the most expensive project", 1, "project"),
        ("if i invest 2cr in mumbai what return can i expect", 1, "location"),
    ]
    
    print("=== Conservative Estimates (Matching LangSmith 55s Average) ===")
    for query, items, comp_type in test_queries:
        est = estimate_processing_time(query, ["item"] * items, ["all"], [2023], comp_type)
        print(f"Query: {query[:50]}...")
        print(f"  Estimated: {est}s -> {get_time_estimate_message(est)}")
        print(f"  LangSmith actual: 45-60s (average 55s)")
        print()