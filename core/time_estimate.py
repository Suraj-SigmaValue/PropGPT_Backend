"""
Estimated Time Calculator for PropGPT Queries

Estimates processing time based on query complexity factors.
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
    Estimate processing time in seconds based on query complexity.
    
    Factors:
    - Number of items (locations/cities/projects)
    - Number of years
    - Number of categories
    - Query complexity (multi-question detection)
    - Comparison type (Project queries are faster - no years)
    
    Returns:
        Estimated time in seconds (5-30 seconds typical range)
    """
    
    # Base time (seconds)
    base_time = 5  # Minimum processing time
    
    # Factor 1: Number of items (each item adds data retrieval time)
    item_count = len(items) if items else 1
    item_time = item_count * 1.5  # 1.5 seconds per item
    
    # Factor 2: Number of years (more years = more data to process)
    year_count = len(years) if years else 1
    year_time = year_count * 0.5  # 0.5 seconds per year
    
    # Factor 3: Number of categories (affects mapping key count)
    category_count = len(categories) if categories else 5  # Default to all
    category_time = category_count * 0.8  # 0.8 seconds per category
    
    # Factor 4: Query complexity (multi-question queries take longer)
    # Detect "AND" or multiple requests
    multi_question_indicators = [
        r'\band\b',  # "sales AND demographics"
        r'\balso\b',  # "also show me"
        r'\bplus\b',  # "sales plus demographics"
        r',',  # "sales, demographics, pricing"
    ]
    
    question_count = 1
    for pattern in multi_question_indicators:
        if re.search(pattern, query, re.IGNORECASE):
            question_count += query.lower().count(pattern.strip('\\b'))
    
    question_time = max(question_count - 1, 0) * 2  # 2 seconds per additional question
    
    # Factor 5: Comparison type (Project queries skip year processing)
    if comparison_type.lower() == "project":
        year_time = 0  # Projects don't use years
    
    # Factor 6: LLM validation overhead (relevance check iterations)
    validation_time = 3  # Average LLM validation time
    
    # Total estimated time
    total_time = base_time + item_time + year_time + category_time + question_time + validation_time
    
    # Cap between 5-30 seconds (reasonable range)
    total_time = max(5, min(30, total_time))
    
    # Round to nearest integer
    return int(round(total_time))


def get_time_estimate_message(seconds: int) -> str:
    """
    Convert seconds to user-friendly message.
    
    Args:
        seconds: Estimated time in seconds
        
    Returns:
        Friendly message like "About 10 seconds"
    """
    if seconds <= 5:
        return "About 5 seconds"
    elif seconds <= 10:
        return "About 10 seconds"
    elif seconds <= 15:
        return "About 15 seconds"
    elif seconds <= 20:
        return "About 20 seconds"
    elif seconds <= 30:
        return "About 30 seconds"
    else:
        return "About half a minute"


# Example calculations for reference:
# 
# Simple query (1 item, 2 years, 1 category):
#   5 + 1.5 + 1.0 + 0.8 + 0 + 3 = 11 seconds
# 
# Complex query (3 items, 5 years, 3 categories, multi-question):
#   5 + 4.5 + 2.5 + 2.4 + 2 + 3 = 19 seconds
# 
# Very complex (5 items, 5 years, all categories, multi-question):
#   5 + 7.5 + 2.5 + 4.0 + 4 + 3 = 26 seconds
