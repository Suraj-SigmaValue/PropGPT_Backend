# -*- coding: utf-8 -*-

"""
source_utils.py

Utility functions for handling data source information
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def get_columns_with_sources(selected_columns: List[str], comparison_type: str) -> List[Dict[str, str]]:
    """
    Enrich selected columns with their data source information.
    
    CRITICAL: Returns ORIGINAL column names from COLUMN_MAPPING (not normalized)
    for display purposes, e.g., "flat_sold - igr" instead of "flat_sold igr"
    
    Args:
        selected_columns: List of normalized column names selected for the query
        comparison_type: Type of comparison (Location/City/Project)
    
    Returns:
        List of dicts with 'column' (ORIGINAL format) and 'source' keys
        Example: [
            {'column': 'flat_sold - igr', 'source': 'IGR-CGDB'},
            {'column': 'shop_sold - igr', 'source': 'IGR-CGDB'}
        ]
    """
    from .mapping import SOURCE_MAPPING_Location, SOURCE_MAPPING_City, SOURCE_MAPPING_Project
    from .config import get_column_mapping
    from .core_utils import normalize_colname
    
    # Select the appropriate SOURCE_MAPPING based on comparison_type
    if comparison_type.lower() == 'location':
        source_mapping = SOURCE_MAPPING_Location
    elif comparison_type.lower() == 'city':
        source_mapping = SOURCE_MAPPING_City
    elif comparison_type.lower() == 'project':
        source_mapping = SOURCE_MAPPING_Project
    else:
        logger.warning(f"Unknown comparison_type: {comparison_type}, defaulting to Location")
        source_mapping = SOURCE_MAPPING_Location
    
    # Get column mapping to find which mapping key each column belongs to
    column_mapping = get_column_mapping(comparison_type)
    
    # CRITICAL: Create TWO reverse mappings:
    # 1. normalized_column -> mapping_key (for source lookup)
    # 2. normalized_column -> original_column_name (for display)
    column_to_key = {}
    normalized_to_original = {}
    
    for mapping_key, columns in column_mapping.items():
        for col in columns:
            # Normalize for lookup
            normalized_col = normalize_colname(str(col))
            
            # Map normalized -> mapping_key (for source lookup)
            column_to_key[normalized_col] = mapping_key
            
            # Map normalized -> original (for display)
            # Keep the FIRST occurrence if there are duplicates
            if normalized_col not in normalized_to_original:
                normalized_to_original[normalized_col] = str(col)
    
    # Build the result list
    columns_with_sources = []
    for column in selected_columns:
        # Normalize for lookup
        column_normalized = normalize_colname(str(column))
        
        # Get the ORIGINAL column name from COLUMN_MAPPING (not normalized)
        original_column_name = normalized_to_original.get(column_normalized, column)
        
        # Find the mapping key for this column
        mapping_key = column_to_key.get(column_normalized)
        
        if mapping_key:
            # Get the source for this mapping key
            source = source_mapping.get(mapping_key, "Unknown")
            logger.debug(f"Column '{original_column_name}' (normalized: '{column_normalized}') -> Key '{mapping_key}' -> Source '{source}'")
        else:
            source = "Unknown"
            logger.warning(f"Column '{column}' (normalized: '{column_normalized}') not found in column_to_key mapping")
        
        # CRITICAL: Use ORIGINAL column name for display
        columns_with_sources.append({
            'column': original_column_name,  # Original format from COLUMN_MAPPING
            'source': source
        })
    
    logger.info(f"Enriched {len(columns_with_sources)} columns with source information (using original column names)")
    return columns_with_sources


def format_data_sources_text(columns_with_sources: List[Dict[str, str]]) -> str:
    """
    Format columns with sources as readable text for display.
    
    Args:
        columns_with_sources: List of dicts with 'column' and 'source' keys
    
    Returns:
        Formatted string like:
        "Data Sources:
         - flat age range wise total sales: RERA
         - shop age range wise total sales: IGR"
    """
    if not columns_with_sources:
        return "Data Sources: No data sources available"
    
    lines = ["Data Sources:"]
    for item in columns_with_sources:
        lines.append(f"- {item['column']}: {item['source']}")
    
    return "\n".join(lines)
