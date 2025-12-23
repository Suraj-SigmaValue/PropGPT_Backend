"""
Universal semantic validator for real estate queries.
No hardcoded terms - uses pattern matching and configuration.
"""
import re
from typing import Dict, List, Any, Tuple, Set  # IMPORTANT
import logging

logger = logging.getLogger(__name__)

class SemanticValidator:
    """
    Universal validator that uses patterns and configuration instead of hardcoded terms.
    """
    
    def __init__(self, column_patterns: Dict[str, List[str]] = None):
        """
        Initialize with semantic patterns.
        
        Args:
            column_patterns: Dict mapping semantic categories to regex patterns
                Example: {
                    "rate": [r"rate", r"per sqft", r"weighted average rate"],
                    "total_price": [r"agreement price", r"total sales"],
                    "demography": [r"pincode", r"age range", r"buyer"]
                }
        """
        self.column_patterns = column_patterns or self._get_default_patterns()
        self.query_patterns = self._get_query_patterns()
        
    def _get_default_patterns(self) -> Dict[str, List[str]]:
        """Get default column patterns."""
        return {
            "rate": [
                r"\brate\b", r"per sqft", r"per square foot", 
                r"price per", r"₹/sqft", r"rate per",
                r"weighted average rate", r"percentile rate",
                r"most prevailing rate"
            ],
            "total_price": [
                r"agreement price", r"total sales", r"sales value",
                r"total agreement", r"avg agreement price",
                r"total_sales"
            ],
            "demography": [
                r"pincode", r"pin code", r"age range", r"buyer",
                r"demographic", r"demography", r"age wise",
                r"top.*buyer", r"buyer.*pincode"
            ],
            "supply": [
                r"total units", r"available units", r"supplied",
                r"inventory", r"capacity", r"total unsold"
            ],
            "demand": [
                r"units sold", r"sold units", r"consumed",
                r"carpet area consumed", r"demand", r"absorbed",
                r"total sold", r"consumption"
            ],
            "area": [
                r"carpet area", r"sqft", r"square feet",
                r"area.*range", r"sq ft", r"area.*sqft"
            ],
        }
    
    def _get_query_patterns(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Get query patterns as (pattern, semantic_category) pairs.
        Returns conflicts to detect ambiguous queries.
        """
        return {
            "rate_analysis": [
                (r"\brate analysis\b", "rate"),
                (r"price per (?:sqft|square foot)", "rate"),
                (r"₹.*per.*sqft", "rate"),
                (r"rate per.*sqft", "rate"),
                (r"price.*square.*foot", "rate")
            ],
            "total_price_analysis": [
                (r"total (?:sales|price|value)", "total_price"),
                (r"agreement (?:price|value)", "total_price"),
                (r"sales value", "total_price"),
                (r"overall sales", "total_price")
            ],
            "demographic_analysis": [
                (r"demographic analysis", "demography"),
                (r"buyer.*profile", "demography"),
                (r"pincode analysis", "demography"),
                (r"age.*analysis", "demography"),
                (r"where.*buyers.*from", "demography")
            ],
            "supply_analysis": [
                (r"supply analysis", "supply"),
                (r"available units", "supply"),
                (r"total inventory", "supply"),
                (r"units available", "supply")
            ],
            "demand_analysis": [
                (r"demand analysis", "demand"),
                (r"units sold", "demand"),
                (r"sales volume", "demand"),
                (r"absorption rate", "demand")
            ]
        }
    
    def detect_query_categories(self, query: str) -> Dict[str, float]:
        """
        Detect semantic categories in query with confidence scores.
        
        Returns:
            Dict mapping category to confidence score (0-1)
        """
        query_lower = query.lower()
        categories = {}
        
        for category, patterns in self.query_patterns.items():
            matches = []
            for pattern, semantic_cat in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    matches.append(pattern)
            
            if matches:
                # Calculate confidence based on number and specificity of matches
                confidence = min(0.3 * len(matches) + 0.4, 1.0)
                categories[category] = {
                    "confidence": confidence,
                    "semantic_category": semantic_cat,
                    "matched_patterns": matches
                }
        
        return categories
    
    def get_columns_by_category(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Categorize columns based on patterns.
        
        Returns:
            Dict mapping semantic category to list of columns
        """
        categorized = {category: [] for category in self.column_patterns.keys()}
        categorized["uncategorized"] = []
        
        for column in columns:
            column_lower = column.lower()
            matched = False
            
            for category, patterns in self.column_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, column_lower, re.IGNORECASE):
                        categorized[category].append(column)
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                categorized["uncategorized"].append(column)
        
        return categorized
    
    def validate(self, query: str, selected_keys: List[str], selected_columns: List[str]) -> Dict[str, Any]:
        """
        
        Returns:
            Dict with validation results, warnings, and suggestions
        """
        query_lower = query.lower()
        
        # Step 1: Detect query categories
        query_categories = self.detect_query_categories(query)
        
        # Step 2: Categorize selected columns
        column_categories = self.get_columns_by_category(selected_columns)
        
        # Step 3: Validate matches
        validation = {
            "is_valid": True,
            "query_categories": query_categories,
            "column_categories": {k: v for k, v in column_categories.items() if v},
            "warnings": [],
            "errors": [],
            "suggestions": [],
            "conflicts": []
        }
        
        # Check for high-confidence query categories
        high_confidence_cats = {
            cat: info for cat, info in query_categories.items() 
            if info["confidence"] > 0.6
        }
        
        for query_cat, cat_info in high_confidence_cats.items():
            semantic_cat = cat_info["semantic_category"]
            
            # Check if we have columns for this semantic category
            if semantic_cat in column_categories and column_categories[semantic_cat]:
                validation["suggestions"].append(
                    f"✓ Query is {query_cat} and has appropriate {semantic_cat} columns"
                )
            else:
                validation["errors"].append(
                    f"✗ Query is {query_cat} (confidence: {cat_info['confidence']:.2f}) "
                    f"but no {semantic_cat} columns selected. "
                    f"Expected columns matching: {self.column_patterns.get(semantic_cat, [])[:3]}"
                )
                validation["is_valid"] = False
        
        # Check for column-category conflicts
        self._detect_conflicts(validation, column_categories)
        
        # Check mapping key coverage
        self._validate_key_coverage(validation, selected_keys, selected_columns)
        
        # Provide actionable suggestions
        self._generate_suggestions(validation, query_categories, column_categories)
        
        return validation
    
    def _detect_conflicts(self, validation: Dict, column_categories: Dict[str, List[str]]):
        """Detect conflicting column categories."""
        conflicts = []
        
        # Rate vs Total Price conflict
        if column_categories.get("rate") and column_categories.get("total_price"):
            conflicts.append({
                "type": "rate_vs_price_conflict",
                "message": "Columns contain both rate (price per sqft) and total price values. "
                          "These are different metrics - rate is ₹/sqft, total price is ₹.",
                "rate_columns": column_categories["rate"],
                "price_columns": column_categories["total_price"]
            })
        
        # Supply vs Demand conflict
        if column_categories.get("supply") and column_categories.get("demand"):
            conflicts.append({
                "type": "supply_vs_demand_conflict",
                "message": "Columns contain both supply (available) and demand (sold) metrics.",
                "supply_columns": column_categories["supply"],
                "demand_columns": column_categories["demand"]
            })
        
        if conflicts:
            validation["conflicts"] = conflicts
            validation["warnings"].append("Potential metric conflicts detected")
    
    def _validate_key_coverage(self, validation: Dict, selected_keys: List[str], selected_columns: List[str]):
        """Validate that selected keys have corresponding columns."""
        if not selected_keys:
            return
            
        key_coverage = {}
        for key in selected_keys:
            # Simple word matching for keys (could be enhanced)
            key_words = set(re.findall(r'\b\w+\b', key.lower()))
            matching_cols = []
            
            for col in selected_columns:
                col_words = set(re.findall(r'\b\w+\b', col.lower()))
                # Check if key words appear in column name
                if key_words.intersection(col_words):
                    matching_cols.append(col)
            
            key_coverage[key] = {
                "matching_columns": matching_cols,
                "coverage_score": len(matching_cols) / max(len(selected_columns), 1)
            }
        
        validation["key_coverage"] = key_coverage
        
        # Check for keys with no columns
        empty_keys = [k for k, v in key_coverage.items() if not v["matching_columns"]]
        if empty_keys:
            validation["warnings"].append(
                f"Mapping keys without columns: {empty_keys}"
            )
    
    def _generate_suggestions(self, validation: Dict, query_categories: Dict, column_categories: Dict):
        """Generate actionable suggestions based on validation."""
        suggestions = []
        
        # If no high-confidence query categories, suggest based on column patterns
        if not any(info["confidence"] > 0.6 for info in query_categories.values()):
            dominant_col_cats = [
                (cat, cols) for cat, cols in column_categories.items() 
                if cols and cat != "uncategorized"
            ]
            
            if dominant_col_cats:
                dominant_cat, dominant_cols = max(dominant_col_cats, key=lambda x: len(x[1]))
                suggestions.append(
                    f"Query category unclear. Columns suggest {dominant_cat} analysis. "
                    f"If this is incorrect, refine your query to specify: "
                    f"rate, price, demography, supply, or demand analysis."
                )
        
        # Suggest missing categories
        for col_cat, columns in column_categories.items():
            if not columns and col_cat != "uncategorized":
                # This category has no columns but might be needed
                suggestions.append(
                    f"No {col_cat} columns selected. If analyzing {col_cat}, "
                    f"include columns matching: {self.column_patterns.get(col_cat, [])[:2]}"
                )
        
        if suggestions:
            validation["suggestions"].extend(suggestions)


# Global validator instance
validator = SemanticValidator()