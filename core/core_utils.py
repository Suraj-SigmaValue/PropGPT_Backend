"""
Core utilities extracted from c_app.py for Django backend.
All functions preserved exactly as-is, only Streamlit-specific decorators removed.
"""

import os
import json
import logging
import re
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Union
import textwrap
import ast
import pandas as pd
import tiktoken
from dotenv import load_dotenv

os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"

from fuzzywuzzy import process
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
import joblib

from .config import (
    EXCEL_FILE,
    PICKLE_FILE,
    SHEET_CONFIG,
    get_category_mapping,
    get_column_mapping
)
from .agents import planner_identify_mapping_keys, agent_pick_relevant_columns, agent_correction_mapping
from .prompts import build_location_prompt, build_city_prompt, build_project_prompt
from .response_cache import SemanticResponseCache

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# GLOBAL DATA CACHE (Removed to avoid hanging issues with large datasets)
# _GLOBAL_DATA_CACHE = None
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize global mappings
CATEGORY_MAPPING = None
COLUMN_MAPPING = None
RESOLVED_ID_COLS = {}


def load_mappings(comparison_type: str):
    """Return (category_mapping, column_mapping) for a comparison type and cache the result."""
    cat_map = get_category_mapping(comparison_type)
    col_map = get_column_mapping(comparison_type)
    try:
        normalized_col_map = {}
        for key, cols in (col_map or {}).items():
            normalized_cols = [normalize_colname(str(c)) for c in cols]
            normalized_col_map[key] = normalized_cols
    except Exception:
        normalized_col_map = col_map
    return cat_map, normalized_col_map


# In core/core_utils.py, add if missing:
def get_graph_app():
    """Get the graph application instance."""
    from .graph_agent import create_graph
    return create_graph()


def set_mappings_for_type(comparison_type: str) -> None:
    """Set the global mappings based on comparison type (uses cached loader)."""
    global CATEGORY_MAPPING, COLUMN_MAPPING
    cat_map, col_map = load_mappings(comparison_type)
    CATEGORY_MAPPING = cat_map
    COLUMN_MAPPING = col_map
    

def get_category_keys(category: str) -> List[str]:
    """Return mapping keys associated with a category."""
    if CATEGORY_MAPPING is None:
        raise RuntimeError("CATEGORY_MAPPING not initialized. Call set_mappings_for_type first.")
    return CATEGORY_MAPPING.get(category.lower(), [])


def get_columns_for_keys(mapping_keys: List[str]) -> Dict[str, List[str]]:
    """Return a dict of mapping_key -> column names filtered by keys."""
    if COLUMN_MAPPING is None:
        raise RuntimeError("COLUMN_MAPPING not initialized. Call set_mappings_for_type first.")
    columns_by_key: Dict[str, List[str]] = {}
    for key in mapping_keys:
        cols = COLUMN_MAPPING.get(key)
        if not cols:
            logger.warning("Mapping key '%s' missing in COLUMN_MAPPING", key)
            continue
        columns_by_key[key] = cols
    return columns_by_key


def get_filtered_dataframe(comparison_type: str, base_dir: Path):
    """Get cached dataframe filtered by comparison type"""
    pickle_path = base_dir / PICKLE_FILE
    
    if not pickle_path.exists():
        df_all = initialize_dataframe(base_dir)
    else:
        df_all = joblib.load(pickle_path)
        if df_all is not None and not df_all.empty:
            df_all.columns = [normalize_colname(str(c)) for c in df_all.columns]
        
    if df_all is None or df_all.empty:
        return None
        
    # Filter for comparison type
    return df_all[df_all["__type"] == comparison_type].copy()


def get_embeddings():
    from .config import get_embeddings as get_global_embeddings
    return get_global_embeddings()


def get_response_cache(embeddings, cache_dir: Path):
    """Initialize semantic response cache."""
    return SemanticResponseCache(
        cache_dir=cache_dir,
        embeddings=embeddings,
        similarity_threshold=0.95,
        ttl_seconds=86400
    )


def flatten_columns(columns_by_key: Dict[str, List[str]]) -> List[str]:
    """Flatten dict of key->columns into a unique column list preserving order."""
    ordered: List[str] = []
    seen = set()
    for cols in columns_by_key.values():
        for col in cols:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
    return ordered


def normalize_colname(name):
    name = re.sub(r'[-\s]+', ' ', name.strip().lower())
    name = re.sub(r'\(in\s+sqft\)', '(in sqft)', name)
    return name


def initialize_dataframe(base_dir: Path):
    """Initialize combined dataframe from Excel"""
    excel_path = base_dir / EXCEL_FILE
    pickle_path = base_dir / PICKLE_FILE
    
    try:
        if pickle_path.exists():
            os.remove(pickle_path)
            logger.info("Refreshing data from Excel file...")
        
        if not excel_path.exists():
            logger.error(f"Excel file not found: {excel_path}")
            return None
        
        logger.info(f"Loading data from {excel_path}...")
        dfs = pd.read_excel(excel_path, sheet_name=None)
        logger.info(f"Excel file loaded successfully")
        
        combined = []
        for ctype, cfg in SHEET_CONFIG.items():
            if cfg["sheet"] in dfs:
                df = dfs[cfg["sheet"]].copy()
                df.columns = [normalize_colname(str(c)) for c in df.columns]
                df["__type"] = ctype
                combined.append(df)
        
        if not combined:
            logger.error("No valid sheets found in Excel file!")
            return None
        
        df_all = pd.concat(combined, ignore_index=True)
        joblib.dump(df_all, pickle_path)
        
        # Invalidate cache if data is re-initialized (Cache removed)
        # global _GLOBAL_DATA_CACHE
        # _GLOBAL_DATA_CACHE = df_all
        
        return df_all
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.exception("Data loading error")
        return None


def load_and_clean_data(excel_path, pickle_path, comparison_type, items=None, years=None, category=None):
    try:
        # Load directly from Disk (Pickle) to avoid RAM hanging issues
        if Path(pickle_path).exists():
            df = joblib.load(pickle_path)
            df.columns = [normalize_colname(str(c)) for c in df.columns]
            logger.info(f"Pickle file loaded from DISK. Shape: {df.shape}")
        else:
            logger.error(f"Pickle file not found at {pickle_path}")
            return None, None, None
        
        # Filter by comparison type
        df = df[df["__type"] == comparison_type].drop(columns=["__type"])

        # Resolve ID column
        configured_id = SHEET_CONFIG[comparison_type]["id_col"]
        id_col = configured_id
        if id_col not in df.columns:
            cols_norm = {normalize_colname(str(c)): c for c in df.columns}
            if id_col in cols_norm:
                matched_col = cols_norm[id_col]
                logger.warning("ID column '%s' not present; using normalized match '%s'", id_col, matched_col)
                id_col = matched_col
            else:
                try:
                    choices = list(cols_norm.keys())
                    best, score = process.extractOne(id_col, choices)
                    if score >= 70:
                        matched_col = cols_norm[best]
                        logger.warning("ID column '%s' not found; fuzzy-matched to '%s' (score=%s)", id_col, matched_col, score)
                        id_col = matched_col
                    else:
                        logger.error("ID column '%s' not found for type '%s'. Available columns: %s", id_col, comparison_type, df.columns.tolist())
                        logger.debug("Expected id_col: %s | Normalized columns: %s", id_col, list(cols_norm.keys()))
                        return None, None, None
                except Exception as exc:
                    logger.exception("Error during id_col fuzzy matching: %s", exc)
                    return None, None, None

        # Clean and normalize ID column values
        try:
            df[id_col] = df[id_col].astype(str).str.strip().str.lower()
        except Exception:
            df[id_col] = df[id_col].astype(str)

        available_items = df[id_col].unique()
        logger.info(f"Available {comparison_type}s (sample): {list(available_items)[:20]}")

        if items:
            lowered = [i.lower() for i in items]
            df = df[df[id_col].isin(lowered)]
            if df.empty:
                # Attempt fuzzy fallback
                try:
                    available = [str(x).strip().lower() for x in available_items]
                    mapped = []
                    mapping_info = {}
                    for orig in lowered:
                        best, score = process.extractOne(orig, available)
                        mapping_info[orig] = (best, score)
                        if score >= 65:
                            mapped.append(best)
                    if mapped:
                        logger.info("Fuzzy-mapped requested items %s -> %s (scores: %s)", items, mapped, mapping_info)
                        df = df[df[id_col].isin(mapped)]
                except Exception as exc:
                    logger.warning("Fuzzy fallback failed: %s", exc)

            if df.empty:
                logger.error(f"No data for {comparison_type}s {items}")
                return None, None, None
            logger.info(f"Filtered data for {comparison_type}s {items}. Shape: {df.shape}")
                
        # Year filtering - only apply if year column exists and comparison type is not project
        if years and "year" in df.columns and comparison_type.lower() != "project":
            years = [y for y in years if isinstance(y, int) and 1900 <= y <= 9999]
            if years:
                df = df[df["year"].isin(years)]
                logger.info(f"Filtered data for years {years}. Shape: {df.shape}")
            else:
                logger.info("Year filter provided but resulted in empty/invalid set; skipping year filter")
        elif years and comparison_type.lower() == "project":
            logger.info("Year filter skipped for project type analysis (no year column in project data)")
        
        # Sort
        sort_cols = [c for c in ["final location", "year"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols)
        
        if category and category != "general":
            relevant_columns = ["final location", "year"]
            category_keys = get_category_keys(category)
            category_columns = flatten_columns(get_columns_for_keys(category_keys))
            for col in df.columns:
                if col in category_columns:
                    relevant_columns.append(col)
            relevant_columns = list(dict.fromkeys(relevant_columns))
            df = df[[col for col in relevant_columns if col in df.columns]]
            logger.info(f"Filtered columns for category '{category}'. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        defaults = {
            "year": 2020,
            "total sold - igr": 0,
            "1bhk_sold - igr": 0,
            "flat total": 0,
            "shop total": 0,
            "office total": 0,
            "others total": 0,
            "1bhk total": 0,
            "<1bhk total": 0
        }
        
        df = df.infer_objects(copy=False).fillna({col: defaults.get(col, 0) for col in df.columns})
        logger.info(f"Final data shape: {df.shape}, columns: {df.columns.tolist()}")
        return df, defaults, id_col
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None


def create_documents(df, item_ids: List[str], defaults, columns_by_key: Dict[str, List[str]], years: List[int] = None, comparison_type: str = "Location", id_col: str = "final location"):
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]

    documents: List[Document] = []
    for mapping_key, columns in columns_by_key.items():
        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            continue

        # Optimize: Filter DF once for all items in this mapping key
        item_ids_lower = [i.lower() for i in item_ids]
        # Pre-filter the dataframe to only include relevant rows
        valid_items_df = df[df[id_col].astype(str).str.lower().isin(item_ids_lower)]
        
        # Group by id_col to avoid repeated filtering
        item_groups = {str(name).lower(): group for name, group in valid_items_df.groupby(valid_items_df[id_col].astype(str).str.lower())}

        content_lines: List[str] = []
        for item_id in item_ids_lower:
            item_df = item_groups.get(item_id, pd.DataFrame())

            if comparison_type.strip().lower() == "project" or "year" not in df.columns:
                for col in valid_cols:
                    value = defaults.get(col, 0)
                    if not item_df.empty and col in item_df.columns:
                        try:
                            value = item_df.iloc[0][col]
                        except Exception:
                            value = item_df[col].iloc[0]
                    content_lines.append(f"{item_id}_{mapping_key}_{col}: {value}")
            else:
                for col in valid_cols:
                    year_values = []
                    for year in years:
                        year_df = item_df[item_df["year"] == year] if not item_df.empty else pd.DataFrame()
                        
                        val = defaults.get(col, 0)
                        if not year_df.empty and col in year_df.columns:
                            val = year_df[col].iloc[0]

                        # Check if value is a string-ified dict/list (common for demographic data)
                        if isinstance(val, str) and (val.strip().startswith('{') or val.strip().startswith('[')):
                            try:
                                parsed = ast.literal_eval(val)
                                if isinstance(parsed, dict):
                                    # Format dict as "Key: Value, Key: Value"
                                    # Limit to top 10 items if it's huge
                                    items_list = list(parsed.items())
                                    formatted_items = [f"{k}: {v}" for k, v in items_list[:15]]
                                    val_str = ", ".join(formatted_items)
                                    val = f"{{ {val_str} }}"
                                elif isinstance(parsed, list):
                                    val = f"{parsed}" 
                            except:
                                pass # Keep original string if parse fails

                        year_values.append(f"{year}:{val}")
                    content_lines.append(f"{item_id}_{mapping_key}_{col}: {', '.join(year_values)}")

        if content_lines:
            documents.append(
                Document(
                    page_content="\n".join(content_lines),
                    metadata={
                        'columns': valid_cols,
                        'items': [i.lower() for i in item_ids],
                        'mapping_key': mapping_key,
                        'years': years,
                    }
                )
            )
            logger.info("Created document for mapping key %s with columns: %s", mapping_key, valid_cols)

    logger.info("Created %s documents for items: %s", len(documents), item_ids)
    return documents

def count_tokens(text, model="gpt-4o-mini"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0


def get_llm(provider_name=None):
    from langchain_openai import ChatOpenAI  

    env_provider = (os.getenv("USE_LLM") or "openai").strip().lower()
    provider = (provider_name or env_provider).strip().lower()
    logger.info("Using LLM provider: %s", provider)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as exc:
            raise RuntimeError(
                "langchain-google-genai not installed. Run: pip install langchain-google-genai google-generativeai"
            ) from exc

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini.")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemma-3-27b-it"),
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=8192,
            convert_system_message_to_human=True,
        )

    # Default to OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not api_key.startswith("sk-"):
        raise RuntimeError("Missing/invalid OPENAI_API_KEY.")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        temperature=0.3,
        max_completion_tokens=15000,
        max_retries=3,
    )


def compute_metrics(df: pd.DataFrame, mapping_keys: List[str], columns_by_key: Dict[str, List[str]], item_ids: List[str], id_col: str = "final location", comparison_type: str = "Location") -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    metrics = {}
    for item_id in item_ids:
        item_data = {}
        for mapping_key in mapping_keys:
            columns = columns_by_key.get(mapping_key, [])
            valid_cols = [col for col in columns if col in df.columns]
            
            if not valid_cols:
                continue

            item_df = df[df[id_col] == item_id.lower()]
            
            if item_df.empty:
                continue
                
            key_metrics = {}
            for col in valid_cols:
                if comparison_type.strip().lower() == "project" or "year" not in df.columns:
                    # Single value for project
                    try:
                        value = float(item_df[col].iloc[0])
                    except:
                        value = 0.0
                    key_metrics[col] = {"value": value}
                else:
                    # Year-wise values
                    year_values = {}
                    for year in [2020, 2021, 2022, 2023, 2024]:
                        year_df = item_df[item_df["year"] == year]
                        val_to_use = 0.0
                        
                        if not year_df.empty and col in year_df.columns:
                            raw_val = year_df[col].iloc[0]
                            
                            # Handle string-ified dict/list
                            if isinstance(raw_val, str) and (raw_val.strip().startswith('{') or raw_val.strip().startswith('[')):
                                try:
                                    val_to_use = ast.literal_eval(raw_val)
                                except:
                                    val_to_use = raw_val # Return raw string if parse fails
                            else:
                                try:
                                    val_to_use = float(raw_val)
                                except:
                                    val_to_use = 0.0
                        else:
                            val_to_use = 0.0
                            
                        year_values[str(year)] = val_to_use
                    key_metrics[col] = year_values
                    
            if key_metrics:
                item_data[mapping_key] = key_metrics
                
        if item_data:
            metrics[item_id] = item_data
            
    return metrics


def build_cache_key(items: List[str], mapping_keys: List[str], columns: List[str]):
    combined = f"{sorted(items)}|{sorted(mapping_keys)}|{sorted(columns)}"
    return md5(combined.encode()).hexdigest()


def build_vector_store(documents: List[Document], embeddings: HuggingFaceEmbeddings, cache_key: str):
    cache_dir = Path("vector_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.faiss"
    
    if cache_path.exists():
        try:
            vector_store = FAISS.load_local(str(cache_path), embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded vector store from cache")
            return vector_store
        except Exception as e:
            logger.warning(f"Failed to load cached vector store: {e}")
    
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(cache_path))
    return vector_store


def build_bm25_retriever(documents: List[Document]):
    return BM25Retriever.from_documents(documents)


def hybrid_retrieve(query: str, mapping_keys: List[str], vector_store: FAISS, bm25_retriever, top_k: int = 6):
    semantic_docs = vector_store.similarity_search(query, k=top_k)
    
    # Updated for newer LangChain versions - use invoke() instead of get_relevant_documents()
    keyword_docs = bm25_retriever.invoke(query)[:top_k]
    
    # Filter by mapping keys
    def matches_keys(doc):
        doc_key = doc.metadata.get('mapping_key', '')
        return any(key.lower() in doc_key.lower() or doc_key.lower() in key.lower() for key in mapping_keys)
    
    semantic_filtered = [d for d in semantic_docs if matches_keys(d)]
    keyword_filtered = [d for d in keyword_docs if matches_keys(d)]
    
    # Combine and deduplicate
    combined = semantic_filtered + keyword_filtered
    seen_content = set()
    unique_docs = []
    for doc in combined:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)
            
    return unique_docs[:top_k]


def clean_response(text: str) -> str:
    """Clean and format LLM response while preserving markdown structure."""
    text = text.strip()
    
    # Remove markdown code block wrappers if present
    if text.startswith("```") and text.endswith("```"):
        lines = text.split('\n')
        if len(lines) > 2:
            text = '\n'.join(lines[1:-1])
    
    text = re.sub(r'\*\*\*+', '**', text)
    text = re.sub(r'_{3,}', '__', text)
    
    # Normalize bullet points
    text = re.sub(r'^\s*[•·∙⋅○●]\s+', '- ', text, flags=re.MULTILINE)
    
    # Remove literal <br> tags which break some markdown renderers or show up as text
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    
    return text.strip()


def get_comparison_items(comparison_type, base_dir: Path):
    """Returns available items for comparison type (locations/cities/projects)"""
    try:
        pickle_path = base_dir / PICKLE_FILE
        
        if not pickle_path.exists():
            logger.error("Pickle file not found")
            return []
        
        df = joblib.load(pickle_path)
        df.columns = [normalize_colname(str(c)) for c in df.columns]
        
        # Filter for comparison type
        df_type = df[df["__type"] == comparison_type]
        
        # Resolve ID column
        configured_id = SHEET_CONFIG[comparison_type]["id_col"]
        id_col = configured_id
        
        if id_col not in df_type.columns:
            cols_norm = {normalize_colname(str(c)): c for c in df_type.columns}
            if id_col in cols_norm:
                id_col = cols_norm[id_col]
            else:
                try:
                    best, score = process.extractOne(id_col, list(cols_norm.keys()))
                    if score >= 70:
                        id_col = cols_norm[best]
                    else:
                        logger.error(f"ID column '{configured_id}' not found")
                        return []
                except:
                    return []
        
        # Get unique values
        items = df_type[id_col].dropna().astype(str).str.strip().unique().tolist()
        items = sorted([item for item in items if item and item.lower() != 'nan'])
        
        logger.info(f"Found {len(items)} {comparison_type}s")
        return items
        
    except Exception as e:
        logger.exception(f"Error in get_comparison_items for {comparison_type}")
        return []


def get_project_recommendations(df):
    """Returns a list of dicts with project_name, village (final_location), and city for project search recommendations."""
    required_cols = ['project name', 'final_location', 'city']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")
    recs = df[required_cols].drop_duplicates().dropna()
    return recs.to_dict(orient='records')


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Approximate token count for a string."""
    if not text:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(text)))


def is_query_relevant(query: str, llm) -> bool:
    """Check if user query is relevant to real estate analysis."""
    relevance_prompt = f"""You are a relevance checker for a real estate analysis assistant.

User Query: "{query}"

Is this query related to real estate analysis, property data, market trends, pricing, sales, demand, supply, or similar real estate topics?

Respond with ONLY "YES" or "NO".

YES - if about real estate, properties, market analysis
NO - if about unrelated topics like weather, sports, recipes, etc.

Response:"""

    try:
        response = llm.invoke(relevance_prompt)
        answer = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
        return "YES" in answer
    except Exception as e:
        logger.warning(f"Relevance check failed: {e}. Assuming relevant.")
        return True
