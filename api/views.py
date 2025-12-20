"""
API Views for PropGPT Django Backend
All views are WRAPPERS - they import and call core functions WITHOUT modification
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from pathlib import Path
import logging
import uuid
from datetime import datetime
import joblib
import re
from hashlib import md5

from .serializers import *
from core.core_utils import *
from core.agents import planner_identify_mapping_keys, agent_pick_relevant_columns, agent_correction_mapping
from core.graph_agent import create_graph
from core.prompts import build_location_prompt, build_city_prompt, build_project_prompt
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# Helper to flatten column structure
def flatten_columns(columns_by_key):
    """Flatten dict of {key: [columns]} into single list"""
    result = []
    for cols in columns_by_key.values():
        if isinstance(cols, list):
            result.extend(cols)
        else:
            result.append(cols)
    return list(set(result))


@method_decorator(csrf_exempt, name='dispatch')
class LoadMappingsView(APIView):
    """Load category and column mappings for a comparison type"""
    
    def post(self, request):
        serializer = LoadMappingsRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        comparison_type = serializer.validated_data['comparison_type']
        
        try:
            # Call core function (ZERO modification)
            cat_map, col_map = load_mappings(comparison_type)
            
            response_data = {
                'category_mapping': cat_map,
                'column_mapping': col_map
            }
            response_serializer = LoadMappingsResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class GetComparisonItemsView(APIView):
    """Get available items for a comparison type"""
    
    def post(self, request):
        serializer = ComparisonItemsRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        comparison_type = serializer.validated_data['comparison_type']
        base_dir = Path(settings.DATA_DIR)
        
        try:
            # Call core function (ZERO modification)
            items = get_comparison_items(comparison_type, base_dir)
            
            response_data = {
                'items': items,
                'count': len(items)
            }
            response_serializer = ComparisonItemsResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error getting comparison items: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class PlannerAgentView(APIView):
    """Planner agent - identifies relevant mapping keys"""
    
    def post(self, request):
        serializer = MappingKeysRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        query = serializer.validated_data['query']
        candidate_keys = serializer.validated_data['candidate_keys']
        llm_provider = serializer.validated_data.get('llm_provider', 'openai')
        
        try:
            # Get LLM instance
            llm = get_llm(llm_provider)
            
            # Call core function (ZERO modification)
            selected_keys = planner_identify_mapping_keys(llm, query, candidate_keys)
            
            # Safety limit (align with user's prompt update: Hard limit 7-10 keys)
            selected_keys = selected_keys[:10]
            
            response_data = {'selected_keys': selected_keys}
            response_serializer = MappingKeysResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in planner agent: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class ColumnAgentView(APIView):
    """Column agent - selects relevant columns"""
    
    def post(self, request):
        serializer = ColumnSelectionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        query = serializer.validated_data['query']
        selected_keys = serializer.validated_data['selected_keys']
        candidate_columns = serializer.validated_data['candidate_columns']
        llm_provider = serializer.validated_data.get('llm_provider', 'openai')
        
        try:
            # Get LLM instance
            llm = get_llm(llm_provider)
            
            # Call core function (ZERO modification)
            selected_columns = agent_pick_relevant_columns(llm, query, selected_keys, candidate_columns)
            
            response_data = {'selected_columns': selected_columns}
            response_serializer = ColumnSelectionResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in column agent: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class CorrectionAgentView(APIView):
    """Correction agent - proposes new mapping keys for HITL"""
    
    def post(self, request):
        serializer = CorrectionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        query = serializer.validated_data['query']
        old_keys = serializer.validated_data['old_keys']
        candidate_keys = serializer.validated_data['candidate_keys']
        llm_provider = serializer.validated_data.get('llm_provider', 'openai')
        
        try:
            # Get LLM instance
            llm = get_llm(llm_provider)
            
            # Call core function (ZERO modification)
            new_keys = agent_correction_mapping(llm, query, old_keys, candidate_keys)
            
            response_data = {'new_keys': new_keys}
            response_serializer = CorrectionResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in correction agent: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class GraphExecuteView(APIView):
    """Execute LangGraph workflow"""
    
    def post(self, request):
        print(f"function entered GraphExecuteView")
        serializer = GraphExecuteRequestSerializer(data=request.data)
        print(f"serializer created {serializer}")
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        print(f"serializer created successfully")
        query = serializer.validated_data['query']
        print(f"here is the query: {query}")
        comparison_type = serializer.validated_data['comparison_type']
        print(f"here is the comparison_type: {comparison_type}")
        candidate_keys = serializer.validated_data['candidate_keys']
        print(f"here is the candidate_keys: {candidate_keys}")
        llm_provider = serializer.validated_data.get('llm_provider', 'openai')
        print(f"GraphExecuteView: Received request with query=")
        
        try:
            # Get LLM and graph app
            llm = get_llm(llm_provider)
            app = get_graph_app()
            print(f"Graph app loaded:")
            # Prepare initial state
            initial_state = {
                "query": query,
                "comparison_type": comparison_type,
                "candidate_keys": candidate_keys,
                "candidate_columns": [],
                "llm": llm,
                "selected_keys": [],
                "selected_columns": [],
                "iteration_count": 0,
                "messages": []
            }
            print(f"Initial state prepared:")
            
            # Execute graph (ZERO modification)
            config = {"configurable": {"thread_id": request.session.session_key or "default"}}
            final_state = app.invoke(initial_state, config=config)
            print(f"Graph execution completed:")
            response_data = {
                'selected_keys': final_state.get('selected_keys', []),
                'selected_columns': final_state.get('selected_columns', []),
                'messages': [{'role': m.type, 'content': m.content} for m in final_state.get('messages', [])],
                'iteration_count': final_state.get('iteration_count', 0)
            }
            response_serializer = GraphExecuteResponseSerializer(response_data)
            print(f"Graph execution completed_2:")
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"Graph execution completed_exception:")
            logger.error(f"Error executing graph: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class RelevanceCheckView(APIView):
    """Check if query is relevant to real estate analysis"""
    
    def post(self, request):
        serializer = RelevanceCheckRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        query = serializer.validated_data['query']
        llm_provider = serializer.validated_data.get('llm_provider', 'openai')
        
        try:
            # Get LLM instance
            llm = get_llm(llm_provider)
            
            # Call core function (ZERO modification)
            is_relevant = is_query_relevant(query, llm)
            
            response_data = {'is_relevant': is_relevant}
            response_serializer = RelevanceCheckResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CacheStatsView(APIView):
    """Get cache statistics"""
    
    def get(self, request):
        try:
            # Return simplified cache stats without loading embeddings
            # This avoids the PyTorch meta tensor error
            cache_dir = Path(settings.DATA_DIR) / 'response_cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Count cache files
            cache_file = cache_dir / 'semantic_cache.pkl'
            
            if cache_file.exists():
                import pickle
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    active_entries = len([e for e in cache_data.get('entries', {}).values() if not e.get('expired', False)])
                    expired_entries = len([e for e in cache_data.get('entries', {}).values() if e.get('expired', False)])
                except Exception as e:
                    logger.warning(f"Could not read cache file: {e}")
                    active_entries = 0
                    expired_entries = 0
            else:
                active_entries = 0
                expired_entries = 0
            
            stats = {
                'active_entries': active_entries,
                'expired_entries': expired_entries,
                'total_entries': active_entries + expired_entries
            }
            
            response_serializer = CacheStatsSerializer(stats)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            # Return empty stats instead of error
            return Response({
                'active_entries': 0,
                'expired_entries': 0,
                'total_entries': 0
            }, status=status.HTTP_200_OK)

@method_decorator(csrf_exempt, name='dispatch')
class CacheClearView(APIView):
    """Clear response cache"""
    
    def post(self, request):
        try:
            # Initialize cache
            embeddings = get_embeddings()
            cache_dir = Path(settings.DATA_DIR) / 'response_cache'
            cache = get_response_cache(embeddings, cache_dir)
            
            # Get count before clearing
            stats_before = cache.get_stats()
            entries_before = stats_before['active_entries']
            
            # Call core function (ZERO modification)
            cache.clear_all()
            
            response_data = {
                'status': 'success',
                'message': 'Cache cleared successfully',
                'entries_cleared': entries_before
            }
            response_serializer = CacheClearResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ====== ADD THE MISSING FeedbackView ======
@method_decorator(csrf_exempt, name='dispatch')
class FeedbackView(APIView):
    """Handle HITL feedback (thumbs up/down)"""
    
    def post(self, request):
        serializer = FeedbackRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        feedback_type = serializer.validated_data['feedback_type']
        
        if feedback_type == 'up':
            # Thumbs up - just log and acknowledge
            logger.info(f"👍 Positive feedback received for query")
            response_data = {
                'status': 'success',
                'message': 'Thank you for your positive feedback!'
            }
            response_serializer = FeedbackResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        
        # Thumbs down - trigger correction
        query = serializer.validated_data['query']
        items = serializer.validated_data['items']
        categories = serializer.validated_data['categories']
        old_keys = serializer.validated_data['old_mapping_keys']
        comparison_type = serializer.validated_data['comparison_type']
        
        logger.info(f"👎 Negative feedback received. Triggering correction for query: {query}")
        
        try:
            base_dir = Path(settings.DATA_DIR)
            
            # Get candidate keys
            set_mappings_for_type(comparison_type)
            candidate_keys = []
            for category in [cat.lower() for cat in categories]:
                candidate_keys.extend(get_category_keys(category))
            candidate_keys = sorted(set(candidate_keys))
            
            # Run correction agent
            llm = get_llm('openai')  # Default to OpenAI for corrections
            new_keys = agent_correction_mapping(llm, query, old_keys, candidate_keys)
            
            logger.info(f"Correction agent proposed {len(new_keys)} new keys: {new_keys}")
            
            # Return new keys for frontend to use
            response_data = {
                'status': 'correction_proposed',
                'message': 'New mapping keys proposed based on feedback',
                'new_mapping_keys': new_keys
            }
            
            response_serializer = FeedbackResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Error processing feedback: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ProjectRecommendationsView(APIView):
    """Get project recommendations"""
    
    def get(self, request):
        try:
            base_dir = Path(settings.DATA_DIR)
            pickle_path = base_dir / 'Pune_Grand_Summary.pkl'
            
            # Load project data
            df = joblib.load(pickle_path)
            df.columns = [normalize_colname(str(c)) for c in df.columns]
            df_projects = df[df["__type"] == "Project"]
            
            # Call core function (ZERO modification)
            recommendations = get_project_recommendations(df_projects)
            
            response_data = {
                'recommendations': recommendations,
                'count': len(recommendations)
            }
            response_serializer = ProjectRecommendationsResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Error getting project recommendations: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ====== KEEP THE UPDATED MainQueryView BELOW ======
# (The MainQueryView class from the previous response should be placed here)
# Make sure to include the complete MainQueryView class here...


@method_decorator(csrf_exempt, name='dispatch')
class MainQueryView(APIView):
    """
    Main query endpoint - orchestrated by LangGraph
    """
    
    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning(f"Query serializer validation failed: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract request data
        query = serializer.validated_data['query']
        items = serializer.validated_data['items']
        comparison_type = serializer.validated_data['comparison_type']
        response_llm_provider = serializer.validated_data.get('response_llm_provider', 'openai')
        bypass_mapping = serializer.validated_data.get('bypass_mapping', False)
        years = serializer.validated_data.get('years') or [2020, 2021, 2022, 2023, 2024]
        
        # Create a configuration hash to detect changes
        config_payload = {
            "comparison_type": comparison_type,
            "items": sorted(items) if items else [],
            "years": sorted(years) if isinstance(years, list) else []
        }
        config_hash = md5(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()
        
        # Initialize session
        if not request.session.session_key:
            request.session.create()
        
        session_key = request.session.session_key
        last_config_hash = request.session.get('last_config_hash')
        
        # If configuration changed, refresh chat and memory
        if last_config_hash and last_config_hash != config_hash:
            logger.info(f"Configuration changed for session {session_key}. Refreshing memory.")
            request.session['chat_history'] = []
            # We will use a unique thread_id for each config to effectively "reset" memory
            thread_id = f"{session_key}_{config_hash}"
        else:
            thread_id = f"{session_key}_{config_hash}" # Always tie thread to config
            
        request.session['last_config_hash'] = config_hash
        chat_history = request.session.get('chat_history', [])
        
        try:
            # Get LLM and Graph App
            llm = get_llm(response_llm_provider)
            app = get_graph_app()
            
            # Initial State for Graph
            initial_state = {
                "query": query,
                "items": items,
                "comparison_type": comparison_type,
                "llm": llm,
                "years": years,
                "chat_history": chat_history,
                "detected_requirements": [],
                "candidate_keys": [],
                "candidate_columns": [],
                "selected_keys": [],
                "selected_columns": [],
                "messages": [HumanMessage(content=query)],
                "iteration_count": 0
            }
            
            # Execute Graph
            config = {"configurable": {"thread_id": thread_id}}
            final_state = app.invoke(initial_state, config=config)
            
            cleaned_response = final_state.get('final_response', "I couldn't generate a response.")
            
            # Update chat history
            chat_history.append({"role": "user", "content": query, "timestamp": datetime.now().isoformat()})
            chat_history.append({"role": "assistant", "content": cleaned_response, "timestamp": datetime.now().isoformat()})
            
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
                
            request.session['chat_history'] = chat_history
            request.session.modified = True
            
            # Return response
            response_data = {
                'response_text': cleaned_response,
                'mapping_keys': final_state.get('selected_keys', []),
                'selected_columns': final_state.get('selected_columns', []),
                'input_tokens': final_state.get('input_tokens', 0),
                'output_tokens': final_state.get('output_tokens', 0),
                'cached': False,
                'session_id': request.session.session_key
            }
            
            response_serializer = QueryResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Error in MainQueryView: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

