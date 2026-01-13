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
from django.http import HttpResponse
from django.template import Template, Context
from xhtml2pdf import pisa
import markdown

from .serializers import *
from core.core_utils import *
from core.agents import planner_identify_mapping_keys, agent_pick_relevant_columns, agent_correction_mapping
from core.graph_agent import create_graph
from core.prompts import build_location_prompt, build_city_prompt, build_project_prompt
from langchain_core.messages import HumanMessage, AIMessage
from core.time_estimate import estimate_processing_time

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
class GetCategoriesView(APIView):
    """Get available categories for a comparison type"""
    
    def post(self, request):
        serializer = GetCategoriesRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        comparison_type = serializer.validated_data['comparison_type']
        print("Comparison type:", comparison_type)
        try:
            # Get category mapping for the comparison type
            category_mapping = get_category_mapping(comparison_type)
            
            # Extract unique categories (keys from the mapping)
            categories = list(category_mapping.keys())
            
            response_data = {
                'categories': categories,
                'comparison_type': comparison_type
            }
            response_serializer = GetCategoriesResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
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
        categories = serializer.validated_data.get('categories', []) or []
        
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
            # Calculate estimated time BEFORE processing
            estimated_time = estimate_processing_time(
                query=query,
                items=items,
                categories=categories,
                years=years,
                comparison_type=comparison_type
            )
            logger.info(f"Estimated processing time: {estimated_time} seconds")
            
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
                "categories": categories,  # Pass selected categories for filtering
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
            
            # Appending contact link
            contact_link = "\n\n[Please click here to contact our property expert](https://sigmavalue.in/contact/?page=contactform)"
            cleaned_response += contact_link
            
            # Get selected columns with source information
            from core.source_utils import get_columns_with_sources
            selected_columns = final_state.get('selected_columns', [])
            columns_with_sources = get_columns_with_sources(selected_columns, comparison_type)
            
            # Update chat history
            chat_history.append({"role": "user", "content": query, "timestamp": datetime.now().isoformat()})
            chat_history.append({"role": "assistant", "content": cleaned_response, "timestamp": datetime.now().isoformat()})
            
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
                
            request.session['chat_history'] = chat_history
            request.session['last_context_text'] = final_state.get('context_text', '')
            request.session['last_query'] = query
            request.session['last_comparison_type'] = comparison_type
            request.session.modified = True
            
            # Return response
            response_data = {
                'response_text': cleaned_response,
                'mapping_keys': final_state.get('selected_keys', []),
                'selected_columns': selected_columns,
                'columns_with_sources': columns_with_sources,  # Added source information
                'input_tokens': final_state.get('input_tokens', 0),
                'output_tokens': final_state.get('output_tokens', 0),
                'cached': False,
                'estimated_time_seconds': estimated_time,  # Include estimated time
                'session_id': request.session.session_key
            }
            
            response_serializer = QueryResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Error in MainQueryView: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class DownloadReportView(APIView):
    """Download chat history as PDF"""
    
    def get(self, request):
        chat_history = request.session.get('chat_history', [])
        
        if not chat_history:
             # If no history, return simple message
            return HttpResponse("No chat history available to download.", content_type='text/plain')

        # Process chat history: Convert Markdown to HTML
        processed_history = []
        for msg in chat_history:
            content = msg.get('content', '')
            # Convert Markdown to HTML
            # extensions=['extra'] enables tables, fenced code blocks, etc.
            html_content = markdown.markdown(content, extensions=['extra', 'sane_lists'])
            
            processed_history.append({
                'role': msg.get('role', 'unknown'),
                'timestamp': msg.get('timestamp', ''),
                'content': html_content  # Now safe HTML
            })

        # HTML Template with improved CSS for Markdown elements
        html_string = """
        <html>
        <head>
            <style>
                @page { size: A4; margin: 2cm; }
                body { font-family: Helvetica, sans-serif; font-size: 11px; line-height: 1.4; color: #333; }
                
                /* Layout */
                .meta { color: #7f8c8d; font-size: 10px; margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
                .message_block { margin-bottom: 25px; }
                
                /* Headers */
                h1 { color: #2c3e50; font-size: 18px; border-bottom: 2px solid #2c3e50; padding-bottom: 5px; margin-bottom: 20px; }
                h2 { color: #2980b9; font-size: 15px; margin-top: 15px; margin-bottom: 8px; font-weight: bold; }
                h3 { color: #34495e; font-size: 13px; margin-top: 12px; margin-bottom: 6px; font-weight: bold; }
                h4 { color: #555; font-size: 12px; margin-top: 10px; font-weight: bold; }
                
                /* Roles */
                .role_user { color: #e67e22; font-weight: bold; font-size: 12px; margin-bottom: 5px; background-color: #fdf2e9; padding: 5px; border-radius: 4px; display: inline-block; }
                .role_assistant { color: #27ae60; font-weight: bold; font-size: 12px; margin-bottom: 5px; background-color: #eafaf1; padding: 5px; border-radius: 4px; display: inline-block; }
                .timestamp { float: right; color: #95a5a6; font-size: 9px; margin-top: 5px; }
                
                /* Content bodies */
                .content { margin-top: 5px; text-align: justify; }
                
                /* Lists */
                ul { margin: 5px 0 5px 20px; padding: 0; }
                ol { margin: 5px 0 5px 20px; padding: 0; }
                li { margin-bottom: 3px; }
                
                /* Text Formatting */
                strong { font-weight: bold; color: #000; }
                em { font-style: italic; }
                p { margin-bottom: 10px; margin-top: 0; }
                
                /* Tables (if any) */
                table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10px; }
                th { border: 1px solid #ddd; padding: 6px; background-color: #f2f2f2; font-weight: bold; text-align: left; }
                td { border: 1px solid #ddd; padding: 6px; }
                
                /* Code blocks */
                pre { background-color: #f8f8f8; border: 1px solid #ddd; padding: 10px; border-radius: 4px; white-space: pre-wrap; font-family: Courier, monospace; font-size: 10px; }
                code { background-color: #f8f8f8; padding: 2px 4px; border-radius: 3px; font-family: Courier, monospace; }
            </style>
        </head>
        <body>
            <h1>PropGPT Session Report</h1>
            <div class="meta">Generated on: {{ date }} | Session ID: {{ session_id }}</div>
            
            {% for msg in chat_history %}
                <div class="message_block">
                    <div>
                        <span class="role_{{ msg.role }}">{{ msg.role|title }}</span>
                        <span class="timestamp">{{ msg.timestamp }}</span>
                    </div>
                    <!-- 'safe' filter is required so Django doesn't escape the HTML tags we just generated -->
                    <div class="content">{{ msg.content|safe }}</div>
                </div>
            {% endfor %}
            
            <div style="text-align: center; margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px; color: #bdc3c7; font-size: 9px;">
                Generated by PropGPT AI Assistant
            </div>
        </body>
        </html>
        """
        
        try:
            template = Template(html_string)
            context = Context({
                'chat_history': processed_history,  # Use processed history
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'session_id': request.session.session_key or "N/A"
            })
            html = template.render(context)
            
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="propgpt_chat_report.pdf"'
            
            pisa_status = pisa.CreatePDF(html, dest=response)
            
            if pisa_status.err:
                return HttpResponse(f'PDF generation error: {pisa_status.err}', status=500)
            return response
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return HttpResponse(f"Error generating PDF: {str(e)}", status=500)

@method_decorator(csrf_exempt, name='dispatch')
class GenerateStructuredReportView(APIView):
    """Generate a structured, analyst-grade report as PDF"""
    
    def post(self, request):
        serializer = StructuredReportRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning(f"Structured report validation failed: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        logger.info(f"Generating structured report for session: {request.session.session_key}")
        query = serializer.validated_data.get('query') or request.session.get('last_query')
        comparison_type = serializer.validated_data.get('comparison_type') or request.session.get('last_comparison_type')
        items = serializer.validated_data.get('items', [])
        sections = serializer.validated_data.get('sections', [])
        preset = serializer.validated_data.get('preset')
        
        context_text = request.session.get('last_context_text', '')
        chat_history = request.session.get('chat_history', [])
        
        if not context_text:
            logger.warning(f"Report generation failed: No context_text in session for key {request.session.session_key}")
            return Response({'error': 'No context available for report generation. Please run a query first.'}, status=status.HTTP_400_BAD_REQUEST)
            
        logger.info(f"Report generation context size: {len(context_text)} chars, History: {len(chat_history)} messages")
        try:
            from core.report_generator import ReportGenerator
            
            # Initialize LLM
            llm = get_llm('openai')
            
            # Initialize Generator
            generator = ReportGenerator(
                llm=llm,
                context_text=context_text,
                chat_history=chat_history,
                query=query
            )
            
            # Generate Report Content
            html_template, sections_data = generator.generate_report(
                selected_sections=sections,
                preset=preset
            )
            
            # Convert to PDF
            pdf_content = generator.create_pdf(
                html_template=html_template,
                sections_data=sections_data,
                session_id=request.session.session_key or "N/A"
            )
            
            if not pdf_content:
                return Response({'error': 'Failed to generate PDF'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
            response = HttpResponse(pdf_content, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="propgpt_structured_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
            
            return response
            
        except Exception as e:
            logger.exception(f"Error generating structured report: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
