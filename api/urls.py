"""
API URL Configuration for PropGPT
Maps all API endpoints to their corresponding views
"""

from django.urls import path
from .views import *

app_name = 'api'

urlpatterns = [
    # Main Query Endpoint
    path('query/', MainQueryView.as_view(), name='query'),
    
    # Data Management
    path('mappings/load/', LoadMappingsView.as_view(), name='load_mappings'),
    path('items/', GetComparisonItemsView.as_view(), name='comparison_items'),
    path('projects/recommendations/', ProjectRecommendationsView.as_view(), name='project_recommendations'),
    
    # Agent Endpoints
    path('agents/planner/', PlannerAgentView.as_view(), name='planner_agent'),
    path('agents/column/', ColumnAgentView.as_view(), name='column_agent'),
    path('agents/correction/', CorrectionAgentView.as_view(), name='correction_agent'),
    
    # LangGraph
    path('graph/execute/', GraphExecuteView.as_view(), name='graph_execute'),
    
    # Cache Management
    path('cache/stats/', CacheStatsView.as_view(), name='cache_stats'),
    path('cache/clear/', CacheClearView.as_view(), name='cache_clear'),
    
    # Utilities
    path('relevance/', RelevanceCheckView.as_view(), name='relevance_check'),
    
    # HITL Feedback
    path('feedback/', FeedbackView.as_view(), name='feedback'),
]
