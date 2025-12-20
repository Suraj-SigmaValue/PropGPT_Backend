"""
API Serializers for PropGPT Django Backend
Handles request/response validation for all API endpoints
"""

from rest_framework import serializers


class QueryRequestSerializer(serializers.Serializer):
    query = serializers.CharField(required=True)
    items = serializers.ListField(child=serializers.CharField(), required=True)
    categories = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_null=True,
        default=[]
    )
    comparison_type = serializers.CharField(required=True)
    mapping_llm_provider = serializers.CharField(required=False, default='openai')
    response_llm_provider = serializers.CharField(required=False, default='openai')
    years = serializers.ListField(child=serializers.IntegerField(), required=False, allow_null=True, default=[2020, 2021, 2022, 2023, 2024])
    forced_mapping_keys = serializers.ListField(child=serializers.CharField(), required=False, allow_null=True, default=[])
    session_id = serializers.CharField(required=False)  # Add this for session tracking
    bypass_mapping = serializers.BooleanField(required=False, default=False)


class QueryResponseSerializer(serializers.Serializer):
    """Main query response"""
    response_text = serializers.CharField()
    mapping_keys = serializers.ListField(child=serializers.CharField())
    selected_columns = serializers.ListField(child=serializers.CharField())
    input_tokens = serializers.IntegerField()
    output_tokens = serializers.IntegerField()
    cached = serializers.BooleanField()
    cache_metadata = serializers.DictField(required=False)


class MappingKeysRequestSerializer(serializers.Serializer):
    """Request for planner agent to identify mapping keys"""
    query = serializers.CharField(required=True)
    candidate_keys = serializers.ListField(child=serializers.CharField(), required=True)
    llm_provider = serializers.CharField(default='openai')


class MappingKeysResponseSerializer(serializers.Serializer):
    """Response from planner agent"""
    selected_keys = serializers.ListField(child=serializers.CharField())


class ColumnSelectionRequestSerializer(serializers.Serializer):
    """Request for column agent"""
    query = serializers.CharField(required=True)
    selected_keys = serializers.ListField(child=serializers.CharField(), required=True)
    candidate_columns = serializers.ListField(child=serializers.CharField(), required=True)
    llm_provider = serializers.CharField(default='openai')


class ColumnSelectionResponseSerializer(serializers.Serializer):
    """Response from column agent"""
    selected_columns = serializers.ListField(child=serializers.CharField())


class CorrectionRequestSerializer(serializers.Serializer):
    """Request for correction agent (HITL)"""
    query = serializers.CharField(required=True)
    old_keys = serializers.ListField(child=serializers.CharField(), required=True)
    candidate_keys = serializers.ListField(child=serializers.CharField(), required=True)
    llm_provider = serializers.CharField(default='openai')


class CorrectionResponseSerializer(serializers.Serializer):
    """Response from correction agent"""
    new_keys = serializers.ListField(child=serializers.CharField())


class FeedbackRequestSerializer(serializers.Serializer):
    """HITL feedback submission"""
    query = serializers.CharField(required=True)
    items = serializers.ListField(child=serializers.CharField(), required=True)
    categories = serializers.ListField(child=serializers.CharField(), required=True)
    old_mapping_keys = serializers.ListField(child=serializers.CharField(), required=True)
    comparison_type = serializers.CharField(required=True)
    feedback_type = serializers.ChoiceField(choices=['up', 'down'], required=True)


class FeedbackResponseSerializer(serializers.Serializer):
    """HITL feedback response"""
    status = serializers.CharField()
    message = serializers.CharField()
    new_response = serializers.CharField(required=False, allow_null=True)
    new_mapping_keys = serializers.ListField(child=serializers.CharField(), required=False)


class CacheStatsSerializer(serializers.Serializer):
    """Cache statistics"""
    active_entries = serializers.IntegerField()
    expired_entries = serializers.IntegerField()
    total_entries = serializers.IntegerField(required=False)  # Made optional


class CacheClearResponseSerializer(serializers.Serializer):
    """Cache clear response"""
    status = serializers.CharField()
    message = serializers.CharField()
    entries_cleared = serializers.IntegerField()


class ComparisonItemsRequestSerializer(serializers.Serializer):
    """Request to get comparison items"""
    comparison_type = serializers.CharField(required=True)


class ComparisonItemsResponseSerializer(serializers.Serializer):
    """Response with comparison items"""
    items = serializers.ListField(child=serializers.CharField())
    count = serializers.IntegerField()


class LoadMappingsRequestSerializer(serializers.Serializer):
    """Request to load mappings"""
    comparison_type = serializers.CharField(required=True)


class LoadMappingsResponseSerializer(serializers.Serializer):
    """Response with mappings"""
    category_mapping = serializers.DictField()
    column_mapping = serializers.DictField()


class RelevanceCheckRequestSerializer(serializers.Serializer):
    """Request to check query relevance"""
    query = serializers.CharField(required=True)
    llm_provider = serializers.CharField(default='openai')


class RelevanceCheckResponseSerializer(serializers.Serializer):
    """Response from relevance check"""
    is_relevant = serializers.BooleanField()


class GraphExecuteRequestSerializer(serializers.Serializer):
    """Request to execute LangGraph workflow"""
    query = serializers.CharField(required=True)
    comparison_type = serializers.CharField(required=True)
    candidate_keys = serializers.ListField(child=serializers.CharField(), required=True)
    llm_provider = serializers.CharField(default='openai')


class GraphExecuteResponseSerializer(serializers.Serializer):
    """Response from graph execution"""
    selected_keys = serializers.ListField(child=serializers.CharField())
    selected_columns = serializers.ListField(child=serializers.CharField())
    messages = serializers.ListField(child=serializers.DictField())
    iteration_count = serializers.IntegerField()


class ProjectRecommendationsRequestSerializer(serializers.Serializer):
    """Request for project recommendations"""
    pass  # No parameters needed


class ProjectRecommendationSerializer(serializers.Serializer):
    """Single project recommendation"""
    project_name = serializers.CharField()
    final_location = serializers.CharField()
    city = serializers.CharField()


class ProjectRecommendationsResponseSerializer(serializers.Serializer):
    """Response with project recommendations"""
    recommendations = ProjectRecommendationSerializer(many=True)
    count = serializers.IntegerField()
