"""Django admin configuration for PropGPT API"""

from django.contrib import admin
from .models import ChatSession, QueryHistory, CacheMetadata


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'user', 'created_at', 'updated_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('session_id',)
    readonly_fields = ('created_at', 'updated_at')


@admin.register(QueryHistory)
class QueryHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'query_text_short', 'comparison_type', 'cached', 'feedback', 'created_at')
    list_filter = ('comparison_type', 'cached', 'feedback', 'created_at')
    search_fields = ('query_text', 'response_text')
    readonly_fields = ('created_at',)
    
    def query_text_short(self, obj):
        return obj.query_text[:50] + '...' if len(obj.query_text) > 50 else obj.query_text
    query_text_short.short_description = 'Query'


@admin.register(CacheMetadata)
class CacheMetadataAdmin(admin.ModelAdmin):
    list_display = ('cache_key_short', 'comparison_type', 'provider', 'hit_count', 'last_accessed')
    list_filter = ('comparison_type', 'provider', 'created_at')
    search_fields = ('cache_key', 'query')
    readonly_fields = ('created_at', 'last_accessed')
    
    def cache_key_short(self, obj):
        return obj.cache_key[:20] + '...'
    cache_key_short.short_description = 'Cache Key'
