"""
Django Models for API app
Handles chat sessions, query history, and metadata
"""

from django.db import models
from django.contrib.auth.models import User


class ChatSession(models.Model):
    """Stores chat session data"""
    session_id = models.CharField(max_length=255, unique=True, db_index=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Session {self.session_id}"


class QueryHistory(models.Model):
    """Stores query and response history"""
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='queries')
    query_text = models.TextField()
    response_text = models.TextField()
    comparison_type = models.CharField(max_length=50)
    items = models.JSONField()
    categories = models.JSONField()
    mapping_keys = models.JSONField()
    selected_columns = models.JSONField()
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    cached = models.BooleanField(default=False)
    feedback = models.CharField(max_length=10, null=True, blank=True, choices=[
        ('up', 'Thumbs Up'),
        ('down', 'Thumbs Down')
    ])
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
        verbose_name_plural = 'Query histories'
    
    def __str__(self):
        return f"Query at {self.created_at}: {self.query_text[:50]}"


class CacheMetadata(models.Model):
    """Stores metadata about cached responses"""
    cache_key = models.CharField(max_length=255, unique=True, db_index=True)
    query = models.TextField()
    items = models.JSONField()
    mapping_keys = models.JSONField()
    comparison_type = models.CharField(max_length=50)
    provider = models.CharField(max_length=50)
    hit_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    last_accessed = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-last_accessed']
    
    def __str__(self):
        return f"Cache {self.cache_key[:20]}... (hits: {self.hit_count})"
