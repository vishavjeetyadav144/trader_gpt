from django.urls import path
from .views import ai_decisions_table

urlpatterns = [
    path('decisions/', ai_decisions_table, name='ai_decisions_table'),
]
