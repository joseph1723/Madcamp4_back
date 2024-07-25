from django.urls import path
from .views import example_view, upload_view, classify_view

urlpatterns = [
    path('api/example/', example_view, name='example_view'),
    path('api/upload/', upload_view, name='upload_view'),
    path('api/classify/', classify_view, name = 'classify_view')
]
