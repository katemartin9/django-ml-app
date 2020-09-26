from django.urls import path
from .views import upload_file, show_table, render_graphs
from .api_views import DataOutputApiView

urlpatterns = [
    path(r'upload-file', upload_file, name='upload_file'),
    path(r'data-preview/<str:title>', show_table, name='data_preview'),
    path(r'model-options/<str:title>', render_graphs, name='render_graphs'),
    path(r'model-options/<str:title>/api', DataOutputApiView.as_view(), name='api'),
]