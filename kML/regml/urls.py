from django.urls import path
from .views import upload_file, show_table

urlpatterns = [
    path(r'upload-file', upload_file, name='upload_file'),
    path(r'data-preview/<str:title>', show_table, name='data_preview')
]