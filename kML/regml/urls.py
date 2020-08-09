from django.urls import path
from .views import upload_file

urlpatterns = [
    path(r'upload-file', upload_file, name='upload_file')
]