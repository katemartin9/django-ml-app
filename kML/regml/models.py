from django.db import models
from django.contrib.postgres.fields import JSONField
from django.contrib.auth.models import User
import django

COL_TYPES = (('c', 'categorical'), ('n', 'numeric'), ('d', 'date'))


# Create your models here.
class FileMetaData(models.Model):
    project_name = models.TextField(primary_key=True, max_length=50)
    date = models.DateTimeField(default=django.utils.timezone.now)
    user = models.ForeignKey(User, on_delete=models.PROTECT, default=User)

    class Meta:
        verbose_name = 'Metadata RegML'


class RegData(models.Model):
    observations = JSONField()
    project_name = models.ForeignKey(FileMetaData, on_delete=models.PROTECT, max_length=50, blank=False, null=False)

    class Meta:
        verbose_name = 'Uploaded Data'


class ColumnTypes(models.Model):
    project_name = models.ForeignKey(FileMetaData, on_delete=models.PROTECT, max_length=50, blank=False, null=False)
    col_name = models.TextField(blank=False, null=False)
    col_type = models.TextField(choices=COL_TYPES, blank=False, null=False)
    y = models.BooleanField(blank=False, null=False)


class DataOutput(models.Model):
    project_name = models.ForeignKey(FileMetaData, on_delete=models.PROTECT, max_length=50, blank=False, null=False)
    output_name = models.TextField(blank=False, null=False, max_length=50)
    output = JSONField()


class Dropdown(models.Model):
    project_name = models.ForeignKey(FileMetaData, on_delete=models.PROTECT, max_length=50, blank=False, null=False)
    col_name = models.TextField(blank=False, null=False)