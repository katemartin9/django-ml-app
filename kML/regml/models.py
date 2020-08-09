from django.db import models
from django.contrib.postgres.fields import JSONField
import datetime as dt
from django.contrib.auth.models import User


# Create your models here.
class FileMetaData(models.Model):
    project_name = models.TextField(primary_key=True, max_length=50)
    date = models.DateTimeField(default=dt.datetime.now())
    user = models.ForeignKey(User, on_delete=models.PROTECT, default=User)

    class Meta:
        verbose_name = 'Metadata RegML'


class RegData(models.Model):
    observations = JSONField()
    project_name = models.ForeignKey(FileMetaData, on_delete=models.PROTECT, max_length=50, blank=False, null=False)

    class Meta:
        verbose_name = 'Uploaded Data'
