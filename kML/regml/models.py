from django.db import models
from django.contrib.postgres.fields import ArrayField


# Create your models here.
class FileMetaData(models.Model):
    project_name = models.TextField(primary_key=True, max_length=50)
    col_names = ArrayField(
            models.TextField(max_length=100, blank=False, null=False),
            size=50, blank=False, default=list
        )
    y_name = models.TextField(max_length=50, blank=False, null=False)


class RegData(models.Model):
    x = ArrayField(
            models.CharField(max_length=100, blank=False, null=False),
            size=50, blank=False, default=list
        )
    y = models.CharField(max_length=100, blank=False, null=False)
    project_name = models.ForeignKey(FileMetaData, on_delete=models.PROTECT, max_length=50, blank=False, null=False)


#class TimeSeriesData(models.Model):
    #pass