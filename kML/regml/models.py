from django.db import models
from django.contrib.postgres.fields import ArrayField


# Create your models here.
class RegData(models.Model):
    x = ArrayField(
            models.CharField(max_length=100, blank=False, null=False),
            size=50, blank=False, default=list
        )
    y = models.CharField(max_length=100, blank=False, null=False)
    filename = models.TextField(max_length=50, blank=False, null=False)


#class TimeSeriesData(models.Model):
    #pass