# Generated by Django 3.0.8 on 2020-09-14 17:24

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('regml', '0007_auto_20200913_2058'),
    ]

    operations = [
        migrations.AlterField(
            model_name='filemetadata',
            name='date',
            field=models.DateTimeField(default=datetime.datetime(2020, 9, 14, 17, 24, 43, 898945)),
        ),
    ]
