# Generated by Django 3.0.8 on 2021-03-24 23:14

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('regml', '0009_auto_20200914_1726'),
    ]

    operations = [
        migrations.CreateModel(
            name='Dropdown',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('col_name', models.TextField()),
                ('project_name', models.ForeignKey(max_length=50, on_delete=django.db.models.deletion.PROTECT, to='regml.FileMetaData')),
            ],
        ),
    ]
