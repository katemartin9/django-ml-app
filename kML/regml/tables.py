import django_tables2 as tables


class DataOverviewTable(tables.Table):
    class Meta:
        attrs = {"class": "mytable"}