from rest_framework.generics import ListAPIView
from rest_framework.serializers import ModelSerializer
from .models import RegData, ColumnTypes


class RegDataSerializer(ModelSerializer):

    class Meta:
        model = RegData
        fields = ['observations']


class ColumnTypesSerializer(ModelSerializer):

    class Meta:
        model = ColumnTypes
        fields = ['col_name', 'col_type', 'y']


class RegDataApiView(ListAPIView):
    serializer_class = RegDataSerializer

    def get_queryset(self):
        title = self.kwargs['title']
        return RegData.objects.filter(project_name=title)


class ColumnTypesApiView(ListAPIView):
    serializer_class = ColumnTypesSerializer

    def get_queryset(self):
        title = self.kwargs['title']
        return ColumnTypes.objects.filter(project_name=title)


