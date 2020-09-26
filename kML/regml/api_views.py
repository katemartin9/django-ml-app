from rest_framework.generics import ListAPIView
from rest_framework.serializers import ModelSerializer
from .models import DataOutput


class DataOutputSerializer(ModelSerializer):

    class Meta:
        model = DataOutput
        fields = ['output', 'output_name']


class DataOutputApiView(ListAPIView):
    serializer_class = DataOutputSerializer

    def get_queryset(self):
        title = self.kwargs['title']
        return DataOutput.objects.filter(project_name=title)




