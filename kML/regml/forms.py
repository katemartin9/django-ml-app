from django import forms
from .models import RegData, ColumnTypes
import urllib

COL_TYPES = ((None, '-'), ('c', 'categorical'), ('n', 'numeric'), ('d', 'date'))


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50, required=True, strip=True)
    file = forms.FileField(required=True)
    tick = forms.BooleanField(required=False)

    def clean_title(self):
        form_title = self.cleaned_data.get("title")
        existing = RegData.objects.filter(project_name=form_title).exists()
        if existing:
            raise forms.ValidationError(u"This project name is already in use.")
        return form_title


class ColumnTypesForm(forms.Form):

    def __init__(self, *args, **kwargs):
        if 'fullpath' in kwargs:
            self.full_url_path = kwargs.pop('fullpath')
        super().__init__(*args, **kwargs)

    col_name = forms.CharField(required=True, label='Name')
    col_type = forms.ChoiceField(required=True, choices=COL_TYPES, label='Type')
    y = forms.BooleanField(required=False)

    def clean(self):
        super().clean()
        form_title = urllib.parse.unquote(self.full_url_path.split('/')[-1])
        existing = ColumnTypes.objects.filter(project_name=form_title).exists()
        if existing:
            ColumnTypes.objects.filter(project_name=form_title).delete()


class ColumnsToRemove(forms.Form):

    def __init__(self, *args, **kwargs):
        if 'fullpath' in kwargs:
            self.full_url_path = kwargs.pop('fullpath')
        super().__init__(*args, **kwargs)

    col_name = forms.CharField(required=True, label='Name')
    remove_add = forms.BooleanField(label='Remove', required=False)