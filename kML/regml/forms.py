from django import forms

COL_TYPES = ((None, '-'), ('c', 'categorical'), ('n', 'numeric'), ('d', 'date'))


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50, required=True, strip=True)
    file = forms.FileField(required=True)
    tick = forms.BooleanField(required=False)


class ColumnTypesForm(forms.Form):
    col_name = forms.CharField(required=True, label='Name', disabled=True)
    col_type = forms.ChoiceField(required=True, choices=COL_TYPES, label='Type')
    y = forms.BooleanField(required=False)

