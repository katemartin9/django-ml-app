from django import forms


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50, required=True, strip=True)
    file = forms.FileField(required=True)
    tick = forms.BooleanField(required=False)