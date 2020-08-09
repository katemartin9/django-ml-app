from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from django.contrib.auth.models import User


# Create your views here.
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            title = form.cleaned_data.get('title')
            tick = form.cleaned_data.get('tick')
            data = handle_uploaded_file(request.FILES['file'], tick)
            # TODO: check why doesn't throw error when project name exists
            user = User.objects.get()
            load_file_into_db(data, title, user)
            return HttpResponseRedirect('regml/data-overview/')
        else:
            print('Error on form: ', form.errors)
            warning_message = f"Not a valid file."
            form = UploadFileForm()
            return render(request, 'upload_file.html', {'form': form, 'warning': warning_message})
    else:
        form = UploadFileForm()
    return render(request, 'upload_file.html', {'form': form})

