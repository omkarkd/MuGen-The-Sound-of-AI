from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect
from .forms import GenerateAudioForm
from .models import GenerateAudioModel
import socket, datetime, os

region, temp, ip, time_now = '', '', '', ''

# Create your views here.
@csrf_protect
def generate_music(request):
    form = GenerateAudioForm()
    global region, temp, ip, time_now
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if request.method == 'POST':
        form = GenerateAudioForm(request.POST)
        
        if form.is_valid():
            from generative.final import main
            region = form.cleaned_data['region']
            region = str(region)
            temp = form.cleaned_data['temp']
            temp = float(temp)
            ip = socket.gethostbyname(socket.gethostname())
            time_now = datetime.datetime.now()
            song_dir = os.path.join(BASE_DIR, 'download')
            if not os.path.isdir(song_dir):
                os.mkdir(song_dir)
            file_name = os.path.join(song_dir, 'download.mid')
            for i in range(1, 500):
                if os.path.isfile(file_name):
                    file_name = os.path.join(song_dir, f'download ({i}).mid')
                else:
                    break
            main(region, temp, file_name)
            GenerateAudioModel(region=region, temp=temp, ip_address=ip, time=time_now).save()
            if os.path.isfile(os.path.join(BASE_DIR, 'music.mid')):
                filename = os.path.join(BASE_DIR, 'music.mid')
                with open(filename, 'rb') as fh:
                    response = HttpResponse(fh.read(), content_type='audio/midi')
                    response['Content-Disposition'] = 'inline; filename=' + os.path.basename(filename)
                return render(request, 'generative/index.html', dict(form=form), content_type='audio/midi')
        # return HttpResponse(f'{region=}\n{temp=}\n{ip=}\n{time_now=}')
            
    return render(request, 'generative/index.html', dict(form=form, region=region, temp=temp, ip=ip, time=time_now))

def output(request):
    if request.method == 'POST':
        region, temp, ip, time_now = '', '', '', ''
        return HttpResponse(f'{region=}\n{temp=}\n{ip=}\n{time_now=}')