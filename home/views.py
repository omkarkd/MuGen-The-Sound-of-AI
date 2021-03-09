from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def index(request):
    from datetime import datetime
    import socket
    from home.models import AccessRecord
    ip = socket.gethostbyname(socket.gethostname())
    internet_available = socket.gethostbyname(socket.gethostname()) != '127.0.0.1'
    time_now = datetime.now()
    AccessRecord(ip_address=ip, time=time_now).save()
    context = dict(ip=ip, internet_available=internet_available, time_now=time_now)
    return render(request, 'home/index.html', context=context)