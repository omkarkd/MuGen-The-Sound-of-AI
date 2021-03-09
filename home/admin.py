from django.contrib import admin
from home.models import AccessRecord
from generative.models import GenerateAudioModel
# Register your models here.
admin.site.register(AccessRecord)
admin.site.register(GenerateAudioModel)