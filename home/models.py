from django.db import models

# Create your models here.
class AccessRecord(models.Model):
    ip_address = models.CharField(max_length=256, unique=False)
    time = models.CharField(max_length=256, unique=False)
        
    def __str__(self):
        ip = self.ip_address
        time = self.time
        return f'{ip=} {time=}'