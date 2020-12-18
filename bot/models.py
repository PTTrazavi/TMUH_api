from django.db import models
from django.core.files import File
import os

#from django.core.files.storage import default_storage #GCS
class Imageupload(models.Model):
    image_file = models.ImageField(upload_to='images/')
    result_file = models.ImageField(upload_to='images/', blank=True)
    readiness = models.CharField(max_length=1, default='0')
    date_of_upload = models.DateTimeField(auto_now_add = True)

    def __str__(self):
        return self.image_file.name
