from rest_framework import serializers
from .models import Imageupload
from django.utils.timezone import now
import uuid

#images
class ImageuploadSerializer(serializers.ModelSerializer):
    #result_file = serializers.SerializerMethodField()
    class Meta:
        model = Imageupload
        fields = ('id','date_of_upload', 'readiness', 'image_file', 'result_file')

    #def get_result_file(self, obj):
    #    return(obj.image_file.name.split('.')[-2] + '_matting.' + obj.image_file.name.split('.')[-1])
