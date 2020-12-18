from django.contrib import admin

from .models import Imageupload

#admin.site.register(Imageupload)

@admin.register(Imageupload)
class ImageuploadmaskAdmin(admin.ModelAdmin):
    list_display = ('pk','date_of_upload', 'readiness', 'image_file', 'result_file')
