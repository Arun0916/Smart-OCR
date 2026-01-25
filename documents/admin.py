from django.contrib import admin
from .models import Document, Page, OCRLine

admin.site.register(Document)
admin.site.register(Page)
admin.site.register(OCRLine)
