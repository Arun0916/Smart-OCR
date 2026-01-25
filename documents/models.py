from django.db import models
from django.contrib.auth.models import User

class Document(models.Model):
    title = models.CharField(max_length=255)
    uploaded_file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.title

class Page(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    page_number = models.IntegerField()
    image = models.ImageField(upload_to='pages/')

    class Meta:
        unique_together = ('document', 'page_number')

    def __str__(self):
        return f"{self.document.title} - Page {self.page_number}"

class OCRLine(models.Model):
    page = models.ForeignKey(Page, on_delete=models.CASCADE)
    line_number = models.IntegerField()
    text = models.TextField()

    class Meta:
        unique_together = ('page', 'line_number')

    def __str__(self):
        return f"Page {self.page.page_number} - Line {self.line_number}: {self.text[:50]}"
