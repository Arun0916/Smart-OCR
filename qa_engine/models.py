from django.db import models
from django.contrib.auth.models import User
from documents.models import Page, OCRLine, Document

class Question(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question_text = models.TextField()
    answer_text = models.TextField(blank=True, null=True)
    page_reference = models.ForeignKey(Page, on_delete=models.SET_NULL, null=True, blank=True)
    line_reference = models.ForeignKey(OCRLine, on_delete=models.SET_NULL, null=True, blank=True)
    asked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q: {self.question_text[:50]} - A: {self.answer_text[:50] if self.answer_text else 'Pending'}"
