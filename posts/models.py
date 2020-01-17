from django.core.management.base import BaseCommand
from django.db import models
from datetime import datetime
import csv
# Create your models here.
class Posts(models.Model):
    title = models.CharField(max_length=200)
    body = models.TextField()
    created_at = models.DateTimeField(default=datetime.now, blank=True)
    location = models.CharField(max_length=50)

    def __str__(self):
        return self.title
    class Meta: 
        verbose_name_plural = "Posts"


    # do python manage.py makemigrations posts
    # then python manage.py migrate
