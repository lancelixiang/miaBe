from django.db import models

# Create your models here.


class User(models.Model):
    username = models.CharField(max_length=12)
    password = models.CharField(max_length=12)
    description = models.TextField()
