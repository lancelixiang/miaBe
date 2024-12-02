from django.db import models

# Create your models here.


class User(models.Model):
    username = models.CharField(max_length=12)
    password = models.CharField(max_length=12)
    description = models.TextField(blank=True)


class Patient(models.Model):
    name = models.CharField(max_length=12)
    age = models.IntegerField(blank=True)
    sex = models.CharField(max_length=1, default='M', blank=True)
    height = models.IntegerField(blank=True)
    weight = models.IntegerField(blank=True)
    liverDiagIds = models.TextField(blank=True)
    retinaDiagIds = models.TextField(blank=True)
    coloneDiagIds = models.TextField(blank=True)
    gleasonDiagIds = models.TextField(blank=True)
    description = models.TextField(blank=True)


class Diagnosis(models.Model):
    patient = models.IntegerField(blank=True)  # 关联患者
    type = models.CharField(max_length=12)  # 模型类型
    path = models.CharField(max_length=30)  # 图片/文件路径
    isFile = models.BooleanField(default=True)  # 文件还是目录
    res = models.TextField()  # 诊断结果
    description = models.TextField(blank=True)
