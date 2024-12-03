from django.db import models

# Create your models here.


class User(models.Model):
    username = models.CharField(max_length=12)
    password = models.CharField(max_length=12)
    description = models.TextField(blank=True)


class Patient(models.Model):
    name = models.CharField(max_length=12)
    age = models.IntegerField(null=True)
    sex = models.CharField(max_length=1, blank=True)
    height = models.IntegerField(null=True)
    weight = models.IntegerField(null=True)
    liverDiagIds = models.TextField(blank=True)
    retinaDiagIds = models.TextField(blank=True)
    coloneDiagIds = models.TextField(blank=True)
    gleasonDiagIds = models.TextField(blank=True)
    description = models.TextField(blank=True)


class Diagnosis(models.Model):
    patient = models.IntegerField(null=True)  # 关联患者
    type = models.CharField(max_length=20)  # 模型类型
    path = models.CharField(max_length=100)  # 图片/文件路径
    isFile = models.BooleanField(default=True)  # 文件还是目录
    res = models.TextField()  # 诊断结果  图表路径-诊断结果-精度-诊断结果索引
    description = models.TextField(blank=True)
