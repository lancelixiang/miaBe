# Generated by Django 4.2.16 on 2024-12-02 12:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('be', '0002_diagnosis_patient'),
    ]

    operations = [
        migrations.AlterField(
            model_name='diagnosis',
            name='description',
            field=models.TextField(blank=True),
        ),
        migrations.AlterField(
            model_name='patient',
            name='description',
            field=models.TextField(blank=True),
        ),
        migrations.AlterField(
            model_name='user',
            name='description',
            field=models.TextField(blank=True),
        ),
    ]
