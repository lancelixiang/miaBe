# Generated by Django 4.2.16 on 2024-12-03 15:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('be', '0008_alter_patient_sex'),
    ]

    operations = [
        migrations.AlterField(
            model_name='diagnosis',
            name='path',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='diagnosis',
            name='type',
            field=models.CharField(max_length=20),
        ),
    ]
