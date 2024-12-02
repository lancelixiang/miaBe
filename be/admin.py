from django.contrib import admin
from .models import User, Patient, Diagnosis

# class UserAdmin(admin.ModelAdmin):
#     list_display = ('username', 'password', 'description')

# Register your models here.
# admin.site.register(User, UserAdmin)
admin.site.register(User)
admin.site.register(Patient)
admin.site.register(Diagnosis)