"""
URL configuration for mia project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from be import views

router = routers.DefaultRouter()
# router.register('users', views.UserView, 'User')
router.register('patient', views.PatientView, 'Patient')
router.register('diagnosis', views.DiagnosisView, 'Diagnosis')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('api/login', views.login),
    path('api/register', views.register),
    path('api/upload', views.upload),
    path('api/models/gleason/<dir>/<img>', views.gleason, name='gleason'),
    path('api/get-csrf-token/', views.get_csrf_token, name='get_csrf_token'),
]
