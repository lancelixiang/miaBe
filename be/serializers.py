# todo/serializers.py

from rest_framework import serializers
from .models import User, Patient, Diagnosis
# from django.utils.decorators import method_decorator 
# from django.views.decorators.csrf import csrf_exempt

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        # fields = ('id', 'username', 'password', 'description')
        fields = "__all__"


class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        # fields = ('id', 'name', 'age', 'sex', 'height', 'weight', 'liverDiagIds', 'retinaDiagIds', 'coloneDiagIds', 'gleasonDiagIds', 'description')
        fields = "__all__"

# @method_decorator(csrf_exempt, name='')
class DiagnosisSerializer(serializers.ModelSerializer):
    # @method_decorator(csrf_exempt, name='')
    class Meta:
        model = Diagnosis
        # fields = ('id', 'patients', 'type', 'path', 'isFile', 'res', 'description')
        fields = "__all__"
