import json
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.middleware.csrf import get_token
from rest_framework import viewsets
from django.http import JsonResponse
from django.utils import timezone
import subprocess

from .serializers import UserSerializer, PatientSerializer, DiagnosisSerializer
from .models import User, Patient, Diagnosis
from modules import uploadSave
from modules.retina import retinaDiag
from modules.colone import coloneDiag
from modules.gleason import gleasonDiag


class UserView(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    queryset = User.objects.all()


class PatientView(viewsets.ModelViewSet):
    serializer_class = PatientSerializer
    queryset = Patient.objects.all()


# @method_decorator(csrf_exempt, name="dispatch")
class DiagnosisView(viewsets.ModelViewSet):
    serializer_class = DiagnosisSerializer
    queryset = Diagnosis.objects.all()


def get_csrf_token(request):
    csrf_token = get_token(request)
    return JsonResponse({'csrf_token': csrf_token})


@csrf_exempt
def login(req):
    if req.method == "POST":
        datas = json.loads(req.body)
        username = datas.get('username')
        password = datas.get('password')

        user = User.objects.filter(username=username, password=password)
        res = {
            "result": "success" if len(user) >= 1 else "fail",
            'role': user[0].role,
            'id': user[0].id,
        }
        return HttpResponse(json.dumps(res), status=200)

    info = {'error': 'can not be a get method'}
    return HttpResponse(json.dumps(info))


@csrf_exempt
def register(req):
    if req.method == "POST":
        datas = json.loads(req.body)
        username = datas.get('username')
        password = datas.get('password')

        user = User(username=username, password=password,
                    createDate=timezone.now())
        user.save()
        res = {"result": "success"}
        return HttpResponse(json.dumps(res), status=200)

    info = {'error': 'can not be a get method'}
    return HttpResponse(json.dumps(info))


@csrf_exempt
def upload(req):
    if req.method == "POST":
        filename = uploadSave.main(req)
        res = {"result": "success", 'filename': filename}
        return HttpResponse(json.dumps(res), status=200)

    info = {'error': 'can not be a get method'}
    return HttpResponse(json.dumps(info))


def retina(request, dir, img):
    if request.method == 'GET':
        # print('img', img)
        response = retinaDiag.main(dir, img)
        return HttpResponse(response, status=200)

    info = {'error': 'can not be a get method'}
    return HttpResponse(json.dumps(info))


def colone(request, dir, img):
    if request.method == 'GET':
        # print('file name', img)
        response = coloneDiag.main(dir, fileName=img)
        return HttpResponse(response, status=200)

    info = {'error': 'can not be a get method'}
    return HttpResponse(json.dumps(info))


def gleason(request, dir, img):
    if request.method == 'GET':
        script_path = 'modules/gleason/gleasonDiag.py'
        try:
            response = subprocess.run(
                ['python', script_path, '--dir', dir, '--img', img], capture_output=True, text=True, check=True)
            return HttpResponse(response.stdout, status=200)
        except subprocess.CalledProcessError as e:
            return HttpResponse('error', status=500)

    info = {'error': 'can not be a get method'}
    return HttpResponse(json.dumps(info))
