import json
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets

from .serializers import UserSerializer
from .models import User
from modules.retina import retinaDiag
from modules.colone import coloneDiag


class UserView(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    queryset = User.objects.all()


@csrf_exempt
def login(req):
    if req.method == "POST":
        print('**************', req.POST, type(req.POST))
        datas = json.loads(req.body)
        username = datas.get('username')
        password = datas.get('password')

        user = User.objects.filter(username=username, password=password)
        res = {"result": "success" if len(user) >= 1 else "fail"}
        return HttpResponse(json.dumps(res), status=200)

    info = {'error': 'can not be a get method'}
    return HttpResponse(json.dumps(info))


def retina(request, img):
    if request.method == 'GET':
        print('img', img)
        response = retinaDiag.main(img)
        return HttpResponse(response, status=200)


def colone(request, img):
    if request.method == 'GET':
        print('file name', img)
        response = coloneDiag.main(fileName=img)
        return HttpResponse(response, status=200)
