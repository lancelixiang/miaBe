from django.http import HttpResponse
# from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets

from .serializers import UserSerializer
from .models import User
from modules.eye import process

class UserView(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    queryset = User.objects.all()
    
# @csrf_exempt
def process_image(request, img):
    if request.method == 'GET':
        # print('img', img)
        response = process.process_img(img)
        return HttpResponse(response, status=200)