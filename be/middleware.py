# be/middleware.py
import re
from django.conf import settings
from django.utils.deprecation import MiddlewareMixin
# from rest_framework.response import Response


class BlogIgnoreCsrfMiddleware(MiddlewareMixin):
    def process_request(self, request, *args, **kwargs):
        # 如果请求是GET请求且没有请求JSON响应，则可能是通过浏览器直接访问
        # if request.method == 'GET' and ('HTTP_ACCEPT' not in request.META or
        #                                 'application/json' not in request.META.get('HTTP_ACCEPT', '')):
        #     return Response({'detail': 'This endpoint is not accessible via browser.'}, status=403)

        if hasattr(settings, 'URL_IGNORE_CSRF_LIST'):
            url_ignore_list = settings.URL_IGNORE_CSRF_LIST
        else:
            url_ignore_list = ['/api/diagnosis/', '/api/patient/']

        for u in url_ignore_list:
            if re.match(u, request.path):
                request.csrf_processing_done = True
