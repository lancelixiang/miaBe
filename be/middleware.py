# be/middleware.py
import re
from django.conf import settings
from django.utils.deprecation import MiddlewareMixin


class BlogIgnoreCsrfMiddleware(MiddlewareMixin):
    def process_request(self, request, *args, **kwargs):
        if hasattr(settings, 'URL_IGNORE_CSRF_LIST'):
            url_ignore_list = settings.URL_IGNORE_CSRF_LIST
        else:
            url_ignore_list = ['/api/diagnosis/','/api/patients/']
            
        for u in url_ignore_list:
            if re.match(u, request.path):
                request.csrf_processing_done = True