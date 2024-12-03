from django.core.files.storage import FileSystemStorage
from datetime import datetime


def main(req):
    file = req.FILES['file']
    fs = FileSystemStorage('./be/static/upload')
    now = datetime.now()
    year = now.year
    month = now.month
    filename = fs.save(f'{year}{month}/{file.name}', file)
    return filename
