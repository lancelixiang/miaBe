from django.core.files.storage import FileSystemStorage
from datetime import datetime
from PIL import Image
import os


def main(req):
    file = req.FILES['file']
    fs = FileSystemStorage('./be/static/upload')
    now = datetime.now()
    year = now.year
    month = now.month
    postFix = file.name.split('.')[-1]
    dir = '/tif' if postFix in ['tif', 'tiff'] else ''
    filename = fs.save(f'{year}{month}{dir}/{file.name}', file)

    if (dir):
        tiff_image = Image.open(os.path.dirname(
            __file__) + '/../be/static/upload/' + filename)
        tiff_image.save(os.path.dirname(__file__) + '/../be/static/upload/' +
                        filename.replace('.tiff', '.png').replace('.tif', '.png'), format='PNG')

    return filename.replace('/tif/', '/')
