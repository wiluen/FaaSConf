import logging
import json
import base64
import logging
from urllib.request import urlopen
from PIL import Image
import io
import os
import requests as reqs
from .customized_LP import MyLineProfiler
from .prelib import _log_msg
profile = MyLineProfiler()
logging.basicConfig(level=logging.INFO)
def handle(req):
    profile.start()
    result = wrapper_function(req)
    # profile.stop()
    return result
    
@profile
def wrapper_function(req):
    img_data = req
    
    base64_img_bytes = img_data.encode('utf-8')
    decoded_image_data = base64.decodebytes(base64_img_bytes)
    #print(type(decoded_image_data))
    # generate img from byte
    image= Image.open(io.BytesIO(decoded_image_data))
    w,h = image.size
    #print("Image size: " + str(w) + "x" + str(h))
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if exif != None and exif_orientation_tag in exif:
            orientation = exif.get(exif_orientation_tag, 1)
#            _log_msg('Image has EXIF Orientation: ' + str(orientation))
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    img_byte_arr = io.BytesIO()
    image.convert('RGB').save(img_byte_arr, format='JPEG')
    img_data = base64.encodebytes(img_byte_arr.getvalue()).decode('utf-8')
    return img_data
