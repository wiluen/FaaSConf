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
profile = MyLineProfiler()
logging.basicConfig(level=logging.INFO)
def handle(req):
    profile.start()
    result = wrapper_function(req)
    profile.stop()
    return result
    
@profile
def wrapper_function(req):
    image_url = req
    logging.info("Predicting from url: " + image_url)
    with urlopen(image_url) as testImage:
        image = Image.open(testImage)
    img_byte_arr = io.BytesIO()
    image.convert('RGB').save(img_byte_arr, format='JPEG')
    img_data = base64.encodebytes(img_byte_arr.getvalue()).decode('utf-8')
    return img_data
