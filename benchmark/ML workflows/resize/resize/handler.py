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
    # profile.stop()
    return result
    
@profile
def wrapper_function(req):
    img_data = req
    #print(type(img_data))
    base64_img_bytes = img_data.encode('utf-8')
    decoded_image_data = base64.decodebytes(base64_img_bytes)
    # generate img from byte
    image= Image.open(io.BytesIO(decoded_image_data))
    w,h = image.size
    #print("Image size: " + str(w) + "x" + str(h))
    
    if h < 1600 and w < 1600:
        #img_data = dump_img(image)
        headers = {
            "Content-type": "application/json",
            "Access-Control-Allow-Origin": "*"
        }

        return img_data
    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    #print("resize: " + str(w) + "x" + str(h) + " to " + str(new_size[0]) + "x" + str(new_size[1]))
    if max(new_size) / max(image.size) >= 0.5:
        method = Image.BILINEAR
    else:
        method = Image.BICUBIC
    image = image.resize(new_size, method)
    img_byte_arr = io.BytesIO()
    image.convert('RGB').save(img_byte_arr, format='JPEG')
    img_data = base64.encodebytes(img_byte_arr.getvalue()).decode('utf-8')
    return img_data
