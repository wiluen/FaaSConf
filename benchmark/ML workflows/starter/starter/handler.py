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
    image_url = req
    logging.info("Predicting from url: " + image_url)
#    headers = {
#        "Content-type": "application/json",
#        "Access-Control-Allow-Origin": "*"
#    }
    try:
        with urlopen(image_url) as testImage:
            image = Image.open(testImage)
    except:
        response_content = 'Bad input'
        return response_content
    if image.mode == "RGB":
        response_content = 'RGB'
        return response_content
    else:
        response_content = 'Convert'
        return response_content
