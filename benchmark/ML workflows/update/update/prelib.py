import logging
import json
import base64
import io
from datetime import datetime
from PIL import Image
def _log_msg(msg):
    logging.info("{}: {}".format(datetime.now(),msg))

def _extract_bilinear_pixel(img, x, y, ratio, xOrigin, yOrigin):
    xDelta = (x + 0.5) * ratio - 0.5
    x0 = int(xDelta)
    xDelta -= x0
    x0 += xOrigin
    if x0 < 0:
        x0 = 0;
        x1 = 0;
        xDelta = 0.0;
    elif x0 >= img.shape[1]-1:
        x0 = img.shape[1]-1;
        x1 = img.shape[1]-1;
        xDelta = 0.0;
    else:
        x1 = x0 + 1;
    
    yDelta = (y + 0.5) * ratio - 0.5
    y0 = int(yDelta)
    yDelta -= y0
    y0 += yOrigin
    if y0 < 0:
        y0 = 0
        y1 = 0
        yDelta = 0.0
    elif y0 >= img.shape[0]-1:
        y0 = img.shape[0]-1
        y1 = img.shape[0]-1
        yDelta = 0.0;
    else:
        y1 = y0 + 1

    #Get pixels in four corners
    bl = img[y0, x0]
    br = img[y0, x1]
    tl = img[y1, x0]
    tr = img[y1, x1]
    #Calculate interpolation
    b = xDelta * br + (1. - xDelta) * bl
    t = xDelta * tr + (1. - xDelta) * tl
    pixel = yDelta * t + (1. - yDelta) * b
    return pixel.astype(np.uint8)
def load_img(img_data):
    base64_img_bytes = img_data.encode('utf-8')
    decoded_image_data = base64.decodebytes(base64_img_bytes)
    # generate img from byte
    img= Image.open(io.BytesIO(decoded_image_data))
    return img
def dump_img(image):
    img_byte_arr = io.BytesIO()
    image.convert('RGB').save(img_byte_arr, format='JPEG')
    img_data = base64.encodebytes(img_byte_arr.getvalue()).decode('utf-8')
#    return json.dumps(img_data)
    return img_data
