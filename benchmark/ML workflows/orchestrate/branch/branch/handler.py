import requests as reqs
import json
def handle(req):
    input_data = json.loads(req)
    image_url = input_data["url"]
    image_type = input_data["type"]
    url = "http://33.33.33.132:31112/function/starter"
    res = reqs.post(url=url, data=image_url)
    if image_type == "rgb":
        url = "http://33.33.33.132:31112/function/rgb"
        res = reqs.post(url=url, data=image_url)
        url = "http://33.33.33.132:31112/function/resize"
        res = reqs.post(url=url, data=res.content.decode("utf-8")[:-1])
        url = "http://33.33.33.132:31112/function/update"
        res = reqs.post(url=url, data=res.content.decode("utf-8")[:-1])
        url = "http://33.33.33.132:31112/function/mobilenet"
        res = reqs.post(url=url, data=image_url)
    else:
        url = "http://33.33.33.132:31112/function/load"
        res = reqs.post(url=url, data=image_url)
        url = "http://33.33.33.132:31112/function/resnet"
        res = reqs.post(url=url, data=res.content.decode("utf-8")[:-1])
    
    results=str(res.content)+"type: "+image_type
    return results
