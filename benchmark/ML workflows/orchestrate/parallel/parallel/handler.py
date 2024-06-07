import requests as reqs
import threading

def invoke(function,res,results,img_url):
    if function=="mobilenet":
        arg=img_url
    else:
        arg=res.content.decode("utf-8")[:-1]
    url = f"http://33.33.33.132:31112/function/{function}"
    res=reqs.post(url=url, data=arg)
    results.append(res.content)

def handle(req):
    image_url = req
    url = "http://33.33.33.132:31112/function/starter"
    res = reqs.post(url=url, data=image_url)
    url = "http://33.33.33.132:31112/function/rgb"
    res = reqs.post(url=url, data=image_url)
    url = "http://33.33.33.132:31112/function/resize"
    res = reqs.post(url=url, data=res.content.decode("utf-8")[:-1])
    url = "http://33.33.33.132:31112/function/update"
    res = reqs.post(url=url, data=res.content.decode("utf-8")[:-1])
    results=[]
    threads = []
    for function in ["resnet","mobilenet"]:
        thread = threading.Thread(target=invoke,args=(function,res,results,image_url))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return results
