import requests as reqs
import time
import threading

def invoke(function,res):
    url = f"http://33.33.33.132:31112/function/{function}"
    res=reqs.post(url=url, data=res.content.decode("utf-8")[:-1])

def handle(req):
    image_url = req
    url = "http://33.33.33.132:31112/function/starter"
    s1=time.time()
    res = reqs.post(url=url, data=image_url)
    print(res)
    url = "http://33.33.33.132:31112/function/rgb"
    res = reqs.post(url=url, data=image_url)
    print(res)
    url = "http://33.33.33.132:31112/function/resize"
    res = reqs.post(url=url, data=res.content.decode("utf-8")[:-1])
    print(res)
    url = "http://33.33.33.132:31112/function/update"
    res = reqs.post(url=url, data=res.content.decode("utf-8")[:-1])
    print(res)
    threads = []
    for function in ["resnet","fastercnn" ]:
        thread = threading.Thread(target=invoke,args=(function,res))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
        
    return "Completed"