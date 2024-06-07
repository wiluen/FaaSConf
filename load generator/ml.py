import time
import requests
from locust import HttpUser, TaskSet, task, stats, between
from random import randint, expovariate, choice
import json
import random
from locust.exception import StopUser
# stats.CSV_STATS_INTERVAL_SEC = 1
import logging

handler = logging.FileHandler("/home/user/code/faas-resource/locustfile_wf.log")
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger = logging.getLogger("Debugging logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
def log_verbose(to_log):
    logger.debug(json.dumps(to_log))
def try_to_read_response_as_json(response):
    try:
        return response.json()
    except:
        try:
            return response.content.decode('utf-8')
        except:
            return response.content
class WebTasks(HttpUser):
    @task
    def parallel(self):
        url = "http://33.33.33.132:31112/function/wf6"
        img_url = "http://33.33.33.223:8888/images/British_Shorthair_168.jpg"  
        a=time.time()
        response_wf = self.client.post(url, data=img_url)
        # time.sleep(expovariate(0.1))
        b=time.time()
        latency=b-a
        to_log={'response_time':latency,'response':try_to_read_response_as_json(response_wf),'code':response_wf.status_code}
        log_verbose(to_log)


    @task
    def sequence(self):
        url = "http://33.33.33.132:31112/function/wf5"
        img_url = "http://33.33.33.223:8888/images/beagle_184.jpg"  
        # branch/parallel British_Shorthair_168         sequence beagle_184
        a=time.time()
        response_wf = self.client.post(url, data=img_url)
        # time.sleep(expovariate(0.1))
        b=time.time()
        latency=b-a
        to_log={'response_time':latency,'response':try_to_read_response_as_json(response_wf),'code':response_wf.status_code}
        log_verbose(to_log)

    @task(2)
    def branch1(self):
        url = "http://33.33.33.132:31112/function/wf7"  
        img_url = "http://33.33.33.223:8888/images/British_Shorthair_168.jpg" 
        data = {
                "url": img_url,
                "type": "rgb"
            }
        json_data = json.dumps(data)
        a=time.time()
        response_wf = self.client.post(url, data=json_data)
        b=time.time()
        latency=b-a
        to_log={'response_time':latency,'response':try_to_read_response_as_json(response_wf),'code':response_wf.status_code}
        log_verbose(to_log)

    @task(3)
    def branch2(self):
        url = "http://33.33.33.132:31112/function/wf7"  
        img_url = "http://33.33.33.223:8888/images/British_Shorthair_168.jpg" 
        data = {
                "url": img_url,
                "type": "convert"
            }
        json_data = json.dumps(data)
        a=time.time()
        response_wf = self.client.post(url, data=json_data)
        b=time.time()
        latency=b-a
        to_log={'response_time':latency,'response':try_to_read_response_as_json(response_wf),'code':response_wf.status_code}
        log_verbose(to_log)
