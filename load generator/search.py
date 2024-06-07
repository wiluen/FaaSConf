from locust import HttpUser, between, task
import random
import sys
import time
import json
import numpy as np
from datetime import datetime
from random import randint
import logging
import os

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
    
class MyUser(HttpUser):
    @task
    def search_ticket(self):
        TRIP=[{"from": "Shang Hai", "to": "Su Zhou"},
                 {"from": "Su Zhou", "to": "Shang Hai"},
                 {"from": "Wu Xi", "to": "Shang Hai"}
                 {"from": "Nan Jing", "to": "Shang Hai"},
                 {"from": "Wu Xi", "to": "Su Zhou"}
        ]
        TRAVEL_DATES = ["2023-12-28","2023-12-29"]
        date=random.choice(TRAVEL_DATES)
        tripinfo=random.choice(TRIP)
        head = {"Accept": "application/json",
                "Content-Type": "application/json"}
        body_start = {
            "startingPlace": tripinfo['from'],
            "endPlace": tripinfo['to'],
            "departureTime": date
        }
        start_time = time.time()
        response = self.client.post(
            url="http://33.33.33.132:31112/function/get-left-trip-tickets",
            headers=head,
            json=body_start,
            )
        end_time=time.time()
        to_log = {'status_code': response.status_code,
                  'response_time': end_time - start_time,'response':try_to_read_response_as_json(response)}
        #             
        log_verbose(to_log)
          
