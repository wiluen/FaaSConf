import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import time
import requests
import re
import csv
import itertools

DATA_PATH = '/home/user/code/faas-resource/tabluar_data.csv'
CPU_UNIT_COST=0.000173   #0.173/1000
MEM_UNIT_COST=0.000012    #0.0123/1024
NUM_RESOURCES=3
# CPU=[250,500,750,1000,1250,1500]   #0-5
# MEM=[128, 256, 384, 512, 768, 1024, 2048]  #0-5
# REP=[1,2,3,4,5,6,7,8]  #0-7

#统一资源配置，对于trainticket适当缩小
CPU_MIN=250
CPU_MAX=2000
MEM_MIN=256
MEM_MAX=2048
REP_MIN=1
REP_MAX=8
CONCURR_MIN=1
CONCURR_MAX=10

# search
# CPU_MIN=250
# CPU_MAX=1000
# MEM_MIN=128
# MEM_MAX=1024
# REP_MIN=1
# REP_MAX=8

def construct_memory_pool():
    df=pd.read_csv(DATA_PATH)
    state=[]
    action=[]
    state_=[]
    reward=[]
    # start=4
    # end=10
    # for x in range(12):
    #     function_state=df.iloc[:,start + x * 10 : end + x * 10]
    #     state.append(function_state)
    # df_new=pd.concat(state,axis=1)
    # df_new.to_csv('/home/user/code/faas-resource/GAT-MF/experience/state_buffer.csv',index=False)

    function_action=df.iloc[:,:36]
    function_action.to_csv('/home/user/code/faas-resource/GAT-MF/experience/action_buffer.csv',index=False)

    CPU_UNIT_COST=0.000173   #0.173/1000
    MEM_UNIT_COST=0.000012    #0.0123/1024
    AVG_P75_TIME=[0.194,3.51,0.0118,0.0083,0.0383,0.0082,0.01,0.0372,0.011,0.0088,0.0109,0.2245]
    AVG_LATENCY=3.5
    AVG_THROUGHPUT=110
    price=[]
    a=0.2
    b=0.7
    c=0.3
    w1=0.5
    w2=0.5
    throughput=df.iloc[:,-2]
    p50=df.iloc[:,-4]
    global_reward= AVG_LATENCY / p50 # + c * throughput / AVG_THROUGHPUT 
    for x in range(12):
        cpu_quota=df.iloc[:,37+x*10]
        mem_quota=df.iloc[:,38+x*10]
        replicas=df.iloc[:,39+x*10]
        exec_time=df.iloc[:,44+x*10]   #P75
        price_cost=(cpu_quota*CPU_UNIT_COST+mem_quota*MEM_UNIT_COST)*replicas          # 范围是0.x
        local_reward=-price_cost+a*(AVG_P75_TIME[x]/exec_time)
        r=w1 * local_reward + w2 * global_reward
        reward.append(r)
    df_reward=pd.concat(reward,axis=1)
    df_reward.to_csv('/home/user/code/faas-resource/GAT-MF/experience/reward_buffer.csv',index=False)






def form_x_to_resource_conf(x):
    num_functions = int(len(x)/NUM_RESOURCES)
    resource_config = [[0, 0, 0, 0] for _ in range(num_functions)]
    for i in range(num_functions):
        scaled_cpu = x[i * 4]
        scaled_memory = x[i * 4 + 1]
        scaled_replicas = x[i * 4 + 2]
        scaled_concurrency = x[i * 4 + 3]     
        resource_config[i][0] = round(scaled_cpu * (CPU_MAX - CPU_MIN) + CPU_MIN, 0)
        resource_config[i][1] = round(scaled_memory * (MEM_MAX - MEM_MIN) + MEM_MIN, 0)
        resource_config[i][2] = int(scaled_replicas * (REP_MAX - REP_MIN) + REP_MIN)
        resource_config[i][3] = int(scaled_concurrency * (CONCURR_MAX - CONCURR_MIN) + CONCURR_MIN)    

    return resource_config

def form_x_to_resource_conf_wo_cl(x):
    num_functions = int(len(x)/3)
    resource_config = [[0, 0, 0] for _ in range(num_functions)]
    for i in range(num_functions):
        scaled_cpu = x[i * 3]
        scaled_memory = x[i * 3 + 1]
        scaled_replicas = x[i * 3 + 2] 
        resource_config[i][0] = round(scaled_cpu * (CPU_MAX - CPU_MIN) + CPU_MIN, 0)
        resource_config[i][1] = round(scaled_memory * (MEM_MAX - MEM_MIN) + MEM_MIN, 0)
        resource_config[i][2] = int(scaled_replicas * (REP_MAX - REP_MIN) + REP_MIN)  
    return resource_config

def update_deploy(function,resource_config,tunecon):  #list
    if tunecon:
        for i in range(len(function)):
            cmd= "/bin/bash /home/user/code/updateOpenfaasYaml.sh " + function[i] + " " + str(resource_config[i][0]) + " " + str(resource_config[i][1]) + " " + str(resource_config[i][2]) + " " + str(resource_config[i][3])
            print(cmd)
            os.system(cmd)
    else:
        for i in range(len(function)):
            cmd= "/bin/bash /home/user/code/updateOpenfaasYaml.sh " + function[i] + " " + str(resource_config[i][0]) + " " + str(resource_config[i][1]) + " " + str(resource_config[i][2]) 
            print(cmd)
            os.system(cmd)

def rm_functon(function):
    for f in function:
        cmd= "faas-cli rm " + f 
        print(cmd)
        os.system(cmd)
    return True


def run_locust(locustfile,url,users,spawn_rate,run_time,csv_file):
    os.system("rm /home/user/code/faas-resource/locustfile_wf.log")
    command=f'locust --locustfile {locustfile} --host {url} --users {users} --spawn-rate {spawn_rate} --run-time {run_time}s --headless --csv={csv_file}'
    _ = os.system(command)

# def data2txt(function_list,cpu,mem,per_func_latency,end2end_latency):
    #需要一个配置字典，格式{"name":"function","cpu":512,"mem":256}
def get_metric(x,resource_config,function_list,users,n,benchmark):
    # function_list=['starter','load','resize','update','resnet']
    cpu_container_sql='sum(rate(container_cpu_usage_seconds_total{name=~".+",namespace="openfaas-fn"}[1m])) by (pod) * 100'
    cpu_throttle_sql='sum(increase(container_cpu_cfs_throttled_periods_total{namespace="openfaas-fn"}[1m])) by(pod)'
    mem_container_sql='sum(container_memory_working_set_bytes{namespace="openfaas-fn",container!= "", container!="POD"}) by (pod) / sum(container_spec_memory_limit_bytes{namespace="openfaas-fn",container!= "", container!="POD"}) by (pod) * 100'
    rec_comtainer_sql='sum(rate(container_network_receive_bytes_total{name=~".+",namespace="openfaas-fn"}[1m])) by (pod)'
    transm_container_sql='sum(rate(container_network_transmit_bytes_total{name=~".+",namespace="openfaas-fn"}[1m])) by (pod)'

    url = "http://33.33.33.132:31090/api/v1/query"
    
    cpures = requests.get(
            url=url,
            params={'query':cpu_container_sql}
            )
    thrres = requests.get(
            url=url,
            params={'query':cpu_throttle_sql}
            )
    memres = requests.get(
            url=url,
            params={'query':mem_container_sql}
            )
    recres = requests.get(
            url=url,
            params={'query':rec_comtainer_sql}
            )
    
    cpu_map={}
    thr_map={}
    mem_map={}
    network_map={}

    for i in function_list:
        temp1=[] 
        for value in cpures.json()["data"]["result"]:
            if str(i) in value["metric"]["pod"]: 
                temp1.append(float(value["value"][1]))
        temp1arr=np.array(temp1)
        cpu_avg=temp1arr.mean()
        cpu_map[str(i)]=cpu_avg
            
    for i in function_list:
        temp2=[]  
        for value in thrres.json()["data"]["result"]:
            if str(i) in value["metric"]["pod"]:
                temp2.append(float(value["value"][1]))
        temp2arr=np.array(temp2)
        thr_avg=temp2arr.mean()
        thr_map[str(i)]=thr_avg
                    
    for i in function_list: 
        temp3=[]    
        for value in memres.json()["data"]["result"]:
            if str(i) in value["metric"]["pod"]:
                temp3.append(float(value["value"][1]))
        temp3arr=np.array(temp3)
        mem_avg=temp3arr.mean()
        mem_map[str(i)]=mem_avg

    for i in function_list:  
        temp4=[]   
        for value in recres.json()["data"]["result"]:
            if str(i) in value["metric"]["pod"]:
                temp4.append(float(value["value"][1]))
        temp4arr=np.array(temp4)
        netrec_avg=temp4arr.mean()
        network_map[str(i)]=netrec_avg
                    
    print("cpu_map:",cpu_map)
    print("thr_map:",thr_map)
    print("mem_map:",mem_map)
    print("network_rec_map:",network_map)
    
    try:
        print("--------解析函数执行日志--------")
        price=get_price(resource_config,n)
        
        latency_map=get_function_latency(function_list,benchmark=benchmark)
            # print("latency_map",latency_map)
        if benchmark=='sequence':
            avg,p95,throughput,total=get_e2e_latency_sequence()
        elif benchmark=='parallel':
            avg,p95,throughput,total=get_e2e_latency_parallel()
        elif benchmark=='branch':
            avg,p95,throughput,total=get_e2e_latency_branch()
        else:
            avg,p95,throughput,total=get_e2e_latency_search()
        
        lst=[]
        lst.append(x)
        for i,k in enumerate(cpu_map):      # 会有空的情况，直接报错
            function_state=[k,resource_config[i][0],resource_config[i][1],resource_config[i][2],int(float(cpu_map[k])*100)/100,int(float(thr_map[k])*100)/100,int(float(mem_map[k])*100)/100,int(float(network_map[k])*100)/100,int(float(latency_map[k][0])*100000)/100000,int(float(latency_map[k][1])*100000)/100000]
            lst.append(function_state)   

        lst.append([avg,p95,throughput,total,price,users]) 
                    
        raw=list(itertools.chain(*lst)) 
                # print(raw)
        # pd.DataFrame([raw]).to_csv(f"/home/user/code/faas-resource/online_step/{benchmark}/{users}/online_tuning_{users}.csv", mode='a', header=False, index=False)
        pd.DataFrame([raw]).to_csv(f"/home/user/code/faas-resource/online_step/{benchmark}/online_tuning.csv", mode='a', header=False, index=False)
    except:
        print("error")
    return lst,avg,p95,throughput,price
    
    

def get_function_latency(function_list,benchmark):
    latency_map={}
    # function=['starter','load','resize','update','resnet']
    for func in function_list:
        os.system("rm -r /home/user/code/faas-resource/function_log/"+benchmark+"/" + func )
        os.system("mkdir /home/user/code/faas-resource/function_log/"+benchmark+"/" + func )
        os.system("kubectl get pod -n openfaas-fn| grep " + func + " | awk '{print $1}' | xargs -I{} sh -c 'kubectl logs {} -n openfaas-fn > /home/user/code/faas-resource/function_log/"+benchmark+"/" + func + "/{}.log'")
        if benchmark=='search':
            pattern=r"\((\d+\.\d+)s\)"     # search的pattern ()代表想提取的东西，\(\)则代表真的括号str
        else:
            pattern=r"Duration: (\d+\.\d+) seconds"
            if func=='mobilenet' or func=='rgb' or func=='resnet':
                pattern=r"Duration: (\d+\.\d+)s"
        for root, dirs, files in os.walk("/home/user/code/faas-resource/function_log/"+benchmark+"/"+func, topdown=False):
            a=[]
            for name in files:
                file_path=os.path.join(root, name)
                with open(file_path,"r") as f:
                    content=f.read()
                    all_times = re.findall(pattern,content)
                    # last_times = all_times[-call_times[i]:]
                    a.append(all_times)
            b=list(itertools.chain(*a))           #flatten
            c = [float(x) for x in b]
            d=np.array(c)
            percent_75=np.percentile(d,75)
            percent_95=np.percentile(d,95)
            latency_map[func]=[percent_75,percent_95]
    #print(latency_map)
    return latency_map

# def get_e2e_latency():
#     df=pd.read_csv("/home/user/code/faas-resource/result1_stats.csv")
#     avg = int(df['Average Response Time'][0])/1000  
#     p95 = df['95%'][0]/1000
#     throughput=df['Request Count'][0]
#     print(avg,p95,throughput)
#     return avg,p95,throughput

# def get_reward():
def get_e2e_latency_sequence():
    #wf1
    response_times = []
    pattern = r'"response_time": ([0-9\.]+)'
    drop=0
    n_line=0
    with open('/home/user/code/faas-resource/locustfile_wf.log') as f:
        for line in f:
            n_line+=1
            if 'beagle' in line:  
                response_time = float(re.search(pattern, line).group(1))  
                response_times.append(response_time)
            else:
                drop+=1
    throughput=n_line-drop
    avg=np.mean(response_times)
    p95=np.percentile(response_times,95)
    print(avg,p95,throughput)
    return avg,p95,throughput,n_line 

def get_e2e_latency_parallel():
    #wf6
    response_times = []
    pattern = r'"response_time": ([0-9\.]+)'
    drop=0
    n_line=0
    with open('/home/user/code/faas-resource/locustfile_wf.log') as f:
        for line in f:
            n_line+=1
            if line.count("Egyptian cat") ==2: 
                response_time = float(re.search(pattern, line).group(1))  
                print(response_time)
                response_times.append(response_time)
            else:
                drop+=1
    throughput=n_line-drop
    avg=np.mean(response_times)
    p95=np.percentile(response_times,95)
    print(avg,p95,throughput,n_line)
    return avg,p95,throughput,n_line


def get_e2e_latency_branch():
    # wf7
    response_times = []
    pattern = r'"response_time": ([0-9\.]+)'
    drop=0
    n_line=0
    with open('/home/user/code/faas-resource/locustfile_wf.log') as f:
        for line in f:
            n_line+=1
            if 'Egyptian cat' in line:
                response_time = float(re.search(pattern, line).group(1))  
                print(response_time)
                response_times.append(response_time)
            else:
                drop+=1
    throughput=n_line-drop
    avg=np.mean(response_times)
    p95=np.percentile(response_times,95)
    print(avg,p95,throughput,n_line)
    return avg,p95,throughput,n_line

def get_e2e_latency_search():
    response_times = []
    pattern = r'"response_time": ([0-9\.]+)'
    drop=0
    n_line=0
    with open('/home/user/code/faas-resource/locustfile_wf.log') as f:
        for line in f:
            n_line+=1
            if 'Success' in line:  
                response_time = float(re.search(pattern, line).group(1))  
                response_times.append(response_time)
            else:
                drop+=1
    throughput=n_line-drop
    avg=np.mean(response_times)
    p95=np.percentile(response_times,95)
    print(avg,p95,throughput,n_line)
    return avg,p95,throughput,n_line 

def single_request_search():   
    TRIP=[{"from": "Shang Hai", "to": "Su Zhou"},
                {"from": "Su Zhou", "to": "Shang Hai"},
                 {"from": "Wu Xi", "to": "Shang Hai"},
                 {"from": "Nan Jing", "to": "Shang Hai"},
                 {"from": "Wu Xi", "to": "Su Zhou"}]
    TRAVEL_DATES = ["2023-12-26","2023-12-27"]
    date=random.choice(TRAVEL_DATES)
    tripinfo=random.choice(TRIP)
    head = {"Accept": "application/json",
            "Content-Type": "application/json"}
    body_start = {
            "startingPlace": tripinfo['from'],
            "endPlace": tripinfo['to'],
            "departureTime": date}
    start_time = time.time()
    response_search = requests.post(
            url="http://33.33.33.132:31112/function/get-left-trip-tickets",
            headers=head,
            json=body_start,
            )
    end_time=time.time()
    print(response_search.content)
    if response_search.status_code==200:
        latency=end_time-start_time
    else:
        latency=10   # failure
    print("latency:",latency)
    # to_log={'response_time':latency,'response':try_to_read_response_as_json(response_search)}
    # log_verbose(to_log)
    return latency

def get_price(resourc_config,n):
    # all_price=[]
    price=0
    for i in range(n):
        cpu_quota=resourc_config[i][0]
        mem_quota=resourc_config[i][1]
        replicas=resourc_config[i][2]
        price_cost=(cpu_quota*CPU_UNIT_COST+mem_quota*MEM_UNIT_COST)*replicas
        print(price_cost)
        # all_price.append(price_cost)
        price+=price_cost
    print(price)
    return price

def test(x,users,benchmark,tunecon):
    if tunecon:
        resource_config=form_x_to_resource_conf(x)
    else:
        resource_config=form_x_to_resource_conf_wo_cl(x)

    if benchmark=='sequence':
        function_list=['starter','load','resize','update','resnet']
    elif benchmark=='parallel':
        function_list=['starter','rgb','resize','update','resnet','mobilenet']
    elif benchmark=='branch':
        function_list=['starter','rgb','resize','update','mobilenet','load','resnet']
    else:
        function_list=['get-left-trip-tickets','get-left-ticket-of-interval','get-price-by-routeid-and-traintype','get-route-by-routeid','get-route-by-tripid',
          'get-sold-tickets','get-traintype-by-traintypeid','get-traintype-by-tripid','query-already-sold-orders','query-config-entity-by-config-name',
          'query-for-station-id-by-station-name','query-for-travel']
      
    rm_functon(function_list)
    time.sleep(40)
    n=len(function_list)
    update_deploy(function_list,resource_config,tunecon)
    time.sleep(20)
    print('finish deployment')
    # for iter in range(10):  # warm
    #     single_request_search()
      
    locustfile="/home/user/code/faas-resource/ml.py"
    # locustfile="/home/user/code/faas-resource/search.py"
    url="http://33.33.33.132:31112/function/wf6"
    # # # search -u 20  -r 10 
    
    spawn_rate =10
    run_time=30
    csv_file="/home/user/code/faas-resource/result1"
    run_locust(locustfile,url,users,spawn_rate,run_time,csv_file)
    time.sleep(20)   #20s延迟最准确的统计
    lst,avg,p95,throughput,price=get_metric(x,resource_config,function_list,users,n,benchmark)
    # price=get_price(resource_config)
    return lst,avg,p95,throughput,price




    
