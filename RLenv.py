from util import test
import pandas as pd
import numpy as np
import itertools

CPU_UNIT_COST=0.000173   #0.173/1000
MEM_UNIT_COST=0.000012    #0.0123/1024
class Environment:
    def __init__(self,function_number):
        super(Environment,self).__init__()
        self.function_number=function_number

    def step(self,action,users,benchmark,tunecon):
        lst,avg,p95,throughput,price=test(action,users,benchmark,tunecon)
        s_=self.get_state(action,lst,users,tunecon)
        s_=np.array([s_]).reshape(-1)
        # print(lst)
        # print(s_)
        # r=0
        # s_=0
        r=self.get_reward_v3(avg,throughput,price)
        return r,s_,avg,p95,throughput,price

    def step_load(self,action,users,next_wk):
        lst,avg,p95,throughput,price=test(action,users)
        print(lst)
        s_=self.get_state(action,lst,next_wk)
        s_=np.array([s_]).reshape(-1)
        # r=0
        # s_=0
        r=self.get_reward_v3(avg,throughput,price)
        return r,s_,avg,p95,throughput,price

    # def reset(self):
    def get_state(self,action,lst,users,tunecon):
        # state=[]
        # for x in range(self.function_number):
        #     state.append(lst[x][4:10])
        if tunecon:
            action_dim=4
        else:
            action_dim=3
        state=[]
        state_bar=[]
        print(lst)
        for x in range(self.function_number):
            conf=action[action_dim*x:action_dim*(x+1)]  #4
            metric=lst[x+1][4:10]  # 4:10
            state.append(conf.tolist())
            state.append(metric)
            state.append([users])
        state_bar = [item for sublist in state for item in sublist]  # 拉平
        # statenp=np.array(state)
        # state_bar=statenp.flatten()
        print('state=',state_bar)
        return state_bar 

 

     
    def get_reward(self,lst,avg,p95,throughput):
        # with local reward and global reward
        AVG_P75_TIME=[0.194,3.51,0.0118,0.0083,0.0383,0.0082,0.01,0.0372,0.011,0.0088,0.0109,0.2245]
        AVG_LATENCY=3.5
        AVG_THROUGHPUT=110
        reward=[]
        price=[]
        a=0.2
        b=0.7
        c=0.3
        w1=0.5
        w2=0.5
        global_reward= b * AVG_LATENCY / avg + c * throughput / AVG_THROUGHPUT 
        for x in range(12):
            cpu_quota=lst[x][1]
            mem_quota=lst[x][2]
            replicas=lst[x][3]
            exec_time=lst[x][8]   #P75
            price_cost=(cpu_quota*CPU_UNIT_COST+mem_quota*MEM_UNIT_COST)*replicas      
            price.append(price_cost)
            local_reward=-price_cost+a*(AVG_P75_TIME[x]/exec_time)
            r=w1 * local_reward + w2 * global_reward
            reward.append(r)
        return reward,price
    
 

    def get_reward_v2(self,avg,throughput,price):
        #reward on paper
        AVG_LATENCY=13
        AVG_THROUGHPUT=20
        total_price=0
        reward=[]
        w1=0.1
        c=0
        if avg>AVG_LATENCY or throughput<AVG_THROUGHPUT:
            c=-0.5
        for x in range(self.function_number):
            r=-(price*w1)+c
            reward.append(r)
        return reward  
    
    def reset(self):
        # ramdon init state from history
        s=np.array([0.2,0.2,0.51489439,0.429013539,7.81,160.06,0.89,15111.85,5.00457,6.52569,10,0.757932541,0.980655166,0.2,0.493473136,16.17,30.33,0.58,38834.19,0.92137,0.99779,10,0.2,0.648902473,0.2,0.2,18.11,288.28,1.8,10346.23,2.43701,3.43862,10,0.2,0.2,0.477144285,0.2,9.57,176.74,1.08,4188.39,2.57374,2.89927,10,0.2,0.279550445,0.2,0.999088274,40.47,631.76,28.04,7429.33,19.04801,20.23789,10,0.900729804,0.820119297,0.906154674,0.696318457,7.51,32.38,0.36,8166.25,1.06297,1.58645,10])
        return s
