import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
from sklearn.preprocessing import StandardScaler
# external packages
import numpy as np
import time
import pandas as pd
import torch
import copy
import random
import json
import logging
# from tqdm import tqdm
from RLenv import Environment
# self writing files
import networks
# from grid_simulator import grid_model
NUM_RESOURCES=3
# CPU_MIN=250
# CPU_MAX=2000
# MEM_MIN=256
# MEM_MAX=2048
CPU_MIN=250
CPU_MAX=1000
MEM_MIN=128
MEM_MAX=1024
REP_MIN=1
REP_MAX=8
CONCURR_MIN=1
CONCURR_MAX=10
CPU_UNIT_COST=0.000173   #0.173/1000
MEM_UNIT_COST=0.000012    #0.0123/1024
N=2 # nsigma in ses
dir_path = os.path.dirname(os.path.realpath(__file__))
handler = logging.FileHandler(os.path.join(dir_path, "log/gatmf_sequence.log"))
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger = logging.getLogger("Debugging logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
class MARL(object):
    def __init__(self,
                num_grid=10,
                num_diamond=2,
                diamond_extent=10,
                num_move=0.1,
                max_steps=10,
                max_episode=1,
                update_batch=100,
                batch_size=8,
                buffer_capacity=4200,
                update_interval=100,
                save_interval=1000,
                lr=0.0001,
                lr_decay=False,
                grad_clip=False,
                max_grad_norm=10,
                soft_replace_rate=0.01,
                gamma=0.9,
                explore_noise=0.1,
                explore_noise_decay=True,
                explore_noise_decay_rate=0.2,
                update_fre=3,
                function_number=12):
        super().__init__()
        '''
        code from: https://github.com/tsinghua-fib-lab/Large-Scale-MARL-GATMF/blob/main/code/grid_train.py
        '''
        
        print('Initializing...')

        # generate config
        config_data=locals()
        del config_data['self']
        del config_data['__class__']
        time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config_data['time']=time_data

        # environment
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(9)
        if self.device=='cuda':
            torch.cuda.manual_seed(9)
        np.random.seed(9)

        self.num_grid=num_grid
        self.num_diamond=num_diamond
        self.diamond_extent=diamond_extent
        self.num_move=num_move


        Gmat_tckt=[[0,1,0,1,0,0,1,0,1,0,1,1],
              [1,0,0,0,1,1,0,1,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,1],
              [1,0,0,0,1,0,0,0,0,0,0,1],
              [0,1,0,1,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,1,0,0,0,1],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [1,0,0,0,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,0,0,0,0,1],
              [1,0,1,1,0,0,1,0,0,0,1,0]]
        self.Gmat=torch.FloatTensor(Gmat_tckt).to(self.device)
        # learning parameters
        self.max_steps=max_steps
        self.max_episode=max_episode
        self.update_batch=update_batch
        self.batch_size=batch_size
        self.update_interval=update_interval
        self.save_interval=save_interval
        self.lr=lr
        self.lr_decay=lr_decay
        self.grad_clip=grad_clip
        self.max_grad_norm=max_grad_norm
        self.soft_replace_rate=soft_replace_rate
        self.gamma=gamma
        self.update_fre=update_fre
        self.explore_noise=explore_noise
        self.explore_noise_decay=explore_noise_decay
        self.explore_noise_decay_rate=explore_noise_decay_rate
        self.function_number=function_number
        self.simulator=Environment(self.function_number)

        # networks and optimizers
        self.actor=networks.Actor().to(self.device)
        # self.actor.load_state_dict(torch.load(os.path.join('..','model','30grid_2022-09-14_22-35-29','actor_92400.pth')))
        self.actor_target=copy.deepcopy(self.actor).eval()
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=0.0001)

        self.critic=networks.Critic().to(self.device)
        self.critic_target=copy.deepcopy(self.critic).eval()
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=0.0001)

        self.actor_attention=networks.Attention().to(self.device)
        # self.actor_attention.load_state_dict(torch.load(os.path.join('..','model','30grid_2022-09-14_22-35-29','actor_attention_92400.pth')))
        self.actor_attention_target=copy.deepcopy(self.actor_attention).eval()
        self.actor_attention_optimizer=torch.optim.Adam(self.actor_attention.parameters(),lr=0.0001)

        self.critic_attention=networks.Attention().to(self.device)
        self.critic_attention_target=copy.deepcopy(self.critic_attention).eval()
        self.critic_attention_optimizer=torch.optim.Adam(self.critic_attention.parameters(),lr=0.0001)

        self.buffer_capacity=300
        # self.buffer_pointer=n
        # # self.buffer_size=0
        
        self.a_dim=3
        self.s_dim=10
        self.buffer_s=np.empty((self.buffer_capacity,self.function_number,self.s_dim),dtype=np.float32) # S,C,D,Sdiff,Cdiff,Ddiff
        self.buffer_a=np.empty((self.buffer_capacity,self.function_number,self.a_dim),dtype=np.float32)
        self.buffer_s1=np.empty((self.buffer_capacity,self.function_number,self.s_dim),dtype=np.float32) # S,C,D,Sdiff,Cdiff,Ddiff
        self.buffer_r=np.empty((self.buffer_capacity,self.function_number,1),dtype=np.float32) 

        # buffer_a=np.loadtxt('/home/user/code/faas-resource/GAT-MF/experience/image1/v2/action_buffer_v2.csv',delimiter=",", dtype=np.float32).reshape(n,self.function_number,self.a_dim)
        # buffer_r=np.loadtxt('/home/user/code/faas-resource/GAT-MF/experience/image1/v2/reward_buffer_maddpg_v2_gatmf.csv',delimiter=",", dtype=np.float32).reshape(n,self.function_number,1)
        wk=20
        benchmark='search'
        self.output_dir=f'/home/user/code/faas-resource/online_step/{benchmark}/all/'
        self.all_s=np.loadtxt(f'/home/user/code/faas-resource/online_step/{benchmark}/nextstate.csv',delimiter=",", dtype=np.float32)       
        self.lib=pd.read_csv(f"/home/user/code/faas-resource/online_step/{benchmark}/20/optimal_a_20.csv",header=None).values
        self.n=len(self.lib)
        self.buffer_pointer=self.n

        # ======================================online memory pool=========================================
        self.optimal_a=np.loadtxt(f'/home/user/code/faas-resource/online_step/{benchmark}/20/optimal_a_20.csv',delimiter=",", dtype=np.float32).reshape(self.n,self.function_number,self.a_dim)
        self.optimal_r=np.loadtxt(f'/home/user/code/faas-resource/online_step/{benchmark}/20/optimal_r_20.csv',delimiter=",", dtype=np.float32).reshape(self.n,self.function_number,1)
        
        self.buffer_s_origin=np.loadtxt(f'/home/user/code/faas-resource/online_step/{benchmark}/20/optimal_s_20.csv',delimiter=",", dtype=np.float32)
        scaled_buffer_state=self.scale_state(self.buffer_s_origin)
        self.optimal_s=scaled_buffer_state.reshape(self.n,self.function_number,self.s_dim)

        self.buffer_s1_origin=np.loadtxt(f'/home/user/code/faas-resource/online_step/{benchmark}/20/optimal_s1_20.csv',delimiter=",", dtype=np.float32)
        scaled_buffer_state_next=self.scale_state(self.buffer_s1_origin)
        self.optimal_s1=scaled_buffer_state_next.reshape(self.n,self.function_number,self.s_dim)

        self.buffer_a[:self.n,:,]=self.optimal_a
        self.buffer_r[:self.n,:,]=self.optimal_r
        self.buffer_s[:self.n,:,]=self.optimal_s
        self.buffer_s1[:self.n,:,]=self.optimal_s1
    #    =======================================================================================================
        # training trackors
        self.episode_return_trackor=list()
        self.episode_price_trackor=list()
        self.episode_avgtime_trackor=list()
        self.episode_p95_trackor=list()
        self.episode_throughput_trackor=list()
        # self.critic_loss_trackor=list()
        # self.actor_loss_trackor=list()


    def get_action(self,action):
        action_vector=torch.softmax(action,dim=-1)

        return action_vector

    def get_entropy(self,action):
        weight=torch.softmax(action,dim=1)
        action_entropy=-torch.sum(weight*torch.log2(weight))

        return action_entropy
    
    def scale_state(self,x):
        standardscaler = StandardScaler()
        scaler=standardscaler.fit(self.all_s)    # scaler-> mean,var
        x = scaler.transform(x)
        return x           #  这是全矩阵，没有做reshape操作

    def update(self):
        actor_loss_sum=0
        critic_loss_sum=0
        # n=4082
        # n=2047
        print("online finetune...")
        for batch_count in range(1):
            batch_index=np.random.randint(self.buffer_pointer,size=64)
            s_batch=torch.FloatTensor(self.buffer_s[batch_index]).to(self.device)
            a_batch=torch.FloatTensor(self.buffer_a[batch_index]).to(self.device)
            s1_batch=torch.FloatTensor(self.buffer_s1[batch_index]).to(self.device)   #s'
            r_batch=torch.FloatTensor(self.buffer_r[batch_index]).to(self.device)
            # end_batch=torch.FloatTensor(self.buffer_end[batch_index]).to(self.device)
            
        #  注意力的方法是：计算注意力分数 + 注意力动作（状态） + 拼接      都是根据state来进行注意力打分，且只进行一次，actor，critic分别算一次
            with torch.no_grad():
                # 通过s'计算a’来计算Q目标
                update_Actor_attention1=self.actor_attention_target(s1_batch,self.Gmat)   #计算出actor注意力分数
                # Gmat是邻接矩阵，对角线是0，GCN中邻接矩阵对角线是1，效果是信息传递，这里是0表示了邻居的状态和动作
                update_Actor_state1_bar=torch.bmm(update_Actor_attention1,s1_batch)   #注意力分数*状态=actor注意力状态
                # actor认为的注意力state
                update_Actor_state1_all=torch.concat([s1_batch,update_Actor_state1_bar],dim=-1)   #拼接原来状态和actor注意力状态
                update_action1=self.actor_target(update_Actor_state1_all)   #输出actor注意力动作
                # 这是DDPG中的 a'=a_t(s')
                # ===========================分割线================================
                update_Critic_attention1=self.critic_attention_target(s1_batch,self.Gmat)  #计算critic注意力分数
                update_Critic_state1_bar=torch.bmm(update_Critic_attention1,s1_batch)   #注意力分数*状态=critic注意力状态
                # critic认为的注意力state
                update_Critic_state1_all=torch.concat([s1_batch,update_Critic_state1_bar],dim=-1)   #拼接原状态和critic注意力状态★★★★★
                update_action1_bar=torch.bmm(update_Critic_attention1,update_action1)   #critic注意力分数*actor注意力动作=被critic修改的actor注意力动作
                # critic认为的注意力action
                update_action1_all=torch.concat([update_action1,update_action1_bar],dim=-1)  #拼接actor注意力动作和critic修改的actor注意力动作★★★★★
                Q1=self.critic_target(update_Critic_state1_all,update_action1_all)  #Q(S,A)=Q(sj,sj~,aj,aj~)
                # 这是DDPG中的 q'=c_t(s',a')
                y=r_batch+self.gamma*Q1

            # 通过a和s来计算Q
            update_Critic_attention=self.critic_attention(s_batch,self.Gmat)
            update_Critic_state_bar=torch.bmm(update_Critic_attention,s_batch)
            update_Critic_state_all=torch.concat([s_batch,update_Critic_state_bar],dim=-1)
            update_action_bar=torch.bmm(update_Critic_attention,a_batch)
            update_action_all=torch.concat([a_batch,update_action_bar],dim=-1)
            Q=self.critic(update_Critic_state_all,update_action_all)
            # 这是DDPG中的 q=c(s,a)   这里的s和a都要来算    
            critic_loss=torch.sum(torch.square(y-Q))/self.batch_size
            critic_loss_sum+=critic_loss.cpu().item()

            self.critic_optimizer.zero_grad()
            self.critic_attention_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_attention.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            self.critic_attention_optimizer.step()
            # critic和critic attention网络都是一样的参数更新
            # ----------------critic、critic attention更新完毕--------------------------
            
            update_Actor_attention=self.actor_attention(s_batch,self.Gmat)   #state在actor下的注意力分数★★★★★  s应该是全局状态，s*adj*att 就是带权重消息传递，将邻居的特征赋予自己
            update_Actor_state_bar=torch.bmm(update_Actor_attention,s_batch)  #actor注意力分数*状态=actor注意力状态
            update_Actor_state_all=torch.concat([s_batch,update_Actor_state_bar],dim=-1)   #拼接注意力状态和原来状态
            update_action=self.actor(update_Actor_state_all)
            # print(update_action)
            #DDPG中的a=a(s)

            with torch.no_grad():
                update_Critic_attention_new=self.critic_attention(s_batch,self.Gmat)
                update_Critic_state_bar_new=torch.bmm(update_Critic_attention_new,s_batch)
                update_Critic_state_all_new=torch.concat([s_batch,update_Critic_state_bar_new],dim=-1)

            update_action_bar_new=torch.bmm(update_Critic_attention_new,update_action)   #critic注意力分数*actor注意力动作
            update_action_all_new=torch.concat([update_action,update_action_bar_new],dim=-1)

            actor_loss=-torch.sum(self.critic(update_Critic_state_all_new,update_action_all_new))/self.batch_size
            # DDPG中的 q=c(s,a(s))来对actor做loss
            actor_loss_sum+=actor_loss.cpu().item()

            self.actor_optimizer.zero_grad()
            self.actor_attention_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor_attention.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.actor_attention_optimizer.step()
            # ----------------actor、actor attention更新完毕-------------------------------

            # target网络软更新
             # adding batchnormal
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.soft_replace_rate * param.data + (1 - self.soft_replace_rate) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.soft_replace_rate * param.data + (1 - self.soft_replace_rate) * target_param.data)
            for param, target_param in zip(self.actor_attention.parameters(), self.actor_attention_target.parameters()):
                target_param.data.copy_(self.soft_replace_rate * param.data + (1 - self.soft_replace_rate) * target_param.data)
            for param, target_param in zip(self.critic_attention.parameters(), self.critic_attention_target.parameters()):
                target_param.data.copy_(self.soft_replace_rate * param.data + (1 - self.soft_replace_rate) * target_param.data)



    def optimal_library(self,action): 
        # read from json
        mean=4.574 
        std=0.34
        upper=mean+N*(std)    
        lower=mean-N*(std)       
        price_total=0
        x=self.form_x_to_resource_conf(action)
        flag=0
        # 全都是分组的
        for i in range(self.function_number):
            cpu_quota=x[i][0]
            mem_quota=x[i][1]
            replicas=x[i][2]
            # exec_time=lst[x][8]   #P75
            price_cost=(cpu_quota*CPU_UNIT_COST+mem_quota*MEM_UNIT_COST)*replicas          # 范围是0.x
            # local_reward=price_cost*exec_time
            price_total+=price_cost
        print("origin_price:",price_total)

        if price_total>upper or price_total<lower:
            print("change action from history")
            flag=1
            random_num = random.randint(0, self.n-30)
            new_action = self.lib[random_num] #一维
            new_action=np.clip(np.random.normal(new_action, 0.05), 0.1, 0.9)   # add noise for exploration
        else:
            new_action=action #5维度 
        return new_action,price_total,flag

    

    def form_x_to_resource_conf(self,x):
        # num_functions = int(len(x)/NUM_RESOURCES)
        resource_config = [[0, 0, 0] for _ in range(self.function_number)]
        for i in range(self.function_number):
            scaled_cpu = x[i][0]
            scaled_memory = x[i][1]
            scaled_replicas = x[i][2]
            # scaled_concurrency = x[i][3]     
            resource_config[i][0] = round(scaled_cpu * (CPU_MAX - CPU_MIN) + CPU_MIN, 0)
            resource_config[i][1] = round(scaled_memory * (MEM_MAX - MEM_MIN) + MEM_MIN, 0)
            resource_config[i][2] = int(scaled_replicas * (REP_MAX - REP_MIN) + REP_MIN)
            # resource_config[i][3] = int(scaled_concurrency * (CONCURR_MAX - CONCURR_MIN) + CONCURR_MIN)    

        return resource_config
    
    def serving(self,users,benchmark):
        for episode in (range(self.max_episode)):
            if self.explore_noise_decay:
                self.explore_noise=self.explore_noise/((episode+1)**self.explore_noise_decay_rate)

            current_state=np.array([0.8,0.614686238,0.8,13.51,49.1,11.21,5616.13,6.78004,13.56775,20,0.1,0.542030923,0.8,8.58,202.14,9.92,3403.81,0.38957,0.8684,20,0.754826881,0.8,0.146714001,4.41,0,6.02,1839.15,0.01042,0.01571,20,0.165390143,0.629255265,0.8,8.5,85.54,8.2,4576.43,0.0095,0.04565,20,0.1,0.8,0.1,18.69,445.69,9.67,17810.66,0.1111,0.29033,20,0.381411753,0.8,0.578906349,3.12,8.31,5.93,1690.18,0.0088,0.01551,20,0.120386161,0.575556375,0.421334538,6.53,109.33,8.5,3790.98,0.0115,0.05945,20,0.22973606,0.256979935,0.8,4.8,60,16.94,3044.77,0.0498,0.11285,20,0.1,0.61383462,0.8,2.21,37.06,6.7,841.63,0.0127,0.10244,20,0.502366936,0.325631157,0.8,2.69,3.3,10.79,1385.84,0.0104,0.02147,20,0.723735507,0.34549336,0.699064294,11.24,6.79,19.95,4409.08,0.0086,0.01396,20,0.794115817,0.8,0.774768331,5.18,13.41,6.69,2491.09,0.19022,0.32098,20])
            current_state=self.scale_state(np.array([current_state])).reshape(self.function_number,self.s_dim)

            for step in range(15):     
                print(f'the {step} iterations')        
                with torch.no_grad():
                    state=torch.FloatTensor(current_state).to(self.device).unsqueeze(0)   
                    #环境加噪声  10次 选argmax Q
                    Actor_attention=self.actor_attention(state,self.Gmat)
                    Actor_state_bar=torch.bmm(Actor_attention,state)
                    Actor_state_all=torch.concat([state,Actor_state_bar],dim=-1)
                    self.actor.eval()
                    self.critic.eval()
                    t1=time.time()
                    action=self.actor(Actor_state_all)     #做出动作
                    t2=time.time()
                    print(t2-t1)
                    print("origin_action:",action)
                    
                    flag=0
                    # SES{
                    action,price,flag=self.optimal_library(np.array(action[0]))    
                    action=action.reshape(-1)
                    # }
                    # original{
                    # action=np.array(action)
                    # action=action.reshape(-1)
                    # action=np.clip(np.random.normal(action, 0.1), 0.1, 0.9) 
                   
                    # }
                    print('action:',action)                      # 一维的，在env中会处理
                    reward,next_state,avg,p95,throughput,price=self.simulator.step(action,users,benchmark,False) #apply进环境
                    print_info = f'--The {step} iteration, change flag:{flag}, concurrrency: {benchmark}-{users}, action: {action}, price: {price}$, avg_e2e_latency: {avg} s,p95: {p95} throughput: {throughput},  reward:{reward}'
                    print(print_info)
                    logger.info(print_info)

                next_state=self.scale_state(np.array([next_state])).reshape(self.function_number,self.s_dim)
                self.buffer_s[self.buffer_pointer]=current_state    #reshape+scale过
                self.buffer_s1[self.buffer_pointer]=next_state     
                self.buffer_a[self.buffer_pointer]=action.reshape(self.function_number,self.a_dim)
                self.buffer_r[self.buffer_pointer]=np.array(reward).reshape(self.function_number,1)
                self.buffer_pointer=self.buffer_pointer+1

                
                current_state=next_state
                if step%self.update_fre==0:
                    self.update()


    def changewk(self):
        trace=np.loadtxt('trace.txt')
        test_trace=trace/2
        current_state=[0.838142539,0.970959035,0.4,0.4,17.59,80.19,0.33,37088.09,1.482,1.96262,25,0.67133974,0.4,0.455369653,0.4,16.35,98.67,1.53,30910.28,2.10704,3.24618,25,0.4,0.960003813,0.791675072,0.4,13.24,137.39,0.45,5155.75,1.80063,4.21682,25,0.4,0.696556939,0.4,0.4,24.23,235.23,1.49,10590.49,2.83481,4.22374,25,0.808651857,0.997559128,0.471737027,0.4,19.33,117.45,7.22,4029.37,3.98749,5.45568,25,0.4,0.4,0.4,0.407458128,28.92,317.82,12.73,22084.86,7.34592,8.21903,25]
        current_state=self.scale_state(np.array([current_state])).reshape(self.function_number,self.s_dim)
        tracelen=len(trace)
        for step in range(tracelen):     
               
            users=int(test_trace[step])  
            next_wk=int(test_trace[step+1])  
            print(f'the {step} iterations, the workload is {users},next_workload is {next_wk}')
            with torch.no_grad():
                state=torch.FloatTensor(current_state).to(self.device).unsqueeze(0)   
                #环境加噪声  10次 选argmax Q
                Actor_attention=self.actor_attention(state,self.Gmat)
                Actor_state_bar=torch.bmm(Actor_attention,state)
                Actor_state_all=torch.concat([state,Actor_state_bar],dim=-1)
                self.actor.eval()
                self.critic.eval()
                action=self.actor(Actor_state_all)     #做出动作
                print("origin_action:",action)
                
                flag=0
                # SES{
                action,flag=self.ses(np.array(action[0]),users)    
                action=action.reshape(-1)
                # }
                # original{
                # action=np.array(action)
                # action=action.reshape(-1)
                # action=np.clip(np.random.normal(action, 0.1), 0.1, 0.9) 
                # action=np.clip(action,0.1,0.9)
                # }
                print('action:',action)                     
                reward,next_state,avg,p95,throughput,price=self.simulator.step_load(action,users,next_wk) #apply进环境,
                print_info = f'--The {step} iteration, change flag:{flag}, concurrrency: parallel-{users},, action: {action}, price: {price}$, avg_e2e_latency: {avg} s, throughput: {throughput},  reward:{reward}'
                print(print_info)
                logger.info(print_info)

            next_state=self.scale_state(np.array([next_state])).reshape(self.function_number,self.s_dim)
            
            self.buffer_s[self.buffer_pointer]=current_state    #reshape+scale过
            self.buffer_s1[self.buffer_pointer]=next_state     
            self.buffer_a[self.buffer_pointer]=action.reshape(self.function_number,self.a_dim)
            self.buffer_r[self.buffer_pointer]=np.array(reward).reshape(self.function_number,1)
            self.buffer_pointer=self.buffer_pointer+1

            current_state=next_state
            self.update()


            self.reward_trackor.append(reward)
            self.price_trackor.append(price)
            self.avgtime_trackor.append(avg)
            self.throughput_trackor.append(throughput)
            if step==(len(trace)-1):
                with open(os.path.join(self.output_dir,'reward_trackor.json'),'w') as f:
                    json.dump(str(self.reward_trackor),f)
                with open(os.path.join(self.output_dir,'price_trackor.json'),'w') as f:
                    json.dump(str(self.price_trackor),f)
                with open(os.path.join(self.output_dir,'avgtime_trackor.json'),'w') as f:
                    json.dump(str(self.avgtime_trackor),f)
                with open(os.path.join(self.output_dir,'throughput_trackor.json'),'w') as f:
                    json.dump(str(self.throughput_trackor),f)
   

    def save_models(self,episode):
        torch.save(self.actor.state_dict(),os.path.join(self.output_dir,f'actor_{episode}.pth'))
        torch.save(self.actor_target.state_dict(),os.path.join(self.output_dir,f'actor_target_{episode}.pth'))

        torch.save(self.actor_attention.state_dict(),os.path.join(self.output_dir,f'actor_attention_{episode}.pth'))
        torch.save(self.actor_attention_target.state_dict(),os.path.join(self.output_dir,f'actor_attention_target_{episode}.pth'))

        torch.save(self.critic.state_dict(),os.path.join(self.output_dir,f'critic_{episode}.pth'))
        torch.save(self.critic_target.state_dict(),os.path.join(self.output_dir,f'critic_target_{episode}.pth'))

        torch.save(self.critic_attention.state_dict(),os.path.join(self.output_dir,f'critic_attention_{episode}.pth'))
        torch.save(self.critic_attention_target.state_dict(),os.path.join(self.output_dir,f'critic_attention_target_{episode}.pth'))

     

    def load_models(self,episode):
        self.actor.load_state_dict(torch.load(os.path.join(self.output_dir,f'actor_{episode}.pth')))
        self.actor_target.load_state_dict(torch.load(os.path.join(self.output_dir,f'actor_target_{episode}.pth')))

        self.actor_attention.load_state_dict(torch.load(os.path.join(self.output_dir,f'actor_attention_{episode}.pth')))
        self.actor_attention_target.load_state_dict(torch.load(os.path.join(self.output_dir,f'actor_attention_target_{episode}.pth')))

        self.critic.load_state_dict(torch.load(os.path.join(self.output_dir,f'critic_{episode}.pth')))
        self.critic_target.load_state_dict(torch.load(os.path.join(self.output_dir,f'critic_target_{episode}.pth')))

        self.critic_attention.load_state_dict(torch.load(os.path.join(self.output_dir,f'critic_attention_{episode}.pth')))
        self.critic_attention_target.load_state_dict(torch.load(os.path.join(self.output_dir,f'critic_attention_target_{episode}.pth')))


if __name__ == '__main__':
    train_platform=MARL()  
    workload=20
    benchmark='search'
    train_platform.load_models('ttall-20')
    train_platform.serving(workload,benchmark)

