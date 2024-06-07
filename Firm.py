import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from torch.nn import BatchNorm1d
import datetime
from RLenv import Environment
import logging
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
handler = logging.FileHandler(os.path.join(dir_path, "log/firm_wkchange.log"))
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger = logging.getLogger("Debugging logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
#####################  hyper parameters  ####################

MAX_EPISODES = 50
MAX_EP_STEPS = 10
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.95  # reward discount
TAU = 0.005  # soft replacement
MEMORY_CAPACITY = 1500
BATCH_SIZE = 32
benchmark='search'
users=20
directory=f"/home/user/code/faas-resource/GAT-MF/firm/{benchmark}/"




###############################  DDPG  ####################################

class ANet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(64, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(64, a_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x


class CNet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 64)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.fca = nn.Linear(a_dim, 64)
        self.fca.weight.data.normal_(0, 0.1)  # initialization
        self.fc = nn.Linear(64, 64)
        self.fc.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(64, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        self.bn1 = nn.BatchNorm1d(64)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x + y)
        net = self.fc(net)
        net = self.bn1(net)
        actions_value = self.out(net)
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.a_dim, self.s_dim = a_dim, s_dim
        # self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
        # self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        # self.scaler = scaler
        
        self.function_number=12
        self.simulator=Environment(self.function_number)
        self.a_dmin=3
        self.s_dim=10
        # self.buffer_a=np.loadtxt('/home/user/code/faas-resource/GAT-MF/experience/image1/ddpg/action_buffer.csv',delimiter=",", dtype=np.float32)
        # self.buffer_r=np.loadtxt('/home/user/code/faas-resource/GAT-MF/experience/image1/ddpg/reward_buffer.csv',delimiter=",", dtype=np.float32).reshape(self.n,1)
        # wk='all'
        # self.buffer_s_origin=np.loadtxt('/home/user/code/faas-resource/model/sequence/10/gatmf/state_all.csv',delimiter=",", dtype=np.float32)
        # scaled_buffer_state=self.scale_state(self.buffer_s_origin)
        # self.buffer_s=scaled_buffer_state
        self.buffer_s_origin=np.loadtxt(f'/home/user/code/faas-resource/GAT-MF/firm/{benchmark}/nextstate_search{users}.csv',delimiter=",", dtype=np.float32)
        # self.buffer_s1_origin=np.loadtxt('/home/user/code/faas-resource/GAT-MF/experience/image1/ddpg/state_next_buffer.csv',delimiter=",", dtype=np.float32)
        # scaled_buffer_state_next=self.scale_state(self.buffer_s1_origin)
        # self.buffer_s1=scaled_buffer_state_next

    def scale_state(self,x):
        standardscaler = StandardScaler()
        scaler=standardscaler.fit(self.buffer_s_origin)    # scaler-> mean,var
        x = scaler.transform(x)
        return x
    
    def choose_action(self, s):  # s []
        s = normalization(s, 1, self.scaler)  # s [[]] for normalize
        s = torch.FloatTensor(s)
        self.Actor_eval.eval()  # when BN or Dropout in testing ################
        # s = torch.unsqueeze(torch.FloatTensor(s), 0)
        act = self.Actor_eval(s)[0].detach()  # ae（s）
        self.Actor_eval.train()  ###############
        return act


    def learn(self,round):
        # for x in self.Actor_target.state_dict().keys():
        #     eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        #
        # for x in self.Critic_target.state_dict().keys():
        #     eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')
     
        for iter in range(round):
            if iter%200==0:
                print(f'train {iter} iters')
            for (target_param,param) in zip(self.Actor_target.parameters(),self.Actor_eval.parameters()):
                target_param.data.copy_(
                    target_param.data*(1-TAU)+param.data*TAU
                )
            for (target_param,param) in zip(self.Critic_target.parameters(),self.Critic_eval.parameters()):
                target_param.data.copy_(
                    target_param.data*(1-TAU)+param.data*TAU
                )


            # indices = np.random.choice(self.pointer, size=BATCH_SIZE)
            # indices=[]
            # for i in range(303):
            #     indices.append(i)
            batch_index=np.random.randint(self.n,size=64)
            bs=torch.FloatTensor(self.buffer_s[batch_index]).to(self.device)
            ba=torch.FloatTensor(self.buffer_a[batch_index]).to(self.device)
            bs_=torch.FloatTensor(self.buffer_s1[batch_index]).to(self.device)   #s'
            br=torch.FloatTensor(self.buffer_r[batch_index]).to(self.device)
            

            a = self.Actor_eval(bs)
            q = self.Critic_eval(bs, a)

            loss_a = -torch.mean(q)

            self.atrain.zero_grad()
            loss_a.backward()
            self.atrain.step()

            a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
            q_ = self.Critic_target(bs_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
            q_target = br + GAMMA * q_  # q_target = 负的
            q_v = self.Critic_eval(bs, ba)
            td_error = self.loss_td(q_target, q_v)
            # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
            # print('critic td_error=',td_error)
            self.ctrain.zero_grad()
            td_error.backward()
            self.ctrain.step()

    def serving(self,users,benchmark):
        agent_number=12
        current_state=np.array([0.8,0.8,0.64062411,18.15,88.54,9.54,7409.67,4.3832,7.54225,20,0.358202992,0.2,0.306629904,20.41,278.64,24.7,7362.66,0.2418,0.44351,20,0.28994719,0.8,0.750424885,2.53,13.8,5.62,1219.59,0.0126,0.02663,20,0.8,0.69306447,0.2,21.96,43.32,9.18,12527.45,0.0058,0.0103,20,0.214844162,0.506682197,0.2,12.19,160.31,11.68,8671.59,0.05732,0.10466,20,0.784428164,0.599292552,0.490237864,4.99,3.15,7.67,2748.8,0.0081,0.0121,20,0.8,0.339434288,0.8,5.65,0.43,12.06,2842.8,0.0114,0.01645,20,0.529000031,0.725408583,0.49978863,5.95,8.18,8.24,5242.74,0.0318,0.04761,20,0.2,0.766296912,0.571430969,2.7,15.7,5.85,1163.21,0.0115,0.01938,20,0.438182836,0.73599337,0.213756483,6.63,2.91,6.86,3693.97,0.0075,0.0106,20,0.8,0.2,0.2,23.3,77.65,27.65,14996.27,0.0066,0.0158,20,0.606030909,0.372541165,0.60626224,6.61,12.19,12.16,3661.73,0.17625,0.33414,20])
        current_state=self.scale_state(current_state.reshape(self.function_number,self.s_dim))
   
        for j in range(15):
            action=[]      #每轮开始置空
            for x in range(self.function_number):        # 智能体共享网络
                state=current_state[x]          
                state=torch.FloatTensor(state).to(self.device)      #unsqueeze加1个维度
                self.Actor_eval.eval()  # when BN or Dropout in testing ################
                a = self.Actor_eval(state.unsqueeze(0))
                # action = np.clip(np.random.normal(a, 0.1), 0.1, 0.9)
                action.append(a.detach()[0])
            arrays = [t.numpy() for t in action]
            action_bar = np.concatenate(arrays, axis=0)
            print("origin_action=",action_bar)
            a = np.clip(np.random.normal(action_bar, 0.05), 0.2, 0.8)
            print("action=",a)
            reward,next_state,avg,p95,throughput,price=self.simulator.step(a,users,benchmark,False) #apply进环境
            
            print_info = f'--The {j} iteration,{benchmark}-{users} price: {price}$,concurrency: {users}action: {a},  avg_e2e_latency: {avg} s,p95:{p95} throughput: {throughput},reward:{reward}'
            current_state=next_state
            
            current_state=np.array([current_state])
            current_state=self.scale_state(current_state.reshape(self.function_number,self.s_dim))
            print(print_info)
            print(current_state)
            logger.info(print_info)

    def changewk(self):
        trace=np.array([50.0, 50.0, 50.0, 55.0, 55.0, 55.0, 47.0, 47.0, 47.0, 49.0, 49.0, 49.0, 57.0, 57.0, 57.0, 39.0, 39.0, 39.0, 24.0, 24.0, 24.0, 26.0, 26.0, 26.0, 55.0, 55.0, 55.0, 33.0, 33.0, 33.0, 41.0, 41.0, 41.0, 87.0, 87.0, 87.0, 55.0, 55.0, 55.0, 50.0, 50.0, 50.0, 52.0, 52.0, 52.0, 55.0, 55.0, 55.0, 65.0, 65.0, 65.0, 54.0, 54.0, 54.0, 56.0, 56.0, 56.0, 54.0, 54.0, 54.0, 51.0, 51.0, 51.0, 56.0, 56.0, 56.0, 47.0, 47.0, 47.0, 51.0, 51.0, 51.0, 61.0, 61.0, 61.0, 59.0, 59.0, 59.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 51.0, 51.0, 51.0])
        test_trace=trace/2
        current_state=np.array([0.4,0.825621662,0.623262245,23.84,252.34,0.4,29845.24,5.56959,13.53646,25,0.883504416,0.882439015,0.4,31.41,80,0.6,51739.39,1.17778,1.97851,25,0.4,0.4,0.4,36.59,404.84,4.11,17563.2,11.67873,17.66522,25,0.4,0.4,0.4,23.93,248.45,0.98,13672.75,1.47443,3.27221,25,0.700099189,0.4,0.76061169,33.21,178.49,0.96,7307.37,3.02348,4.37693,25,0.4,0.4,0.4,42.21,488.15,5.99,34812.97,8.52047,19.31171,25])
        current_state=self.scale_state(current_state.reshape(self.function_number,self.s_dim))
        tracelen=87
        for step in range(tracelen):       
               
            users=int(test_trace[step])  
            next_wk=int(test_trace[step+1])  
            print(f'the {step} iterations, the workload is {users},next_workload is {next_wk}')
            
       
            action=[]      #每轮开始置空
            for x in range(self.function_number):        # 智能体共享网络
                state=current_state[x]          
                state=torch.FloatTensor(state).to(self.device)      #unsqueeze加1个维度
                self.Actor_eval.eval()  # when BN or Dropout in testing ################
                a = self.Actor_eval(state.unsqueeze(0))
                # action = np.clip(np.random.normal(a, 0.1), 0.1, 0.9)
                action.append(a.detach()[0])
            arrays = [t.numpy() for t in action]
            action_bar = np.concatenate(arrays, axis=0)
            a = np.clip(np.random.normal(action_bar, 0.05), 0.3, 0.8)
            print("action=",a)
            reward,next_state,avg,p95,throughput,price=self.simulator.step(a,users) #apply进环境
            
            print_info = f'--The {step} iteration,{benchmark}-{users},next-workload:{next_wk},price: {price}$,concurrency: {users}action: {a},  avg_e2e_latency: {avg} s, throughput: {throughput},reward:{reward}'
            current_state=next_state
            
            current_state=np.array([current_state])
            current_state=self.scale_state(current_state.reshape(self.function_number,self.s_dim))
            print(print_info)
            print(current_state)
            logger.info(print_info)


    def load_model(self, episode):
        model_name_c = "Critic" + str(episode) + ".pt"
        model_name_a = "Actor" + str(episode) + ".pt"
        self.Critic_target = torch.load(directory + model_name_c)
        self.Critic_eval = torch.load(directory + model_name_c)
        self.Actor_target = torch.load(directory + model_name_a)
        self.Actor_eval= torch.load(directory + model_name_a)

    def save_model(self, episode):
        model_name_c = "Critic" + str(episode) + ".pt"
        model_name_a = "Actor" + str(episode) + ".pt"
        torch.save(self.Critic_eval, directory + model_name_c)
        torch.save(self.Actor_eval, directory + model_name_a)
###############################  training  ####################################


def normalization(x,typeid,scaler):
    # x:FloatTensor to array normalize then back to FloatTensor
    if typeid==1:   # choose action (s)
        x=np.array([x])
        x=scaler.transform(x)
    else:   # batch normal(s)
        x=np.array(x)
        x = scaler.transform(x)
        x=torch.FloatTensor(x)
    return x



if __name__=='__main__':
    s_dim = 10
    a_dim = 3
    ddpg = DDPG(a_dim, s_dim)
    ddpg.load_model('search-wk20-5k')
    ddpg.serving(users,benchmark)
    # ddpg.changewk()
    


