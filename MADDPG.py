import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import StandardScaler
import logging
from RLenv import Environment
from model import openai_actor, openai_critic

dir_path = os.path.dirname(os.path.realpath(__file__))
handler = logging.FileHandler(os.path.join(dir_path, "log/maddpg.log"))
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger = logging.getLogger("Debugging logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


lr_a=0.001
lr_c=0.003
batch_size=64
tao=0.05
gamma=0.95
max_grad_norm=0.5
benchmark='branch'
function_number=7
model_file_dir=f'/home/user/code/faas-resource/online_step/{benchmark}/maddpg'
class MADDPG(object):
    def __init__(self):
        super().__init__()
        self.function_number=function_number
        self.s_dim=11
        self.a_dmin=4
        self.n=6056
        self.buffer_capacity=10000
        self.buffer_pointer=0
        self.buffer_size=0
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # self.function_number=12
        # self.a_dmin=3
        # self.s_dim=6
        # self.buffer_a=np.loadtxt('/home/user/code/faas-resource/model/ml/maddpg/dataset/action_buffer_v2.csv',delimiter=",", dtype=np.float32)
        # self.buffer_r=np.loadtxt('/home/user/code/faas-resource/model/ml/maddpg/dataset/reward_buffer_maddpg_v2.csv',delimiter=",", dtype=np.float32)
       
        self.buffer_s_origin=np.loadtxt(f'/home/user/code/faas-resource/online_step/{benchmark}/state.csv',delimiter=",", dtype=np.float32)
        scaled_buffer_state=self.scale_state(self.buffer_s_origin)
        self.buffer_s=scaled_buffer_state

       

        self.simulator=Environment(self.function_number)


    def scale_state(self,x):
        standardscaler = StandardScaler()
        scaler=standardscaler.fit(self.buffer_s_origin)    # scaler-> mean,var
        x = scaler.transform(x)
        return x

        
    def init_trainers(self):
        """
        init the trainers or load the old model
        """

        actors_cur = [None for _ in range(self.function_number)]
        critics_cur = [None for _ in range(self.function_number)]
        actors_tar = [None for _ in range(self.function_number)]
        critics_tar = [None for _ in range(self.function_number)]
        optimizers_c = [None for _ in range(self.function_number)]
        optimizers_a = [None for _ in range(self.function_number)]

        # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
        for i in range(self.function_number):
            actors_cur[i] = openai_actor().to(self.device)
            critics_cur[i] = openai_critic().to(self.device)
            actors_tar[i] = openai_actor().to(self.device)
            critics_tar[i] = openai_critic().to(self.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), lr_a)
            optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), lr_c)
        actors_tar = self.update_target_network(actors_cur, actors_tar, 1.0) # update the target par using the cur
        critics_tar = self.update_target_network(critics_cur, critics_tar, 1.0) # update the target par using the cur
        return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


    def update_target_network(self,agents_cur, agents_tar, tao):
        """
        update the trainers_tar par using the trainers_cur
        This way is not the same as copy_, but the result is the same
        out:
        |agents_tar: the agents with new par updated towards agents_current
        """
        for agent_c, agent_t in zip(agents_cur, agents_tar):
            key_list = list(agent_c.state_dict().keys())
            state_dict_t = agent_t.state_dict()
            state_dict_c = agent_c.state_dict()
            for key in key_list:
                state_dict_t[key] = state_dict_c[key]*tao + \
                                    (1-tao)*state_dict_t[key]
            agent_t.load_state_dict(state_dict_t)
        return agents_tar

    def agents_train(self,actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c,iter):
        obs_size=[]
        action_size=[]
        head_o, head_a, end_o, end_a = 0, 0, 0, 0
        obs_shape_n= [10]*self.function_number
        action_shape_n= [4]*self.function_number
       
        for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
            end_o = end_o + obs_shape
            end_a = end_a + action_shape
            range_o = (head_o, end_o)
            range_a = (head_a, end_a)
            obs_size.append(range_o)
            action_size.append(range_a)
            head_o = end_o
            head_a = end_a
            # update every agent in different memory batch
        for i in range(2000):
            
            loss_actor=[]
            loss_critic=[]
            
            for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                        enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
                   
                batch_index=np.random.randint(self.n,size=batch_size)
                batch_s=torch.FloatTensor(self.buffer_s[batch_index]).to(self.device)
                batch_a=torch.FloatTensor(self.buffer_a[batch_index]).to(self.device)
                batch_s_next=torch.FloatTensor(self.buffer_s1[batch_index]).to(self.device)   #s'
                batch_r=torch.FloatTensor(self.buffer_r[batch_index]).to(self.device)

              
                action_tar = torch.cat([a_t(batch_s_next[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                            for idx, a_t in enumerate(actors_tar)], dim=1)
                   
                q = critic_c(batch_s, batch_a).reshape(-1)       # q(batch_s,batch_a)
                # q(s,a)
                  
                q_ = critic_t(batch_s_next, action_tar).reshape(-1)      # q_(batch_s',batch_a')
                    # all s' and all a_target
                tar_value = q_*gamma + batch_r       # q_*gamma*done + reward

                loss_c = torch.nn.MSELoss()(q, tar_value) # bellman equation
                loss_critic.append(loss_c.cpu().item())
                opt_c.zero_grad()
                loss_c.backward()
                nn.utils.clip_grad_norm_(critic_c.parameters(),max_grad_norm)
                opt_c.step()
                    # all global
            #============================update every critic==========================
                action_i_new = actor_c(batch_s[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]])   #a=actor(batch_s)  
                
                    # self obs，self action
                batch_a[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = action_i_new   
               
                loss_a = torch.mul(-1, torch.mean(critic_c(batch_s, batch_a)))    
                loss_actor.append(loss_a.cpu().item())  
                opt_a.zero_grad()
                loss_a.backward()
                nn.utils.clip_grad_norm_(actor_c.parameters(), max_grad_norm)
                opt_a.step()
            
            if i%50==0:
                print(f'train {i} iters, actor loss: {loss_actor}, critic loss: {loss_critic}')  
                print(action_i_new)
            actors_tar = self.update_target_network(actors_cur, actors_tar, tao)
            critics_tar = self.update_target_network(critics_cur, critics_tar, tao)
        return actors_cur, actors_tar, critics_cur, critics_tar
    
    def save_model(self,actors_cur, actors_tar, critics_cur, critics_tar,iter):
        for agent_idx, (a_c, a_t, c_c, c_t) in enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
            torch.save(a_c.state_dict(), os.path.join(model_file_dir, 'a_c_{}_{}.pt'.format(agent_idx,iter)))
            torch.save(a_t.state_dict(), os.path.join(model_file_dir, 'a_t_{}_{}.pt'.format(agent_idx,iter)))
            torch.save(c_c.state_dict(), os.path.join(model_file_dir, 'c_c_{}_{}.pt'.format(agent_idx,iter)))
            torch.save(c_t.state_dict(), os.path.join(model_file_dir, 'c_t_{}_{}.pt'.format(agent_idx,iter)))
            
            
    def load_model(self,actors_cur, actors_tar, critics_cur, critics_tar,iter):
        print(f'load model {iter} round')
        for agent_idx in range(self.function_number):
            actors_cur[agent_idx].load_state_dict(torch.load(os.path.join(model_file_dir, 'parallel_a_c_{}_{}.pt'.format(agent_idx,iter)),map_location=torch.device('cpu')))
            actors_tar[agent_idx].load_state_dict(torch.load(os.path.join(model_file_dir, 'parallel_a_t_{}_{}.pt'.format(agent_idx,iter)),map_location=torch.device('cpu')))
            critics_cur[agent_idx].load_state_dict(torch.load(os.path.join(model_file_dir, 'parallel_c_c_{}_{}.pt'.format(agent_idx,iter)),map_location=torch.device('cpu')))
            critics_tar[agent_idx].load_state_dict(torch.load(os.path.join(model_file_dir, 'parallel_c_t_{}_{}.pt'.format(agent_idx,iter)),map_location=torch.device('cpu')))

    def serving(self,benchmark,users):
        
        actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c=self.init_trainers()
        self.load_model(actors_cur, actors_tar, critics_cur, critics_tar,'all3000')
        
        for episode in range(1):
            # if self.explore_noise_decay:
            #     self.explore_noise=self.explore_noise/((episode+1)**self.explore_noise_decay_rate)
            current_state=[0.837007029,0.97392916,0.079646526,0.722264321,27.96,157.5,2.51,54189.47,2.7961,3.10467,20,0.602650339,0.957091306,0.262348644,0.392138103,23.45,127.1,0.48,29243.02,1.58683,2.82555,20,0.736038787,0.443195852,0.299986769,0.001767755,6.84,0,1.15,5458.16,0.74177,0.76003,20,0.335977577,0.579846898,0.202353557,0.895821927,17.51,251.83,2.64,6426.88,5.10975,6.95751,20,0.192674326,0.971329383,0.764625056,0.890389062,1.91,38.63,2.9,4903.02,4.45267,7.34593,20,0.99523863,0.355332371,0.500246339,0.316736398,8.24,12.18,0.68,19453.67,0.78884,1.1674,20,0.128732335,0.514587422,0.729740595,0.092225911,7.75,148.98,10.32,3715.16,5.90764,6.2345,20]    #当前状态       1维
            current_state=self.scale_state(np.array([current_state])).reshape(self.function_number,self.s_dim)

            for step in range(15):     
                print(f'the {step} iterations')  
                action_n = np.zeros((self.function_number,4))
                for idx in range(self.function_number):
                    state=torch.FloatTensor(current_state[idx]).to(self.device).unsqueeze(0) 
                    actors_cur[idx].eval()
                    action_n[idx]=actors_cur[idx](state).detach().cpu().numpy()
                    # action_n.append(a)
                # state=torch.FloatTensor(current_state).to(self.device).unsqueeze(0)   
                action=action_n.reshape(-1)
                action=np.clip(np.random.normal(action, 0.1), 0.2, 0.9) 
                print('action=',action)
                # action=action+torch.randn_like(action)*torch.mean(torch.abs(action))*0.3
                
                reward,next_state,avg,p95,throughput,price=self.simulator.step(action,users,benchmark,True) #apply进环境
                # print("state:",next_state)
                print_info = f'--The {step} iteration,{benchmark}-{users}, action: {action}, price: {price}$, avg_e2e_latency: {avg} s, throughput: {throughput}'
                current_state=next_state
                current_state=np.array([current_state])
                current_state=self.scale_state(current_state).reshape(self.function_number,self.s_dim)
                print(print_info)
                logger.info(print_info)
                reward=np.array([reward])
                price=np.array([price])
                instance_price=price.sum()
                total_reward=reward.sum()
                
                print('reward:',total_reward)
                print('instance price:',instance_price)

    def pretrain(self,load_iter):
        actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c=self.init_trainers()
        self.load_model(actors_cur, actors_tar, critics_cur, critics_tar,load_iter)

        actors_cur, actors_tar, critics_cur, critics_tar=self.agents_train(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c,iter)
        self.save_model(actors_cur, actors_tar, critics_cur, critics_tar,3000)


        
if __name__ == '__main__':
    maddpg=MADDPG()
    maddpg.serving(benchmark=benchmark,users=20)
    

       
