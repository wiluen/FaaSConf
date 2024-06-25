import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
from sklearn.preprocessing import StandardScaler
# external packages
import numpy as np
import time
import torch
import copy
import logging
import json
from tqdm import tqdm
from RLenv import Environment
# self writing files
import networks
dir_path = os.path.dirname(os.path.realpath(__file__))
handler = logging.FileHandler(os.path.join(dir_path, "log/mf.log"))
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
                batch_size=64,
                # buffer_capacity=3000000,
                buffer_capacity=3478,
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
                explore_noise_decay_rate=0.2):
        super().__init__()

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


     

        # adjency matrix
         self.Gmat=self.Gmat
        # print(self.Gmat)
        # self.Gmat=torch.FloatTensor(self.Gmat).to(self.device)
        
        self.Gmat=torch.FloatTensor(Gmat).unsqueeze(0).to(self.device)
        # self.Gmat=Gmat.repeat(64,1,1).to(self.device)
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
        self.explore_noise=explore_noise
        self.explore_noise_decay=explore_noise_decay
        self.explore_noise_decay_rate=explore_noise_decay_rate
        
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


        # buffer
        # self.n=4082
        # self.n=2047
        wk='20'
        benchmark='branch'
        self.n=1268  # all good 411                  
        self.buffer_capacity=buffer_capacity
        self.buffer_pointer=0
        self.buffer_size=0
        self.function_number=7
        self.a_dmin=4
        self.s_dim=11      #11
        self.simulator=Environment(self.function_number)
  
        self.output_dir=f'/home/user/code/faas-resource/online_step/{benchmark}/mf'
        self.all_s=np.loadtxt(f'/home/user/code/faas-resource/online_step/{benchmark}/state.csv',delimiter=",", dtype=np.float32)       
   

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
        return x

    def update(self):
        actor_loss_sum=0
        critic_loss_sum=0
        

        for batch_count in range(self.update_batch):
            batch_index=np.random.randint(self.n,size=self.batch_size)
            s_batch=torch.FloatTensor(self.buffer_s[batch_index]).to(self.device)
            a_batch=torch.FloatTensor(self.buffer_a[batch_index]).to(self.device)
            s1_batch=torch.FloatTensor(self.buffer_s1[batch_index]).to(self.device)   #s'
            r_batch=torch.FloatTensor(self.buffer_r[batch_index]).to(self.device)
            # end_batch=torch.FloatTensor(self.buffer_end[batch_index]).to(self.device)
            
       
            with torch.no_grad():
               
                s1_avg=torch.bmm(self.Gmat,s1_batch)   
                row_sums = self.Gmat.sum(dim=2)    
                v_s1 = s1_avg / row_sums.unsqueeze(2)   # neibor feature sum/neibor number = neibor's mean feature = mf 
                
                update_Actor_state1_all=torch.concat([s1_batch,v_s1],dim=-1)   #self feature+neibor's mean feature
                update_action1=self.actor_target(update_Actor_state1_all)   
                a1_avg=torch.bmm(self.Gmat,update_action1)   
                v_a1 = a1_avg / row_sums.unsqueeze(2)
                update_action1_all=torch.concat([update_action1,v_a1],dim=-1)  
                Q1=self.critic_target(update_Actor_state1_all,update_action1_all)  #Q(S,A)=Q(sj,sj~,aj,aj~)
                y=r_batch+self.gamma*Q1

               

            s_avg=torch.bmm(self.Gmat,s_batch) 
            v_s = s_avg / row_sums.unsqueeze(2)
            a_avg=torch.bmm(self.Gmat,a_batch)   #att score*s=actor' att s
            v_a = a_avg / row_sums.unsqueeze(2)
            update_Critic_state_all=torch.concat([s_batch,v_s],dim=-1)
            update_action_all=torch.concat([a_batch,v_a],dim=-1)
            Q=self.critic(update_Critic_state_all,update_action_all)
           
            critic_loss=torch.sum(torch.square(y-Q))/self.batch_size
            critic_loss_sum+=critic_loss.cpu().item()

            self.critic_optimizer.zero_grad()
            # self.critic_attention_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                # torch.nn.utils.clip_grad_norm_(self.critic_attention.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
           
            # ----------------critic、critic attention update done--------------------------
            
            update_action=self.actor(update_Critic_state_all)
            a_new_avg=torch.bmm(self.Gmat,update_action)   
            v_a_new = a_new_avg / row_sums.unsqueeze(2)
            update_action_all_new=torch.concat([update_action,v_a_new],dim=-1)

            actor_loss=-torch.sum(self.critic(update_Critic_state_all,update_action_all_new))/self.batch_size
            
            actor_loss_sum+=actor_loss.cpu().item()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
          

                # adding batchnormal
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.soft_replace_rate * param.data + (1 - self.soft_replace_rate) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.soft_replace_rate * param.data + (1 - self.soft_replace_rate) * target_param.data)
        

        critic_loss_mean=critic_loss_sum/self.update_batch
        self.critic_loss_trackor.append(critic_loss_mean)
        actor_loss_mean=actor_loss_sum/self.update_batch
        self.actor_loss_trackor.append(actor_loss_mean)

        tqdm.write(f'Update: Critic Loss {critic_loss_mean} | Actor Loss {actor_loss_mean}')

    def serving(self,users,benchmark):
        for episode in (range(self.max_episode)):
            if self.explore_noise_decay:
                self.explore_noise=self.explore_noise/((episode+1)**self.explore_noise_decay_rate)

            current_state=np.array([0.182048676,0.781688166,0.908404303,0.51531472,8.6,159.88,0.88,8494.39,5.97777,9.88776,20,0.025545876,0.925883898,0.315831585,0.453563092,2.39,85.9,1.25,339.95,12.83281,21.64915,20,0.053242383,0.440246802,0.74617361,0.612663009,1.65,31.74,0.39,883.86,3.86955,4.44621,20,0.664588651,0.175247716,0.897819552,0.814078338,0.5,0,0.65,471.36,0.77266,0.91431,20,0.288605516,0.82228224,0.013934331,0.572644265,11.31,86.68,5.52,8697.17,5.03901,5.36557,20,0.966884706,0.422207308,0.46991345,0.144768207,8.59,4.47,0.77,16716.16,0.75763,0.78942,20,0.994742797,0.250885014,0.629502114,0.799207764,9.8,43.41,18.27,2996.75,1.84989,2.70791,20])    #当前状态       1维
            current_state=self.scale_state(np.array([current_state])).reshape(self.function_number,self.s_dim)

            for step in range(15):     
                print(f'the {step} iterations')        
                with torch.no_grad():
                    state=torch.FloatTensor(current_state).to(self.device).unsqueeze(0)   
                   
                    Actor_attention=self.actor_attention(state,self.Gmat)
                    Actor_state_bar=torch.bmm(Actor_attention,state)
                    Actor_state_all=torch.concat([state,Actor_state_bar],dim=-1)
                    self.actor.eval()
                    self.critic.eval()
                    action=self.actor(Actor_state_all)    
                    print("origin_action:",action)
                    
                    flag=0
                    # SES{
                    # action,price,flag=self.optimal_library(np.array(action[0]))    
                    # action=action.reshape(-1)
                    # }
                    # original{
                    action=np.array(action)
                    action=action.reshape(-1)
                    # action=np.clip(np.random.normal(action, 0.1), 0.1, 0.9) 
                    action=np.clip(action,0.1, 0.9) 
                  
                    # }
                    print('action:',action)                     
                    reward,next_state,avg,p95,throughput,price=self.simulator.step(action,users)
                    print_info = f'--The {step} iteration, change flag:{flag}, concurrrency: {benchmark}-{users}, action: {action}, price: {price}$, avg_e2e_latency: {avg} s, throughput: {throughput},  reward:{reward}'
                    print(print_info)
                    logger.info(print_info)

                next_state=self.scale_state(np.array([next_state])).reshape(self.function_number,self.s_dim)
                # self.buffer_s[self.buffer_pointer]=current_state    
                # self.buffer_s1[self.buffer_pointer]=next_state     
                # self.buffer_a[self.buffer_pointer]=action.reshape(self.function_number,self.a_dim)
                # self.buffer_r[self.buffer_pointer]=np.array(reward).reshape(self.function_number,1)
                # self.buffer_pointer=self.buffer_pointer+1                
                current_state=next_state

    def save_models(self,episode):
        torch.save(self.actor.state_dict(),os.path.join(self.output_dir,f'actor_{episode}.pth'))
        torch.save(self.actor_target.state_dict(),os.path.join(self.output_dir,f'actor_target_{episode}.pth'))

        torch.save(self.actor_attention.state_dict(),os.path.join(self.output_dir,f'actor_attention_{episode}.pth'))
        torch.save(self.actor_attention_target.state_dict(),os.path.join(self.output_dir,f'actor_attention_target_{episode}.pth'))

        torch.save(self.critic.state_dict(),os.path.join(self.output_dir,f'critic_{episode}.pth'))
        torch.save(self.critic_target.state_dict(),os.path.join(self.output_dir,f'critic_target_{episode}.pth'))

        torch.save(self.critic_attention.state_dict(),os.path.join(self.output_dir,f'critic_attention_{episode}.pth'))
        torch.save(self.critic_attention_target.state_dict(),os.path.join(self.output_dir,f'critic_attention_target_{episode}.pth'))

        # with open(os.path.join(self.output_dir,'episode_return.json'),'w') as f:
            # json.dump(str(self.episode_return_trackor),f)
        with open(os.path.join(self.output_dir,'critic_loss.json'),'w') as f:
            json.dump(str(self.critic_loss_trackor),f)
        with open(os.path.join(self.output_dir,'actor_loss.json'),'w') as f:
            json.dump(str(self.actor_loss_trackor),f)

    def load_models(self,episode):
        self.actor.load_state_dict(torch.load(os.path.join(self.output_dir,f'actor_{episode}.pth'),map_location=torch.device('cpu')))
        self.actor_target.load_state_dict(torch.load(os.path.join(self.output_dir,f'actor_target_{episode}.pth'), map_location=torch.device('cpu')))

        # self.actor_attention.load_state_dict(torch.load(os.path.join(self.output_dir,f'actor_attention_{episode}.pth'), map_location=torch.device('cpu')))
        # self.actor_attention_target.load_state_dict(torch.load(os.path.join(self.output_dir,f'actor_attention_target_{episode}.pth'), map_location=torch.device('cpu')))

        self.critic.load_state_dict(torch.load(os.path.join(self.output_dir,f'critic_{episode}.pth'), map_location=torch.device('cpu')))
        self.critic_target.load_state_dict(torch.load(os.path.join(self.output_dir,f'critic_target_{episode}.pth'), map_location=torch.device('cpu')))

        # self.critic_attention.load_state_dict(torch.load(os.path.join(self.output_dir,f'critic_attention_{episode}.pth'), map_location=torch.device('cpu')))
        # self.critic_attention_target.load_state_dict(torch.load(os.path.join(self.output_dir,f'critic_attention_target_{episode}.pth'), map_location=torch.device('cpu')))


if __name__ == '__main__':
    mf=MARL()
    mf.load_models('mf')
    users=20
    benchmark='branch'
    mf.serving(users, benchmark)
  


#经历了奖励的修改和batchnormal之后，动作到了0.5附近了
