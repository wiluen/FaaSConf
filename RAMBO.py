from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
from RLenv import Environment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import os
import logging
import pandas as pd

'''
rely on bayes_opt lib
reivse bayse_optimization.py and target_space.py to load data for warm-up,  class targetspace from bayes.py
'''
dir_path = os.path.dirname(os.path.realpath(__file__))
handler = logging.FileHandler(os.path.join(dir_path, "log/rambo.log"))
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger = logging.getLogger("Debugging logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
QOS=15
AVG_PRICE=6 
func_number=6
benchmark='search'
users=30
class RAMBO(object):
    def __init__(self):
        super().__init__()
        self.simulator=Environment(func_number)

        print('Initializing...')
 
        self.pbounds = {
        'x1':(0,1),'x2':(0,1),'x3':(0,1),'x4':(0,1),'x5':(0,1),'x6':(0,1),'x7':(0,1),'x8':(0,1),'x9':(0,1),'x10':(0,1),
        'x11':(0,1),'x12':(0,1),'x13':(0,1),'x14':(0,1),'x15':(0,1),'x16':(0,1),'x17':(0,1),'x18':(0,1)}
        

        self.optimizer = BayesianOptimization(
            f=self.f, 
            pbounds=self.pbounds, 
            random_state=42, 
            verbose=2 
        )
        
        self.optimizer.set_gp_params(
            alpha=1e-5, 
            n_restarts_optimizer=2, 
        )

      
        #trick2 cloud noise
        self.utility = UtilityFunction(kind='ei', kappa=2.5, xi=0)

    def f(self,x,users,step):
        a=[x['x1'],x['x2'],x['x3'],x['x4'],x['x5'],x['x6'],x['x7'],x['x8'],x['x9'],
                x['x10'],x['x11'],x['x12'],x['x13'],x['x14'],x['x15'],x['x16'],x['x17'],x['x18']]
                # ,x['x19'],x['x20'],x['x21'],x['x22'],x['x23'],x['x24'],x['x25'],x['x26'],
                # x['x27'],x['x28'],x['x29'],x['x30'],x['x31'],x['x32'],x['x33'],x['x34'],x['x35'],x['x36']]
      
        action=np.array(a)
        action=np.clip(action,0.2,0.8)
        
        print(action)
        reward,next_state,avg,p95,throughput,price=self.simulator.step(action,users,benchmark,False) 
        
        print_info = f'-- the {step} iter,{benchmark}-{users},action: {action}, price: {price}$, avg_e2e_latency: {avg} s, throughput: {throughput}'
        print(print_info)
        scale_price=(AVG_PRICE-price)/AVG_PRICE
        scale_qos=(QOS-avg)/QOS
      
        logger.info(print_info)

        return (scale_price+2*scale_qos)
    
  
    def run(self,users):
        
        for step in range(30):
            print(f"-------{step} iteraition--------")
            next_point = self.optimizer.suggest(self.utility)
            
            target = self.f(next_point,users,step)
            self.optimizer.register(params=next_point, target=target)
            
        print('best optimaztion:',self.optimizer.max)
        x=[]
        for i, res in enumerate(self.optimizer.res):
            x.append(res['target'])

    def run_wkchange(self):
        trace=np.loadtxt('trace.txt')
        tracelen=len(trace)
        for step in range(tracelen):
            users=int(trace[step])  
            next_wk=int(trace[step+1])  
            print(f'the {step} iterations, the workload is {users},next_workload is {next_wk}')
    
            next_point = self.optimizer.suggest(self.utility)
            
            target = self.f(next_point,users,step)
            self.optimizer.register(params=next_point, target=target)
            
        print('best optimaztion:',self.optimizer.max)
        x=[]
        for i, res in enumerate(self.optimizer.res):
            x.append(res['target'])
if __name__=='__main__':
    bo=RAMBO()
    bo.run(users)


