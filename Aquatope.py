import scipy
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.get_sampler import SobolQMCNormalSampler
from botorch.test_functions import Hartmann
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import time
import numpy as np
from RLenv import Environment
import logging
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
handler = logging.FileHandler(os.path.join(dir_path, "log/aquatope.log"))
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger = logging.getLogger("Debugging logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
NUM_RESOURCES=3
QOS=12
FUNCTION_NUMBER=5
simulator=Environment(FUNCTION_NUMBER)

benchmark='sequence'
users=25
def get_obj_and_constraint(x,wk,iter):
    # return neg_hartmann6(X)
    
    action=x.numpy()[0]
    print(action)
    reward,next_state,avg,p95,throughput,price=simulator.step(action,users,benchmark,False) 

    print_info = f'-- the {iter} iter, {benchmark}-{users},action: {action}, price: {price}$, avg_e2e_latency: {avg} s, throughput: {throughput}'
        
        # price，avg越小则目标越大
    logger.info(print_info)
    constraint=avg-QOS
    print(price,constraint)
    return torch.tensor([-price]).unsqueeze(-1),torch.tensor([constraint]).unsqueeze(-1)         #看到这里是sample cost




def obj_callable(Z):
    return Z[..., 0]


def constraint_callable(Z):
    return Z[..., 1]

def optimize_acqf_and_get_observation(
    acq_func, bounds, batch_size, num_restarts, raw_samples,wk,iter
):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,         #这里的q是指同时推荐batchsize个配置,但我们无法同时evaluate多个配置,所以q=1
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    new_x= torch.clamp(new_x, min=0.2, max=0.8)  
    print("本轮推荐配置：",new_x)
    new_obj,new_con = get_obj_and_constraint(new_x,wk,iter) # add output dimension  获得目标值，price
  #限制值，其实就是qos
    return new_x, new_obj, new_con

def generate_initial_data(users):
    # generate training data
   
    data=np.loadtxt(f'/home/user/code/faas-resource/GAT-MF/noisebo/{benchmark}{users}-aquatope.csv',delimiter=',')
    train_x=torch.from_numpy(data[:5,:15])
    train_obj=torch.from_numpy(data[:5,-2]).view(-1,1)
    train_con=torch.from_numpy(data[:5,-1]).view(-1,1)
    # best_observed_value = weighted_obj(train_x).max().item()
    return train_x, train_obj, train_con  #, best_observed_value


def initialize_model(train_x, train_obj, train_con, state_dict=None):
    # define models for objective and constraint
    # print(train_x,train_obj,train_con)
    model_obj = SingleTaskGP(train_x, train_obj)
    model_con = SingleTaskGP(train_x, train_con)
    # combine into a multi-output GP model
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    print('---------train model--------------')
    return mll, model

def bo_loop(
    n_init=10,
    n_batch=20,
    mc_samples=32,
    batch_size=1,
    num_restarts=10,
    raw_samples=32,
    infeasible_cost=0,
    anomaly_detection=True,
    confidence=0.95,
    verbose=True,
):
    
    bounds = torch.tensor(
        [[0.0] * NUM_RESOURCES * FUNCTION_NUMBER, [1.0] * NUM_RESOURCES * FUNCTION_NUMBER])

    best_observed_nei, best_random = [], []


    # call helper functions to generate initial training data and initialize model
    (train_x_nei,train_obj_nei,train_con_nei) = generate_initial_data(users)
    print("init data")
    # best_observed_value_nei = best_observed_value
    # best_x = None
    mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei)
        #  初始化模型
    print("init model")    
    # best_observed_nei.append(best_observed_value_nei)
    # best_random.append(best_observed_value)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    # for iteration in range(1, n_batch + 1):   #      开始迭代
    # trace=np.array([50.0, 50.0, 50.0, 55.0, 55.0, 55.0, 47.0, 47.0, 47.0, 49.0, 49.0, 49.0, 57.0, 57.0, 57.0, 39.0, 39.0, 39.0, 24.0, 24.0, 24.0, 26.0, 26.0, 26.0, 55.0, 55.0, 55.0, 33.0, 33.0, 33.0, 41.0, 41.0, 41.0, 87.0, 87.0, 87.0, 55.0, 55.0, 55.0, 50.0, 50.0, 50.0, 52.0, 52.0, 52.0, 55.0, 55.0, 55.0, 65.0, 65.0, 65.0, 54.0, 54.0, 54.0, 56.0, 56.0, 56.0, 54.0, 54.0, 54.0, 51.0, 51.0, 51.0, 56.0, 56.0, 56.0, 47.0, 47.0, 47.0, 51.0, 51.0, 51.0, 61.0, 61.0, 61.0, 59.0, 59.0, 59.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 51.0, 51.0, 51.0])
    # test_trace=trace/2
    # current_state=np.array([0.4,0.825621662,0.623262245,23.84,252.34,0.4,29845.24,5.56959,13.53646,25,0.883504416,0.882439015,0.4,31.41,80,0.6,51739.39,1.17778,1.97851,25,0.4,0.4,0.4,36.59,404.84,4.11,17563.2,11.67873,17.66522,25,0.4,0.4,0.4,23.93,248.45,0.98,13672.75,1.47443,3.27221,25,0.700099189,0.4,0.76061169,33.21,178.49,0.96,7307.37,3.02348,4.37693,25,0.4,0.4,0.4,42.21,488.15,5.99,34812.97,8.52047,19.31171,25])
    #current_state=self.scale_state(current_state.reshape(self.function_number,self.s_dim))
    # tracelen=87

    # for step in range(50,tracelen):     
    for step in range(15):
        
        # users=int(test_trace[step])  
        # next_wk=int(test_trace[step+1])  
        # print(f'the {step} iterations, the workload is {users},next_workload is {next_wk}')

        t0 = time.monotonic()
        # fit the models
        fit_gpytorch_model(mll_nei)
        # sample_shape=(32,21)
        # sample_shape_torch = torch.Size(sample_shape)
        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))  
        # print(qmc_sampler)       # 采样
            #QMC
        # define a feasibility-weighted objective for optimization          原文中的解释是  保持QOS的概率*NEI
              #  constraint 目标值
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable],
            infeasible_cost=infeasible_cost,
        )
        # QMC和NEI 两种uncertainty-aware的贝叶斯中采样和优化的方法 ，这里qEI函数的目标是满足qos
        
       # qnoiseEI 获得函数
        qNEI = qNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=train_x_nei,
            sampler=qmc_sampler,
            objective=constrained_obj,    
        )

        # optimize and get new observation        获得推荐点，获得结果
        new_x_nei, new_obj_nei, new_con_nei = optimize_acqf_and_get_observation(
            acq_func=qNEI,
            bounds=bounds,
            batch_size=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            wk=users,
            iter=step
        )

        # update training points
        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
        train_con_nei = torch.cat([train_con_nei, new_con_nei])

        # remove outliers
        

        # update progress
       
       

    #    把这次采样的点又放进去了，train模型
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            train_con_nei,
            model_nei.state_dict(),
        )

        t1 = time.monotonic()

        
if __name__=='__main__':
    bo_loop()
