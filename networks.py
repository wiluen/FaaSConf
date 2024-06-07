import torch
import torch.nn.functional as F
from torch import nn

state_dim=10 #11
action_dim=3 #4
func_number=12
# state_dim=6
# action_dim=3
Actor_in_features=2*state_dim
# Actor_hidden_features1=32
# Actor_hidden_features2=16
hidden_features=64
# Actor_hidden_features2=128
Critic_in_features=2*(state_dim+action_dim)
# Critic_hidden_features1=32
# Critic_hidden_features2=16

Attention_in_features=state_dim
Attention_hidden_features=32

class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()

        self.lin1=nn.Linear(in_features=Actor_in_features,out_features=hidden_features)
        self.lin2=nn.Linear(in_features=hidden_features,out_features=hidden_features)
        self.lin3=nn.Linear(in_features=hidden_features,out_features=action_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.drop1=nn.Dropout()
        # self.drop2=nn.Dropout()

    def forward(self,s):
        s= s.view(-1, state_dim*2)   # 用到bn 对整个s维度都用，这个是mean field所以是10*2=20  22
        s=self.lin1(s)
        s=F.relu(s)
        s=self.bn1(s)
        # s=self.drop1(s)
        s=self.lin2(s)
        s=F.relu(s)
        s=self.bn2(s)
        # s=self.drop2(s)
        s=self.lin3(s)
        s=torch.sigmoid(s)
        s = s.view(-1,func_number, action_dim)
        return s

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()

        self.lin1=nn.Linear(in_features=Critic_in_features,out_features=hidden_features)
        self.lin2=nn.Linear(in_features=hidden_features,out_features=hidden_features)
        self.lin3=nn.Linear(in_features=hidden_features,out_features=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self,s,a):
        x=torch.concat((s,a),dim=2)
        x= x.view(-1, (state_dim+action_dim)*2)   #(11+4)*2  
        x=self.lin1(x)
        x=F.relu(x)
        x=self.bn1(x)
        x=self.lin2(x)
        x=F.relu(x)
        x=self.bn2(x)
        x=self.lin3(x)
        x = x.view(-1,func_number, 1)
        return x

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        # 初始化注意力机制中的 Q 和 K 权重矩阵   后面的意思是将张量的值限定在一定范围，减少梯度消失或者爆炸
        self.Qweight=nn.Parameter(torch.rand(Attention_in_features,Attention_hidden_features)*((4/Attention_in_features)**0.5)-(1/Attention_in_features)**0.5)
        self.Kweight=nn.Parameter(torch.rand(Attention_in_features,Attention_hidden_features)*((4/Attention_in_features)**0.5)-(1/Attention_in_features)**0.5)
        # K,Q shape  (6(state),32(att_feature)) 
    def forward(self,s,Gmat):
         # 计算 Q 和 K 矩阵
        # s=64*12*6  
        q=torch.einsum('ijk,km->ijm',s,self.Qweight)                          # batch*12*32
        k=torch.einsum('ijk,km->ijm',s,self.Kweight).permute(0, 2, 1)         # batch*32*12
         # 计算注意力分数
        att=torch.square(torch.bmm(q,k))*Gmat    # batch*12*12
         # 归一化注意力分数
        att=att/(torch.sum(att,dim=2,keepdim=True)+0.001)

        return att

# class Attention(nn.Module):
#     def __init__(self):
#         super(Attention,self).__init__()
#         self.scaling_factor = Attention_hidden_features ** 0.5
#         self.Qweight = nn.Parameter(torch.rand(Attention_in_features, Attention_hidden_features) * ((4/Attention_in_features)**0.5) - (1/Attention_in_features)**0.5)
#         self.Kweight = nn.Parameter(torch.rand(Attention_in_features, Attention_hidden_features) * ((4/Attention_in_features)**0.5) - (1/Attention_in_features)**0.5)
     
#     def forward(self,s,Gmat):
#         q = torch.einsum('ijk,km->ijm',s,self.Qweight)
#         k = torch.einsum('ijk,km->ijm',s,self.Kweight).permute(0, 2, 1)

#         # Scaled dot product
#         att_raw = torch.bmm(q,k) / self.scaling_factor
         
#         # Apply mask with Gmat before softmax
#         att_raw_masked = att_raw * Gmat   

#          # Apply softmax along the last dimension of att_raw_masked.
#          # Add a small value to the denominator for numerical stability.
#          # Softmax is applied to normalize the weights.
#         att= F.softmax(att_raw_masked,dim=-1)     #这里归一化之后，就不是0和1了

#         return att