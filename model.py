import torch
import torch.nn as nn
import torch.nn.functional as F

class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()

    def act(self, input):
        policy, value = self.forward(input) # flow the input through the nn
        return policy, value

class actor_agent(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)
        self.reset_parameters()
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain_tanh)

    def forward(self, input):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        policy = self.tanh(self.linear_a(x))
        return policy

class critic_agent(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, args.num_units_1)
        self.linear_a_c1 = nn.Linear(action_shape_n, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1*2, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)
        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_o = self.LReLU(self.linear_o_c1(obs_input))
        x_a = self.LReLU(self.linear_a_c1(action_input))
        x_cat = torch.cat([x_o, x_a], dim=1)
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class openai_critic(abstract_agent):
    def __init__(self):
        super(openai_critic, self).__init__()
        # (s_dmin+a_dmin)*5=14*5=70  15*7=105
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(105, 40)
        self.linear_c2 = nn.Linear(40, 40)
        self.linear_c = nn.Linear(40, 1)
        self.bn1 = nn.BatchNorm1d(40)     # norm for feature
        self.bn2 = nn.BatchNorm1d(40)
        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x=self.bn1(x_cat)
        x = self.LReLU(self.linear_c2(x))
        x=self.bn2(x)
        value = self.linear_c(x)
        return value

class openai_actor(abstract_agent):
    def __init__(self):
        super(openai_actor, self).__init__()
        self.tanh= nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(11, 40)
        self.linear_a2 = nn.Linear(40, 40)
        self.linear_a = nn.Linear(40,4)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.bn0 = nn.BatchNorm1d(10) 
        self.bn1 = nn.BatchNorm1d(40)     # norm for feature
        self.bn2 = nn.BatchNorm1d(40)

    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x=self.linear_a1(input)
        x = self.LReLU(x)
        x=self.bn1(x)
        x=self.linear_a2(x)
        x = self.LReLU(x)
        x=self.bn2(x)
        x=self.linear_a(x)
        x = torch.sigmoid(x)
        return x