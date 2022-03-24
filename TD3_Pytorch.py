from turtle import forward
from xml.etree.ElementTree import tostring
import gym
import random
import torch
import torch.nn as TNN
import torch.nn.functional as TF
import numpy as np
from collections import deque
import torch.utils.data as Data
import matplotlib.pyplot as plt

#参数定义
ENV_NAME = 'MountainCarContinuous-v0'
BUFFER_SIZE = 1000000

GAMMA = 0.99
BATCHSIZE = 100
TEST = 5
SAVINGPATH = "./modelTD3/"
EPISODE = 10000
TAU = 0.005
STEP = 200
D = 2

# OU噪声生成
class OrnsteinUhlenbeckActionNoise:
	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return torch.Tensor(self.X)

# ReplayBuffer
class ReplayBuffer():
    def __init__(self,env,buffersize):
        self.buffer = deque(maxlen=buffersize)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
    def append(self,content):
        self.buffer.append([content[0],content[1],content[2],content[3]])

# 参数化策略网络
# 输入为状态state，输出为确定的动作
class Policy_Net(TNN.Module):
    def __init__(self,env):
        super().__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.bound = env.action_space.high
        self.bound_low = env.action_space.low
        self.net = TNN.Sequential()
        self.net.add_module("fc1",TNN.Linear(self.state_dim,30))
        self.net.add_module("relu1",TNN.ReLU())
        self.net.add_module("fc2",TNN.Linear(30,20))
        self.net.add_module("relu2",TNN.ReLU())
        self.net.add_module("fc3",TNN.Linear(20,self.action_dim))
        self.net.add_module("tanh1",TNN.Tanh())
        # 初始化最后一层
        TNN.init.uniform_(self.net.fc3.weight,a=-0.003,b=0.003)
        TNN.init.uniform_(self.net.fc3.bias,a=-0.003,b=0.003)
        # 初始化OU噪声
        self.OU_Noise = OrnsteinUhlenbeckActionNoise(self.action_dim)
        self.optimizer = torch.optim.Adam([{'params':self.net.parameters()}],lr=0.001)
        
    def forward(self,x):
        Actions = self.net(x)
        return Actions
    
    def action(self,state):
        state = torch.Tensor(state)
        a=self.forward(state)
        return a * self.bound.item()

    #带 Gauss 或 OU 噪声的action
    def action_with_noise(self,state,noise_type='Gauss'):
        state = torch.Tensor(state)
        if noise_type == 'Gauss':
            ret = self.action(state) + torch.normal(mean = torch.zeros(1), std = 0.1)
        elif noise_type == 'OU':
            ret = self.action(state) + self.OU_Noise.sample()
        return ret

# Critic Net
# 输入为状态state和动作action，输出为价值
class Critic_Net(TNN.Module):
    def __init__(self,env,target=False) :
        super().__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0] 

        # 隐层网络定义
        self.fc1 = TNN.Linear(self.state_dim+self.action_dim,20)
        self.fc2 = TNN.Linear(20,20)
        self.fc3 = TNN.Linear(20,1)

        if target:
            pass
        else:
            # 初始化最后一层
            TNN.init.uniform_(self.fc3.weight,a=0.0003,b=0.0003)
            TNN.init.uniform_(self.fc3.bias,a=0.0003,b=0.0003)
            self.train_setup()

    def forward(self,state,action):
        cat = torch.cat((state,action),axis=1)
        h1 = TF.relu(self.fc1(cat))
        
        h2 = TF.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out

    # 定义训练优化器和损失函数
    def train_setup(self):
        self.optimizer = torch.optim.Adam([
                    {'params':self.fc1.parameters()},
                    {'params':self.fc2.parameters()},
                    {'params':self.fc3.parameters()}],lr=0.001)
        self.loss_fn = TNN.MSELoss()

    # 根据targetQ和targetU生成的y训练Critic网络
    def update_criticNet(self,tensor_state,tensor_action,tensor_y):
        x = self.forward(tensor_state,tensor_action)
        loss = self.loss_fn(tensor_y,x)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

# 更新target网络，输入tau为更新权重
def update_tNet(Net,Net_t,tau):
    # 用于target网络的初始化
    if tau == 0:
        Net_t.load_state_dict(Net.state_dict())
        return
    
    state_dict = Net.state_dict()
    state_dict_t = Net_t.state_dict()
    #print(state_dict)
    for key in state_dict:
        state_dict_t[key] = state_dict[key]*tau + (1-tau)*state_dict_t[key]
    Net_t.load_state_dict(state_dict_t)

# 采样minibatch
# 生成标签 y 及 minibatch
def sample_Batch(replaybuffer,Q_t1,Q_t2,u_t,batch_size=BATCHSIZE):
    # 根据BATCHSIZE采样
    minibatch = random.sample(replaybuffer.buffer,batch_size)
    
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    #将数据取出转换为成为张量
    tensor_state = torch.Tensor(state_batch).reshape(batch_size,Q_t1.state_dim)
    tensor_action = torch.Tensor(action_batch).reshape(batch_size,Q_t1.action_dim)
    tensor_r = torch.Tensor(reward_batch).reshape(batch_size,1)
    tensor_nextstate = torch.Tensor(next_state_batch).reshape(batch_size,Q_t1.state_dim)

    tensor_u_t = u_t.net(tensor_nextstate) + torch.clamp(torch.normal(mean=torch.zeros(1),std=0.2),-0.5,0.5)# 加上不相关的高斯噪声
    # 将数据送入Q_target计算
    # 取Q1Q2输出最小的作为y值
    tensor_y = tensor_r + GAMMA * torch.min(Q_t1(tensor_nextstate,tensor_u_t),Q_t2(tensor_nextstate,tensor_u_t))

    return tensor_y,tensor_action,tensor_state

# 从replaybuffer中采样，完成算法的一次迭代
def update_Network(replaybuffer,Q1,Q2,Q_t1,Q_t2,u,u_t):
    # 如果replaybuffer数据量小于BATCHSIZE则不更新
    if len(replaybuffer.buffer)<BATCHSIZE:
        return 
    # 形成 minibatch
    tensor_y,tensor_action,tensor_state  = sample_Batch(replaybuffer,Q_t1,Q_t2,u_t)

    ## update Critic网络
    Q1.update_criticNet(tensor_state,tensor_action,tensor_y)
    Q2.update_criticNet(tensor_state,tensor_action,tensor_y)
    ## update Actor网络
    # 使用Actor生成动作 u_action
    # 每D次更新actor和target网络
    if len(replaybuffer.buffer)% D ==0:
        u_action = u(tensor_state)
        # 将采样的状态state和Actor生成的动作输入Critic网络计算评判结果
        loss_u_Grad = - Q1(tensor_state,u_action)
        # 损失函数为取平均，对Q的评判结果的平均值取负，因为要做的是梯度上升
        #loss_u_Grad =   Q_critic
        loss_u_Grad = loss_u_Grad.mean()
        
        # 将反向传播通路上的Q网络的节点梯度清零
        Q1.optimizer.zero_grad()
        # 反向传播
        u.optimizer.zero_grad()
        loss_u_Grad.backward()
        u.optimizer.step()

        ### update target网络
        ## update Q_target网络
        update_tNet(Q1,Q_t1,TAU)
        update_tNet(Q2,Q_t2,TAU)
        ## 更新u_target网络
        update_tNet(u,u_t,TAU)



def main():
    Mode = 'Train'
    env = gym.make(ENV_NAME)
    
    # 创建网络
    Q1 = Critic_Net(env)
    Q2 = Critic_Net(env)
    u = Policy_Net(env)
    # target
    Q_t1 = Critic_Net(env)
    Q_t2 = Critic_Net(env)
    u_t = Policy_Net(env)

    # 初始化target
    update_tNet(Q1,Q_t1,0)
    update_tNet(Q2,Q_t2,0)
    update_tNet(u,u_t,0)

    replaybuffer = ReplayBuffer(env,BUFFER_SIZE)
    if Mode == 'Train':
        ave_reward = []
        for episode in range(EPISODE):
            # 初始化环境
            state = env.reset()
            acc_reward = 0 #累积奖赏
            # 训练
            for step in range(STEP):
                # 产生动作
                action = u.action_with_noise(state)
                # 观测
                next_state,reward,done,_ = env.step(action.detach().numpy())
                replaybuffer.append([state,action,reward,next_state])
                # 更新四个网络
                update_Network(replaybuffer,Q1,Q2,Q_t1,Q_t2,u,u_t)
                state = next_state
                if done : 
                    break
            if episode%100 == 0:#测试
                acc_reward = 0
                for i in range(TEST):
                    state = env.reset()
                    for step in range(STEP):
                        action = u.action(state)
                        state,reward,done,_ = env.step(action.detach().numpy())
                        acc_reward += reward
                        if done:
                            break
                ave_reward.append(acc_reward/TEST)
                print('episode: ',episode,'Evaluation Average Reward:',acc_reward/TEST)
                if acc_reward/TEST >93:
                    #保存
                    torch.save(u.state_dict(),SAVINGPATH+"model-"+str(acc_reward/TEST)+".pth")                
                    break
            
    else :
        u.load_state_dict(torch.load("modelDDPG\model-94.23124390777878.pth"))
        state = env.reset()
        acc_reward = 0
        while True:
            action = u.action(state)
            state,reward,done,_ = env.step(action.detach().numpy())
            acc_reward += reward
            if done:
                break
        print("total reward: {}".format(acc_reward))

if __name__ == '__main__':
    main()