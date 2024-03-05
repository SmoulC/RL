import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):# max_size（回放缓冲区的最大大小），input_shape（环境状态的形状），和n_actions（可执行动作的数量）。
        self.mem_size = max_size #mem_size（回放缓冲区的大小）
        self.mem_cntr = 0 #mem_cntr（用于追踪当前存储的记忆数量，即经验的计数器）。
        self.state_memory = np.zeros((self.mem_size, *input_shape)) #state_memory 用于存储状态，其大小为 (max_size, *input_shape)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape)) #new_state_memory 用于存储下一状态，其形状与 state_memory 相同。
        self.action_memory = np.zeros((self.mem_size, n_actions)) #action_memory 用于存储采取的动作，其大小为 (max_size, n_actions)
        self.reward_memory = np.zeros(self.mem_size) #reward_memory 用于存储每个动作的奖励。
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_) #terminal_memory 是一个布尔数组，用于标记每个状态是否为终止状态。

    def store_transition(self, state, action, reward, state_, done): #(存储经验)方法用于在回放缓冲区中存储一个经验样本，包括当前状态、采取的动作、获得的奖励、下一状态以及是否结束
        index = self.mem_cntr % self.mem_size #使用模运算确保索引值不会超过回放缓冲区的最大容量，实现循环覆盖旧经验。
        self.state_memory[index] = state  
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action #将传入的经验样本分别存储到对应的数组中

        self.mem_cntr += 1 #更新已存储经验的计数器

    def sample_buffer(self, batch_size): # sample_buffer 方法从回放缓冲区中随机采样一批经验样本，其大小由参数 batch_size 指定。
        max_mem = min(self.mem_cntr, self.mem_size) #计算可以采样的最大经验数量，以避免采样到未初始化的记忆部分

        batch = np.random.choice(max_mem, batch_size)# 随机选择一批经验的索引

        states = self.state_memory[batch] 
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch] # 根据选定的索引批量提取状态、动作、奖励、下一状态和终止标志。

        return states, actions, rewards, states_, dones #返回采样的经验批次。

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
            name, chkpt_dir='tmp/td3'):#这定义了一个名为 CriticNetwork 的类，它有几个参数：学习率 beta，输入维度 input_dims，第一和第二全连接层的维度 fc1_dims 和 fc2_dims，动作的数量 n_actions，网络的名称 name，和检查点目录 chkpt_dir。
        super(CriticNetwork, self).__init__() #这一行调用了基类 nn.Module 的构造函数，是在创建任何PyTorch模型时必须做的。
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3') # 这一部分代码初始化类属性，包括网络的配置和检查点文件的路径。

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims) 
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        self.q1 = nn.Linear(self.fc2_dims, 1) #定义了三个全连接层：fc1 是第一层，接受状态和动作作为输入；fc2 是第二层；q1 是输出层，输出单个值表示状态-动作对的价值（Q值）。

        self.optimizer = optim.Adam(self.parameters(), lr=beta) #使用Adam优化器来优化网络参数，学习率为 beta
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') #初始化优化器，这里使用了 Adam 优化器，并将网络配置到适当的设备上（CUDA设备如果可用，否则使用CPU）。

        self.to(self.device) #将网络的所有参数和缓冲区移动到指定的设备

    def forward(self, state, action): #定义了网络的前向传播方法，它接收状态和动作作为输入
        
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value) #将状态和动作拼接起来，通过第一个全连接层，然后应用ReLU激活函数；接着通过第二个全连接层，再次应用ReLU激活函数

        q1 = self.q1(q1_action_value) #通过输出层计算Q值

        return q1

    def save_checkpoint(self): #定义了一个方法来保存网络的参数到一个检查点文件
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file) #使用PyTorch的 save 方法保存网络的状态字典到指定的检查点文件。

    def load_checkpoint(self): #定义了一个方法来从检查点文件加载网络的参数
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file)) #使用PyTorch的 load 方法读取检查点文件，并通过 load_state_dict 方法加载这些参数到网络中。

class ActorNetwork(nn.Module): #接收几个参数来配置网络：学习率(alpha)、输入维度(input_dims)、两个全连接层的维度(fc1_dims 和 fc2_dims)、动作数量(n_actions)、网络名称(name)以及检查点文件保存的目录(chkpt_dir)。
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
            n_actions, name, chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3') #这些行将传入的参数保存为类实例的属性，并构造检查点文件的路径。

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions) #这里定义了三个全连接层（nn.Linear）。第一层将输入维度转换为 fc1_dims，第二层将 fc1_dims 转换为 fc2_dims，最后一层将 fc2_dims 转换为动作维度 n_actions。

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) #使用Adam优化器来优化网络参数，学习率为 alpha
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') #确定模型应该运行在GPU还是CPU上。如果CUDA可用，将使用第一个CUDA设备，否则使用CPU。

        self.to(self.device)#将网络模块及其所有参数移动到指定的设备上（GPU或CPU）。

    def forward(self, state): #定义网络的前向传播逻辑。state 是输入状态。
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob) #输入状态通过两个全连接层，中间使用ReLU激活函数。

        mu = T.tanh(self.mu(prob)) #最后，输出层的结果通过tanh激活函数，以产生一个介于-1和1之间的动作值（假设动作空间已经被归一化）。

        return mu

    def save_checkpoint(self):#定义一个方法来保存当前网络的状态到一个文件中。
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)#使用PyTorch的 save 函数将网络的状态字典保存到指定的检查点文件。

    def load_checkpoint(self): #定义一个方法来从文件中加载网络状态
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            gamma=0.99, update_actor_interval=2, warmup=1000,
            n_actions=2, max_size=1000000, layer1_size=400,
            layer2_size=300, batch_size=100, noise=0.1):
        """这是Agent类的构造函数，用于初始化代理的各个参数和网络。这里的参数包括：
                alpha 和 beta：分别是Actor和Critic网络的学习率。
                input_dims：环境观察值的维度。
                tau：目标网络软更新的参数。
                env：代理将要交互的环境。
                gamma：折扣因子，用于计算未来奖励的当前价值。
                update_actor_interval：Actor网络更新频率。
                warmup：预热步数，在此期间代理采取随机动作，以填充经验回放缓冲区。
                n_actions：动作空间的维度。
                max_size：经验回放缓冲区的最大容量。
                layer1_size 和 layer2_size：神经网络中两个隐藏层的大小。
                batch_size：从经验回放缓冲区中采样的批量大小。
                noise：添加到动作中的探索噪声。"""
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions) # 初始化经验回放缓冲区，用于存储代理与环境交互的经验
        self.batch_size = batch_size
        #self.learn_step_cntr 用于跟踪代理进行学习（即调用 learn 方法）的次数,如果设置了每隔2次学习步骤更新一次Actor网络（通过update_actor_interval参数控制），那么这个计数器就可以帮助判断是否到了更新Actor网络的时候。
        self.learn_step_cntr = 0 
        self.time_step = 0 #self.time_step 用于记录代理与环境交互的总步数
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        #初始化Actor网络，用于生成动作
        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='actor') 
        
        #初始化两个Critic网络，用于评估动作的价值。TD3算法使用两个Critic网络来减少Q值估计的偏差。
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='critic_2') 
        
        #初始化目标网络。目标网络是原网络的复制，用于稳定训练过程。目标网络参数定期从原网络软更新而来。
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                    layer2_size, n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                layer2_size, n_actions=n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                layer2_size, n_actions=n_actions, name='target_critic_2')
        

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup: #在预热期（self.time_step < self.warmup）内，动作是随机生成的。这是通过从一个以 self.noise 为标准差的正态分布中随机抽样来实现的。这个过程有助于在学习开始阶段填充经验回放缓冲区，并增加初始的探索。
            mu = T.tensor(np.random.normal(scale=self.noise, 
                                            size=(self.n_actions,)))
        
        #预热期结束后，代理使用其Actor网络来选择动作。首先，当前观察值（observation）被转换为张量（T.tensor），并发送到Actor网络所在的设备（CPU或GPU）。然后，通过Actor网络的前向传递（forward 方法）来生成动作值 mu。
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        
        # 为了增加探索性，生成的动作 mu 会加上一些噪声。这个噪声同样是从一个正态分布中抽样得到的，其标准差为 self.noise。这样做可以防止代理在学习过程中过早收敛到局部最优解。
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                    dtype=T.float).to(self.actor.device)

        # 接着，使用 T.clamp 函数将加噪声后的动作值 mu_prime 限制在环境允许的动作范围内（即 self.min_action 到 self.max_action 之间）。这是必要的步骤，以确保生成的动作对环境有效。
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1 #每次调用 choose_action 方法时，时间步 self.time_step 都会递增。这个计数器用于跟踪代理与环境交互的总次数，以及控制何时结束预热期。

        return mu_prime.cpu().detach().numpy()
    
    
    def remember(self, state, action, reward, new_state, done):# 这行定义了 remember 方法，它接收五个参数：state（当前状态），action（代理采取的动作），reward（执行动作后获得的奖励），new_state（执行动作后的新状态），和 done（一个布尔值，表示是否达到终止状态）
        self.memory.store_transition(state, action, reward, new_state, done)#这行代码调用了代理中 memory 属性（一个 ReplayBuffer 实例）的 store_transition 方法，将传入的经验元组保存到经验回放缓冲区中。store_transition 方法负责在缓冲区中找到正确的位置存储这些值，并管理缓冲区的大小，确保当缓冲区满时，能够按照某种策略（如覆盖最早的记录）进行处理。

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:#检查是否有足够的样本进行学习
            return 
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)#从经验回放缓冲区中随机采样一批数据，每个数据包括状态、动作、奖励、下一个状态和结束标志。

        # 将采样的数据转换为PyTorch张量，并移动到相应的设备上（GPU或CPU）。
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        
        #生成目标动作并添加噪声
        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5) #确保添加到动作上的噪声值在 [-0.5, 0.5] 的范围内
        target_actions = T.clamp(target_actions, self.min_action[0], 
                                self.max_action[0])#确保添加到动作上的噪声值在 action 的范围内
        
        #使用目标Critic网络和带噪声的目标动作来评估下一个状态的Q值。
        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        #如果结束了Q就设为零
        q1_[done] = 0.0
        q2_[done] = 0.0

        # 不管 q1_ 和 q2_ 原来的形状是什么，.view(-1) 都会将它们展平成一个一维向量，其中包含了所有原始元素
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_) #函数计算两个预测中的逐元素最小值，生成一个新的张量

        target = reward + self.gamma*critic_value_ #计算目标Q值
        target = target.view(self.batch_size, 1)

        #计算当前Critic网络的Q值与目标Q值之间的均方误差损失，然后反向传播误差，更新Critic网络的权重
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau#用于软更新目标网络的参数
        
        #使用.named_parameters()获取网络参数，这个方法返回一个生成器，包含参数的名称和值。
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        #将参数名称和值对存储在字典中，方便后续按名称更新。
        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        #以下循环实现了软更新机制：对每个参数，计算一个新值，该值是当前网络参数和目标网络参数的加权平均。clone()方法用于复制参数，以避免在原地修改导致的问题。
        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                    (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                    (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                    (1-tau)*target_actor[name].clone()

        #更新目标网络的参数：使用.load_state_dict()方法将计算出的新参数设置给目标网络。
        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()