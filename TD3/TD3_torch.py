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
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool) #terminal_memory 是一个布尔数组，用于标记每个状态是否为终止状态。

    def store_transition(self, state, action, reward, state_, done): #(存储经验)方法用于在回放缓冲区中存储一个经验样本，包括当前状态、采取的动作、获得的奖励、下一状态以及是否结束
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones