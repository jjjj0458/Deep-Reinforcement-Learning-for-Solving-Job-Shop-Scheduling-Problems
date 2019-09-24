# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 15:32:14 2018

@author: banana
"""
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.conv1 = nn.Conv2d(3, s_dim, (1,2), padding=0)
        self.conv2 = nn.Conv2d(3, s_dim, (1,2), padding=0)
        self.pi1 = nn.Linear(s_dim*10*9, 100)
        self.pi2 = nn.Linear(100, a_dim)
        self.v1 = nn.Linear(s_dim*10*9, 100)
        self.v2 = nn.Linear(100, 1)
        set_init([self.conv1,self.conv2,self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical
        self.explore = True

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c1 = c1.view(c1.size(0),-1)
        c2 = c2.view(c1.size(0),-1)
        

        pi1 = F.relu(self.pi1(c1))
        logits = self.pi2(pi1)
        v1 = F.relu(self.v1(c2))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        if np.random.randint(100) < 10 and self.explore :
#            act = np.random.randint(self.a_dim,size = m.sample().numpy().size)
            act = m.sample().numpy()
        else:
            act = m.sample().numpy()
        return act

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss
    
##x = torch.randn(100, 100)
##a = v_wrap(np.repeat(1,100))
#r = torch.randn(2,1)
#net = Net(26,2)
##net.loss_func(x,a,r)
#a = v_wrap(net.choose_action(v_wrap(state)))
#state = np.array([[active_m,machine,process_time],[active_m,machine,process_time]])
#
#opt = SharedAdam(net.parameters(), lr=0.0001) 
#loss = net.loss_func(v_wrap(state),a,r)
#opt.zero_grad()
#loss.backward()
#opt.step()
#
#m = nn.Conv2d(3, 10, (1,2), padding=0)
#c = m(v_wrap(state))
#c.shape
#(2 - 1)//2
