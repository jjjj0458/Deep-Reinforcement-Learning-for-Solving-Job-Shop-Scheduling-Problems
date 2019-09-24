# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:52:42 2018

@author: banana
"""
import copy
import numpy as np
from net2015 import Net
from shared_adam import SharedAdam
from utils import v_wrap
from la02 import optt
import torch
import time 
#np.loadtxt("C:/Users/banana/Desktop/RFCODE/la02.txt")

class job_shop_env():
    def __init__(self,ins = "ft06"):
        f = open("C:/Users/banana/Desktop/RFCODE/"+ins+".txt", 'r', encoding = 'utf-8-sig')
        x = f.readlines()
        f.close()
        
        y = [i.split() for i in x]
        z = [list(map(int,j)) for j in y]
        
        
        m_array = np.array(z)
        self.machine = m_array[:,0::2]
        self.machine_num  = m_array.shape[1]//2
        self.job_num = m_array.shape[0]
        self.process_time = m_array[:,1::2]
        self.process_time_ori = m_array[:,1::2]
        ##
        self.active_m = np.zeros((self.machine.shape)).astype(int)
        self.machine_status = np.repeat(np.nan,self.machine_num)
        self.machine_run = np.repeat(-1,self.machine_num)
        self.doing = np.repeat(np.nan,self.machine_num)
        self.job_dic={}
        self.oper_dic={}
        self.process_dic={}
        self.total_time = 0
        self.idle = 0
        self.done = False
        
        for i in range(self.machine_num):
            self.job_dic[i] = []
            self.oper_dic[i] = []
            self.process_dic[i] = []
        
        self.state = []
        for i in range(self.machine_num):
            self.state.append([self.active_m,(self.machine==i)*1,self.process_time])
        self.state = np.array(self.state)
        self.idlefix = 0
    
    
    def reset(self):
        self.active_m = np.zeros((self.machine.shape)).astype(int)
        self.machine_status = np.repeat(np.nan,self.machine_num)
        self.machine_run = np.repeat(-1,self.machine_num)
        self.doing = np.repeat(np.nan,self.machine_num)
        self.job_dic={}
        self.oper_dic={}
        self.process_dic={}
        for i in range(self.machine_num):
            self.job_dic[i] = []
            self.oper_dic[i] = []
            self.process_dic[i] = []
        self.total_time = 0
        self.idle = 0
        self.done = False
        self.state = []
        for i in range(self.machine_num):
            self.state.append([self.active_m,(self.machine==i)*1,self.process_time])
        self.state = np.array(self.state)[np.isnan(self.doing)]
        np.random.shuffle(self.machine)
        self.process_time = self.process_time_ori + np.random.randint(-5,5,size=(self.job_num,self.machine_num))
        self.idlefix = 0

        
    def step(self,action=0):
        act = action
        reward = []
        for i in range(self.machine_num):
            machine_flag = i
            if ~np.isnan(self.doing[i]):
                continue
    
            a  = (self.machine == machine_flag)*1
            flag_m = a*(1-self.active_m)
            operation_flag = np.sum(self.active_m,axis = 1)
            done_flag = np.argwhere(operation_flag>(self.machine_num-1)).flatten()
            joo = np.array(range(self.job_num))
            operation_flag[done_flag] = 0
                
            active_choose = np.argwhere(self.machine[joo,operation_flag]==machine_flag).flatten()
            active_choose = np.setdiff1d(active_choose,done_flag)
            if active_choose.size == 0  :
                self.job_dic[machine_flag].append(np.nan)
                self.oper_dic[machine_flag].append(np.nan)
                self.process_dic[machine_flag].append(np.nan)
                self.doing[machine_flag] = np.nan
                self.machine_run[machine_flag] = self.machine_run[machine_flag]+1
                if act[0] == 0: 
                    reward.append(0.1)
                else:
                    reward.append(0)
                act = np.delete(act,0)
            else :
                self.process_time[joo,operation_flag][active_choose]
                if act[0] == 0:
                    self.job_dic[machine_flag].append(np.nan)
                    self.oper_dic[machine_flag].append(np.nan)
                    self.process_dic[machine_flag].append(np.nan)
                    self.doing[machine_flag] = np.nan
                    self.machine_run[machine_flag] = self.machine_run[machine_flag]+1
                    reward.append(0)
                    act = np.delete(act,0)
                    continue
                elif act[0] == 1:
                    final_job = active_choose[np.argmin(self.process_time[joo,operation_flag][active_choose])]
                elif act[0] == 2:
                    final_job = active_choose[np.argmax(self.process_time[joo,operation_flag][active_choose])]
                elif act[0] == 3:
                    final_job = active_choose[np.argmin(np.sum(self.process_time*(1-self.active_m),axis=1)[active_choose])]
                else :
                    final_job = active_choose[np.argmax(np.sum(self.process_time*(1-self.active_m),axis=1)[active_choose])]
                final_oper = operation_flag[final_job]
                final_process_time = self.process_time[final_job,final_oper]
                reward.append(1-0.01*final_process_time)
                self.job_dic[machine_flag].append(final_job)
                self.oper_dic[machine_flag].append(operation_flag[final_job])
                self.process_dic[machine_flag].append(final_process_time+self.idle)
                self.doing[machine_flag] = final_process_time
                self.machine_run[machine_flag] = self.machine_run[machine_flag]+1
        self.idle= np.nanmin(self.doing)
        self.idlefix = np.nanmin(self.doing)
        if np.isnan(self.idle):
            self.idlefix = 0
        self.total_time = self.total_time + self.idlefix
        self.doing = self.doing - self.idle
        done_machine  = np.argwhere(self.doing==0).flatten()
        self.doing[np.argwhere(self.doing==0).flatten()] = np.nan
        for i in range(done_machine.size):
            self.active_m[self.job_dic[done_machine[i]][self.machine_run[done_machine[i]]],self.oper_dic[done_machine[i]][self.machine_run[done_machine[i]]]] = 1
        
        if sum(self.active_m.flatten()) == self.machine_num*self.job_num:
            self.done = True
    #    for i in range(machine_num):
    #        print(np.nansum(process_dic[i]))
        self.state = []
        for i in range(self.machine_num):
            self.state.append([self.active_m,(self.machine==i)*1,self.process_time])
        self.state = np.array(self.state)[np.isnan(self.doing)]
        return self.state , np.array(reward) , self.done
    def expand(self,net):
        job = copy.deepcopy(self)
        s = self.state
        while(True):
            a = v_wrap(net.choose_action(v_wrap(s)))
            s_,r,done = self.step(a.numpy())
            s = s_
            if done :
#                print(self.total_time)
                break
        return job ,self.total_time
    def expand_dispatch(self,dis = 1):
        job = copy.deepcopy(self)
        s = self.state
        while(True):
            a = v_wrap(np.repeat(dis,net.choose_action(v_wrap(s)).size))
            s_,r,done = self.step(a.numpy())
            s = s_
            if done :
#                print(self.total_time)
                break
        return job ,self.total_time
        

enet = Net(16,5)
tnet = Net(16,5)
enet.explore = False
job = job_shop_env("abz9")
s = job.state
for i in range(50):
    j = 0
    ot = 656
    while(True):
        j+= 1
        job,pre = job.expand(enet)
        a = v_wrap(enet.choose_action(v_wrap(s)))
        a.numpy()
        s_,r,done  =  job.step(a.numpy())
        if abs(pre - ot ) == 0 :
            r = 10 + r
        else:
            r = 10/abs(pre - ot )+r        
	
        opt = SharedAdam(enet.parameters(), lr=0.000001) 
        _,q_t = tnet.forward(v_wrap(s))
        loss = enet.loss_func(v_wrap(s),a,v_wrap(r)+0.01*q_t)
        if j % 2 ==0:
            tnet.load_state_dict(enet.state_dict())
        opt.zero_grad()
        loss.backward()
        opt.step()
        s = s_
        if done :
            print(job.total_time)
            job = job_shop_env("abz9")
#            job.reset()
            s = job.state
            break
    print(i)

#torch.save(net.state_dict(), "model10X10.tc")
#job2 = job_shop_env("orb02")
##
#_,time = job2.expand(net)
#print(time)
##print(optt(job2))
#job2.reset()




job2 = job_shop_env("abz9")
timmmer = []
#net.load_state_dict(torch.load("model6X6.tc"))

for i in range(100):
    job2,time_d1 = job2.expand_dispatch(1)
    job2,time_d2 = job2.expand_dispatch(2)
    job2,time_rl = job2.expand(enet)
#    optt(job2)
    timmmer.append([time_d1,time_d2,time_rl,optt(job2)])
    
    job2.reset()
timmmer = np.array(timmmer)
a = np.reshape(timmmer[:,3],(-1,1))
tt = timmmer-a
np.mean(a)
np.mean(tt,axis = 0)
np.std(tt,axis = 0 )
#926
c = time.time()
job2.expand_dispatch(1)
d = time.time()
d-c
job2 = job_shop_env("abz9")
c = time.time()
job2.expand_dispatch(2)
d = time.time()
d-c
job2 = job_shop_env("abz9")
c = time.time()
job2.expand(enet)
d = time.time()
d-c

#orb 1294
#opt_list_la = []
#time_list_la = []
#for i in range(5):
#    s = time.time()
#    job2 = job_shop_env("la1"+str(i+1))
#    a  = optt(job2)
#    e = time.time()
#    opt_list_la.append(a)
#    time_list_la.append(e-s)
#
#opt_list_orb = []
#time_list_orb = []
#
#for i in range(9):
#    s = time.time()
#    job2 = job_shop_env("orb0"+str(i+1))
#    a  = optt(job2)
#    e = time.time()
#    opt_list_orb.append(a)
#    time_list_orb.append(e-s)
#
#s = time.time()
#job2 = job_shop_env("orb10")
#a  = optt(job2)
#e = time.time()
#opt_list_orb.append(a)
#time_list_orb.append(e-s)