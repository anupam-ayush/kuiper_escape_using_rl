import gym
import numpy as np 
import random
import matplotlib.pyplot as plt
import time

#alpha=range(0,1], small eps>0

P={((1,1),0) : {0:1/3,1:1/3,2:1/3}, ((1,1),1) : {0:1/3,1:1/3,2:1/3}, ((1,1),2) : {0:1/3,1:1/3,2:1/3}, ((1,1),3) : {0:1/3,1:1/3,2:1/3}, ((1,2),0) : {0:1/3,1:1/3,2:1/3}, ((1,2),1) : {0:1/3,1:1/3,2:1/3}, ((1,2),2) : {0:1/3,1:1/3,2:1/3}, ((1,2),3) : {0:1/3,1:1/3,2:1/3}, ((1,3),0) : {0:1/3,1:1/3,2:1/3}, ((1,3),1) : {0:1/3,1:1/3,2:1/3}, ((1,3),2) : {0:1/3,1:1/3,2:1/3}, ((1,3),3) : {0:1/3,1:1/3,2:1/3}, ((1,4),0) : {0:1/3,1:1/3,2:1/3}, ((1,4),1) : {0:1/3,1:1/3,2:1/3}, ((1,4),2) : {0:1/3,1:1/3,2:1/3}, ((1,4),3) : {0:1/3,1:1/3,2:1/3}, ((2,1),0) : {0:1/3,1:1/3,2:1/3}, ((2,1),1) : {0:1/3,1:1/3,2:1/3}, ((2,1),2) : {0:1/3,1:1/3,2:1/3}, ((2,1),3) : {0:1/3,1:1/3,2:1/3}, ((2,2),0) : {0:1/3,1:1/3,2:1/3}, ((2,2),1) : {0:1/3,1:1/3,2:1/3}, ((2,2),2) : {0:1/3,1:1/3,2:1/3}, ((2,2),3) : {0:1/3,1:1/3,2:1/3}, ((2,3),0) : {0:1/3,1:1/3,2:1/3}, ((2,3),1) : {0:1/3,1:1/3,2:1/3}, ((2,3),2) : {0:1/3,1:1/3,2:1/3}, ((2,3),3) : {0:1/3,1:1/3,2:1/3}, ((2,4),0) : {0:1/3,1:1/3,2:1/3}, ((2,4),1) : {0:1/3,1:1/3,2:1/3}, ((2,4),2) : {0:1/3,1:1/3,2:1/3}, ((2,4),3) : {0:1/3,1:1/3,2:1/3}, ((3,1),0) : {0:1/3,1:1/3,2:1/3}, ((3,1),1) : {0:1/3,1:1/3,2:1/3}, ((3,1),2) : {0:1/3,1:1/3,2:1/3}, ((3,1),3) : {0:1/3,1:1/3,2:1/3}, ((3,2),0) : {0:1/3,1:1/3,2:1/3}, ((3,2),1) : {0:1/3,1:1/3,2:1/3}, ((3,2),2) : {0:1/3,1:1/3,2:1/3}, ((3,2),3) : {0:1/3,1:1/3,2:1/3}, ((3,3),0) : {0:1/3,1:1/3,2:1/3}, ((3,3),1) : {0:1/3,1:1/3,2:1/3}, ((3,3),2) : {0:1/3,1:1/3,2:1/3}, ((3,3),3) : {0:1/3,1:1/3,2:1/3}, ((3,4),0) : {0:1/3,1:1/3,2:1/3}, ((3,4),1) : {0:1/3,1:1/3,2:1/3}, ((3,4),2) : {0:1/3,1:1/3,2:1/3}, ((3,4),3) : {0:1/3,1:1/3,2:1/3}, ((4,1),0) : {0:1/3,1:1/3,2:1/3}, ((4,1),1) : {0:1/3,1:1/3,2:1/3}, ((4,1),2) : {0:1/3,1:1/3,2:1/3}, ((4,1),3) : {0:1/3,1:1/3,2:1/3}, ((4,2),0) : {0:1/3,1:1/3,2:1/3}, ((4,2),1) : {0:1/3,1:1/3,2:1/3}, ((4,2),2) : {0:1/3,1:1/3,2:1/3}, ((4,2),3) : {0:1/3,1:1/3,2:1/3}, ((4,3),0) : {0:1/3,1:1/3,2:1/3}, ((4,3),1) : {0:1/3,1:1/3,2:1/3}, ((4,3),2) : {0:1/3,1:1/3,2:1/3}, ((4,3),3) : {0:1/3,1:1/3,2:1/3}, ((4,4),0) : {0:1/3,1:1/3,2:1/3}, ((4,4),1) : {0:1/3,1:1/3,2:1/3}, ((4,4),2) : {0:1/3,1:1/3,2:1/3}, ((4,4),3) : {0:1/3,1:1/3,2:1/3}}
q_values={}
for x in range(1,5):
    for y in range(1,5):
        for z in range(4):
            q_values[((x,y),z)]={0:0,1:0,2:0}



terminal=(4,4)
nA=3
epi=200
max_steps=75

def QL_control(env,q_values,P,epi,terminal,nA,max_steps,gamma=0.9,alpha=0.15):
    eps=1.0
    steps=[]
    total_reward=[]
    for epi_no in range(1,epi+1):
        env.reset()
        k=0
        R=0
        state=(env.agent_pos,env.agent_dir)
        eps=eps_decay(eps,epi_no)
        alpha=alpha_decay(alpha,epi_no)
        while state[0]!=terminal:
            env.render()
            action=eps_greedy2(eps,P,state)
            _,reward,done,_,_=env.step(action)
            R+=reward
            new_state=(env.agent_pos,env.agent_dir)
            q_values[state][action]+=alpha*(reward + gamma*max(list(q_values[new_state].values())) - q_values[state][action])
            A=max(q_values[state], key=q_values[state].get)
            for a in range(nA):
                if a ==A :
                    P[state][a]=(1-eps+eps/nA)
                else:
                    P[state][a]=(eps/nA)
            k+=1
            state=new_state

        steps.append(k)
        total_reward.append(R)
    return q_values,P,steps,total_reward



def alpha_decay(alpha,epi_no ):
    new_alpha=alpha*float('0.9'+'9'*epi_no)
    return new_alpha

def eps_decay(eps,epi_no,min_eps=0.01 ):
    new_eps=eps*0.95
    return new_eps


def eps_greedy(eps,P,state):
    r=random.random()
    action = None
    cumulative_prob = 0.0
    for a in range(3):
      cumulative_prob += P[state][a]
      if r < cumulative_prob:
        action = a
        break
    return action

def plotting(x,y,xname,yname,title):
    plt.figure()
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.grid(True)
    plt.show()


 


env=gym.make('MiniGrid-Empty-6x6-v0')#,render_mode='human')

initial=time.time()
q_values,P,steps,total_reward = QL_control(env,q_values,P,epi,terminal,nA,max_steps,gamma=0.8,alpha=0.2) 
final=time.time()
duration=final-initial
episode=list(range(1,epi+1))
plotting(episode,steps,"episode","steps","Steps vs Episode")
plotting(episode,total_reward,"episode","reward","Reward vs Episode")

print("Time taken: ",duration)
print("\n q_value: \n",q_values)
print("\n P: \n",P)
print("\n steps: \n ",steps)
print("\n returns: \n ",total_reward)
print(f'\nNo. of episode: {epi}\nStep list length: {len(steps)}\nReturn list length: {len(total_reward)}')
