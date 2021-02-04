import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from collections import deque
import matplotlib.pyplot as plt
import copy

env_name='CartPole-v0' #using the CartPole-v0 environment
env=gym.make(env_name)

obs_dim=env.observation_space.shape[0] #size of the observation_space
n_acts=env.action_space.n # size of the action space

M = 1000000 # number of episodes to train for
T = 5000 #maximum timestep for each game
batch_size=256
N=1000000 # size of experience replay memory
render=False
target_update_interval =100 # how many frames the target function updates after
lr=0.0001



hidden_sizes=[24,48,96,48,24]
# hidden_sizes=[24,48,24]

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def epsilon_function(frame, start=1, end=0.1,n_frames=1e5):
    # defines episilon (the probability that the agent takes a random aciton)
    #this changes linearly from start to end over the first n_frames, and after that returns end
    if frame<n_frames:
        return start-frame/n_frames*(start-end)
    else:
        return end

def target_function(expers,Q,gamma=0.99):
    #defines our target function, for use in the Bellman Equation
    #takes a batch of experiences expers, and for each of these calculates the target value,
    # using the given Q function

    #expers is a list  of experiences
    #each experience has the form [s_i,a_i,reward, s_i+1, done]

    batch_size=len(expers) #size of the batch
    targets=torch.zeros(batch_size,dtype=float)
    for ii in range(batch_size):
        exper=expers[ii] #for this experience

        if exper[4]: #if the experience terminates
            targets[ii]=exper[2]
        else: #if the experience doesn't terminate
            targets[ii]=exper[2]+gamma*torch.max(Q(torch.FloatTensor(exper[3])))

    return targets #return the targets for each of the experiences

def compute_loss(targets,batch,Q):
    #computes the losses for each of the each experience in batch, using the Q function Q,
    #and the previously calculated target values
    batch_size=len(targets)
    losses=torch.zeros((batch_size))
    for ii in range(batch_size):
        exper=batch[ii] #for each experience
        losses[ii]=(targets[ii]-Q(torch.FloatTensor(exper[0]))[exper[1]])**2 #calculate the squared error
    return losses


def select_batch(D, batch_size):
    #this function randomly selects a batch of size batch_size of experiences from D
    inds=np.random.randint(low=0,high=batch_size,size=(1,))

    batch=[D[ind] for ind in inds]
    return batch


def train(env_name="CartPole-v0",hidden_sizes=hidden_sizes
        ,M=M,T=T,N=N,batch_size=batch_size,render=render
        ,target_update_interval=target_update_interval,lr=lr, print_output_episode=50):

    env=gym.make(env_name)
    obs_dim=env.observation_space.shape[0] #size of the observation_space
    n_acts=env.action_space.n # size of the action space


    Q=mlp(sizes=[obs_dim]+hidden_sizes+[n_acts],activation=nn.ReLU,output_activation=nn.ReLU) #building our Q network

    #choosing which optimizer to use
    # optimizer = torch.optim.RMSprop(Q.parameters())#,lr=0.001)
    optimizer = torch.optim.Adam(Q.parameters(),lr=lr)

    D=deque([],maxlen=N) #empty experience replay memory
    total_reward=[] #track the reward of each episode
    frame_num=0
    for episode in range(M): #for M episodes
        s=env.reset() #reset the environment
        done=False
        tt=0 # tract the timesteps for the episode
        reward_sum=0 #track the reward for each episode

        while not done:
            frame_num+=1
            if render: #render if you want to
                env.render()

            # apply epsilon-greedy exploration strategy

            if np.random.rand()<epsilon_function(frame_num): #selcet random action with probability epsilon
                aa=torch.randint(low=0,high=n_acts,size=(1,))
            else: #otherwise select the action with the highest expected reward, determined by Q
                aa=torch.argmax(Q(torch.FloatTensor(s)))

            experience = [s,aa] #save the initial state and action for the experience

            s, rew, done, _ = env.step(aa.item()) # take a step in the environment, using the selected action

            experience.append(rew) #save reward for experience
            experience.append(s) # save the updated state (s_i+1)
            experience.append(done) #save whether this expe terminates
            D.append(experience) # add the experience to the replay memory
            reward_sum += rew #update the reward

            if tt>=T: #terminate after this episode if reached mx timestep
                done=True

            batch=select_batch(D,np.min([batch_size,len(D)])) #select a batch from D
            #the batch size is either batch_size, or if there aren't enough experiences for this just all of D

            if (frame_num-1) % target_update_interval ==0:
                #update the Q for the targets every target_update_interval frames
                Q_target=copy.deepcopy(Q)

            targets=target_function(batch,Q_target) #compute the targets

            optimizer.zero_grad()
            batch_loss=compute_loss(targets,batch,Q) #compute the batch loss
            #take optimizer step
            batch_loss.backward()
            optimizer.step()

        total_reward.append(reward_sum)

        if print_output_episode and (episode+1)%print_output_episode==0:
            print('Episode: '+ str(episode+1) +', Frames: '+str(frame_num))
            print(np.mean(total_reward[-print_output_episode:]))

# train()

def train_2(env_name="CartPole-v0",hidden_sizes=hidden_sizes
        ,M=M,T=T,N=N,batch_size=batch_size,render=render
        ,target_update_interval=target_update_interval,lr=lr, print_output_episode=50):

    env=gym.make(env_name)
    obs_dim=env.observation_space.shape[0] #size of the observation_space
    n_acts=env.action_space.n # size of the action space


    Q=mlp(sizes=[obs_dim]+hidden_sizes+[n_acts],activation=nn.ReLU,output_activation=nn.ReLU) #building our Q network

    #choosing which optimizer to use
    # optimizer = torch.optim.RMSprop(Q.parameters())#,lr=0.001)
    optimizer = torch.optim.Adam(Q.parameters(),lr=lr)

    D=deque([],maxlen=N) #empty experience replay memory
    total_reward=[] #track the reward of each episode
    average_Q=[]
    frame_num=0
    for episode in range(M): #for M episodes
        s=env.reset() #reset the environment
        done=False
        tt=0 # tract the timesteps for the episode
        reward_sum=0 #track the reward for each episode

        while not done:
            frame_num+=1
            if render: #render if you want to
                env.render()

            # apply epsilon-greedy exploration strategy

            if np.random.rand()<epsilon_function(frame_num): #selcet random action with probability epsilon
                aa=torch.randint(low=0,high=n_acts,size=(1,))
                average_Q.append(np.nan)
            else: #otherwise select the action with the highest expected reward, determined by Q
                avec=Q(torch.FloatTensor(s))
                aa=torch.argmax(avec)
                average_Q.append(torch.mean(avec).item())
            experience = [s,aa] #save the initial state and action for the experience

            s, rew, done, _ = env.step(aa.item()) # take a step in the environment, using the selected action

            experience.append(rew) #save reward for experience
            experience.append(s) # save the updated state (s_i+1)
            experience.append(done) #save whether this expe terminates
            D.append(experience) # add the experience to the replay memory
            reward_sum += rew #update the reward

            if tt>=T: #terminate after this episode if reached mx timestep
                done=True

            if (frame_num-1) % target_update_interval ==0:
                #update the Q for the targets every target_update_interval frames
                Q_target=Q#copy.deepcopy(Q)

            if len(D)>batch_size:
                batch=select_batch(D,batch_size) #select a batch from D
                #the batch size is either batch_size, or if there aren't enough experiences for this just all of D

                targets=target_function(batch,Q_target) #compute the targets

                optimizer.zero_grad()
                batch_loss=compute_loss(targets,batch,Q) #compute the batch loss
                #take optimizer step
                batch_loss.backward()
                optimizer.step()

        total_reward.append(reward_sum)

        if print_output_episode and (episode+1)%print_output_episode==0:
            print('Episode: '+ str(episode+1) +', Frames: '+str(frame_num))
            print('Episode reward: '+str(np.mean(total_reward[-print_output_episode:]))
            + ', Avegage Q: '+ str(np.nanmean(average_Q[-print_output_episode:])))


def train_3(env_name="CartPole-v0",hidden_sizes=hidden_sizes
        ,M=M,T=T,N=N,batch_size=batch_size,render=render
        ,target_update_interval=target_update_interval,lr=lr, print_output_episode=500):

    env=gym.make(env_name)
    obs_dim=env.observation_space.shape[0] #size of the observation_space
    n_acts=env.action_space.n # size of the action space


    Q=mlp(sizes=[obs_dim]+hidden_sizes+[n_acts],activation=nn.ReLU) #building our Q network

    #choosing which optimizer to use
    optimizer = torch.optim.RMSprop(Q.parameters(),lr=lr)
    # optimizer = torch.optim.Adam(Q.parameters(),lr=lr)

    D=deque([],maxlen=N) #empty experience replay memory
    total_reward=[] #track the reward of each episode
    average_Q=[]
    frame_num=0
    final_weights=torch.zeros((M,48),dtype=float)
    for episode in range(M): #for M episodes
        s=env.reset() #reset the environment
        done=False
        tt=0 # tract the timesteps for the episode
        reward_sum=0 #track the reward for each episode

        while not done:
            frame_num+=1
            if render: #render if you want to
                env.render()

            # apply epsilon-greedy exploration strategy

            if np.random.rand()<epsilon_function(episode): #select random action with probability epsilon
                aa=torch.randint(low=0,high=n_acts,size=(1,))
                average_Q.append(np.nan)
            else: #otherwise select the action with the highest expected reward, determined by Q
                avec=Q(torch.FloatTensor(s))
                aa=torch.argmax(avec)
                average_Q.append(torch.mean(avec).item())
            experience = [s,aa] #save the initial state and action for the experience

            s, rew, done, _ = env.step(aa.item()) # take a step in the environment, using the selected action

            experience.append(rew) #save reward for experience
            experience.append(s) # save the updated state (s_i+1)
            experience.append(done) #save whether this expe terminates
            D.append(experience) # add the experience to the replay memory
            reward_sum += rew #update the reward

            if tt>=T: #terminate after this episode if reached mx timestep
                done=True

        if (episode-1) % 10 == 0:
            # update the Q for the targets every target_update_interval frames
            Q_target=copy.deepcopy(Q)

        if len(D)>batch_size:
            batch=select_batch(D,batch_size) #select a batch from D
            #the batch size is either batch_size, or if there aren't enough experiences for this just all of D

            targets=target_function(batch,Q_target) #compute the targets

            optimizer.zero_grad()
            batch_loss=compute_loss(targets,batch,Q) #compute the batch loss
            #take optimizer step
            batch_loss.backward()
            optimizer.step()

        total_reward.append(reward_sum)
        final_weights[episode,:]=torch.reshape(Q[-2].weight,(48,))
        if print_output_episode and (episode+1)%print_output_episode==0:
            print('Episode: '+ str(episode+1) +', Frames: '+str(frame_num))
            print('Episode reward: '+str(np.mean(total_reward[-print_output_episode:]))
            + ', Avegage Q: '+ str(np.nanmean(average_Q[-print_output_episode:])))

    plt.imshow(final_weights.detach().numpy(),aspect='auto')
    plt.colorbar()
    plt.show()

# def train_3(env_name="CartPole-v0",hidden_sizes=hidden_sizes
#         ,M=M,T=T,N=N,batch_size=batch_size,render=render
#         ,target_update_interval=target_update_interval,lr=lr, print_output_episode=500):
#
#     env=gym.make(env_name)
#     obs_dim=env.observation_space.shape[0] #size of the observation_space
#     n_acts=env.action_space.n # size of the action space
#
#
#     Q=mlp(sizes=[obs_dim]+hidden_sizes+[n_acts],activation=nn.ReLU) #building our Q network
#     loss_fun=nn.MSELoss()
#     #choosing which optimizer to use
#     optimizer = torch.optim.RMSprop(Q.parameters(),lr=lr)
#     # optimizer = torch.optim.Adam(Q.parameters(),lr=lr)
#
#     D=deque([],maxlen=N) #empty experience replay memory
#     total_reward=[] #track the reward of each episode
#     average_Q=[]
#     frame_num=0
#     final_weights=torch.zeros((M,48),dtype=float)
#     for episode in range(M): #for M episodes
#         s=env.reset() #reset the environment
#         done=False
#         tt=0 # tract the timesteps for the episode
#         reward_sum=0 #track the reward for each episode
#
#         while not done:
#             frame_num+=1
#             if render: #render if you want to
#                 env.render()
#
#             # apply epsilon-greedy exploration strategy
#
#             if np.random.rand()<epsilon_function(episode): #select random action with probability epsilon
#                 aa=torch.randint(low=0,high=n_acts,size=(1,))
#                 average_Q.append(np.nan)
#             else: #otherwise select the action with the highest expected reward, determined by Q
#                 avec=Q(torch.FloatTensor(s))
#                 aa=torch.argmax(avec)
#                 average_Q.append(torch.mean(avec).item())
#             experience = [s,aa] #save the initial state and action for the experience
#
#             s, rew, done, _ = env.step(aa.item()) # take a step in the environment, using the selected action
#
#             experience.append(rew) #save reward for experience
#             experience.append(s) # save the updated state (s_i+1)
#             experience.append(done) #save whether this expe terminates
#             D.append(experience) # add the experience to the replay memory
#             reward_sum += rew #update the reward
#
#             if tt>=T: #terminate after this episode if reached mx timestep
#                 done=True
#
#         if (episode-1) % 10 == 0:
#             # update the Q for the targets every target_update_interval frames
#             Q_target=copy.deepcopy(Q)
#
#         if len(D)>batch_size:
#             batch=select_batch(D,batch_size) #select a batch from D
#             #the batch size is either batch_size, or if there aren't enough experiences for this just all of D
#
#             targets=target_function(batch,Q_target) #compute the targets
#
#             Q_vals=torch.zeros(batch_size,dtype=float)
#
#             for ii, ex in enumerate(batch):
#                 Q_vals[ii] = Q(torch.FloatTensor(ex[0]))[ex[1]]
#
#             optimizer.zero_grad()
#             batch_loss=loss_fun(targets,Q_vals)
#             # batch_loss=compute_loss(targets,batch,Q) #compute the batch loss
#             #take optimizer step
#             batch_loss.backward()
#             optimizer.step()
#
#         total_reward.append(reward_sum)
#         final_weights[episode,:]=torch.reshape(Q[-2].weight,(48,))
#         if print_output_episode and (episode+1)%print_output_episode==0:
#             print('Episode: '+ str(episode+1) +', Frames: '+str(frame_num))
#             print('Episode reward: '+str(np.mean(total_reward[-print_output_episode:]))
#             + ', Avegage Q: '+ str(np.nanmean(average_Q[-print_output_episode:])))
#
#     plt.imshow(final_weights.detach().numpy(),aspect='auto')
#     plt.colorbar()
#     plt.show()
train_3()
