#Code by Doga Dokuz, 2025
import numpy as np
import matplotlib.pyplot as plt

N = 10  # # of possible actions
num_runs = 2000  # random value choices
num_steps = 1000  # # of action choices per run
epsilons = [0, 0.01, 0.1]  # random action probability (exploration probability)

#creating a dictionary to store the average rewards for each epsilon value
avg_rewards = {epsilon: np.zeros(num_steps) for epsilon in epsilons}


for epsilon in epsilons:
    rewards = np.zeros((num_runs, num_steps))  #creating a 2D array to store the rewards for each run and time step for each epsilon value

    for run in range(num_runs): #looping through each run of 2000
        
        q_true = np.random.normal(0, 1, N) #randomly generating 10 values of q(a) which is the true action value
        
        # Initialize estimated values and action counts
        Q = np.zeros(N) #we want to estimate the highest estimated reward of exploitation so we create an array of zeros for Q(a) value
        action_counts = np.zeros(N) #similarly we create an array of zeros for action counts for sample average method

        for t in range(num_steps): 
            #selecting an action based on the epsilon value
            if np.random.rand() < epsilon: #randomly select a value if it is less than epsilon, the agent explores
                action = np.random.randint(N)  #exploration action
            else:
                action = np.argmax(Q)  #best estimated exploitation action 

            # Generate reward with noise
            reward = q_true[action] + np.random.normal(0, 1) #generating the reward for the selected action adding Gaussian distributed noise with a mean of 0
            rewards[run, t] = reward #storing the rewards in the array created

            action_counts[action] += 1 #incrementing the action count for the selected action
            Q[action] += (reward - Q[action]) / action_counts[action] #updating the estimated value of the selected action using the sample average method
            
    avg_rewards[epsilon] = np.mean(rewards, axis=0) #averaging the rewards over the runs for each epsilon value

#plotting the average rewards for each epsilon value
plt.figure(figsize=(10, 6))
for eps in epsilons:
    plt.plot(avg_rewards[eps], label=f"ε = {eps}")

plt.xlabel("Time steps")
plt.ylabel("Average reward")
plt.title("Epsilon-Greedy Action Selection Performance")
plt.legend()
plt.grid(True)
plt.show()

'''
when ε = 0 , the agent always picks the action with the highest estimated rewards, this causes it to choose not many optimal choices before enough exploration. 
we can see that this is the worst performing epsilon value and the performance increases with ε>0. This is pure exploitation.

when ε = 0.01, the agent occasionally and minimally explores which allows it to correct poor initial estimates, we can say that this model balances exploration and exploitation.
also it can achieve higher rewards in the long run because of the trend seen in the graph

when ε = 0.1, the agent explores more frequenty which prevents convergence at the first time steps of the model. Although it starts with lower rewards, it eventually converges 
and finds the optimal actions. We can say that excessive exploration prevents the model from fully capitalizing on the best choices.'''
