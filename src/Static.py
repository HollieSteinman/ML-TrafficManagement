import numpy as np
from tqdm import tqdm

from env.traffic_intersection import TrafficIntersection

env = TrafficIntersection('/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/env/sumo/test/traffic.net.xml', '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/env/sumo/test/traffic.rou.xml', '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/env/sumo/test/traffic.add.xml', gui=True)
env.reset()

change_steps = 13
no_actions = env.action_space.n
rewards = []
for e in tqdm(range(10)):
    env.reset()
    done = False
    next_change = change_steps
    current_step = 0
    total_reward = 0
    steps = 0
    while not done:
        # step
        if next_change == 0:
            next_change = change_steps
            current_step += 1
            if current_step > no_actions - 1:
                current_step = 0
        obsv, reward, done, info = env.step(current_step)
        total_reward += reward
        next_change -= 1
        steps +=1
    print("Episode finished after {} steps. Total reward: {}.".format(steps, total_reward))
    rewards.append(total_reward)
print("Average reward: {}".format(np.mean(rewards)))
env.close()

        
