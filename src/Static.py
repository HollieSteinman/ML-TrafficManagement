import numpy as np
from tqdm import tqdm
from env.traffic_intersection import TrafficIntersection

EVALUATION = True
PROJECT_PATH = '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/'

if not EVALUATION:
    env = TrafficIntersection(
        PROJECT_PATH + 'src/env/sumo/test/traffic.net.xml',
        PROJECT_PATH + 'src/env/sumo/test/traffic.rou.xml',
        PROJECT_PATH + 'src/env/sumo/test/traffic.add.xml',
        gui=False,
        max_dur=500,
        action_dur=6,
        yellow_dur=5,
        green_dur=10)
    phase_dur = {0: 45, 1: 30, 2: 45}
else:
    env = TrafficIntersection(
        PROJECT_PATH + 'src/env/sumo/real/traffic.net.xml',
        PROJECT_PATH + 'src/env/sumo/real/traffic.rou.xml',
        PROJECT_PATH + 'src/env/sumo/real/traffic.add.xml',
        gui=False,
        max_dur=500,
        action_dur=6,
        yellow_dur=5,
        green_dur=10)
    phase_dur = {0: 60, 1: 45, 2: 30}

env.reset()

no_actions = env.action_space.n
rewards = []
avg_steps = []
episodes = 10
for e in tqdm(range(episodes)):
    env.reset()
    done = False
    current_step = 0
    next_change = phase_dur[current_step]
    total_reward = 0
    steps = 0
    while not done:
        # step
        if next_change == 0:
            current_step += 1
            if current_step > no_actions - 1:
                current_step = 0
            next_change = phase_dur[current_step]
        obsv, reward, done, info = env.step(current_step)
        total_reward += reward
        next_change -= 1
        steps +=1
    print(f"Episode finished after {steps} steps. Total reward: {total_reward}.")
    rewards.append(total_reward)
    avg_steps.append(steps)
print(f"Average reward: {np.average(rewards)}, Average steps: {np.average(avg_steps)}")
env.close()

        
