from dis import disco
import numpy as np
from tqdm import tqdm

from env.traffic_intersection import TrafficIntersection

def QLearning(
        env: TrafficIntersection,
        learning,
        discount,
        epsilon,
        episodes
    ):
    rewards = []
    state = env.discretise_state(env.reset())
    q_table = {state: [0 for _ in range(env.action_space.n)]}
    epsilon_decay = (epsilon) / episodes

    for e in tqdm(range(episodes)):
        done = False
        total_reward = 0
        state = env.discretise_state(env.reset())

        while not done:
            if np.random.random() < 1 - epsilon:
                action = np.argmax(q_table[state])
            else:
                action = int(env.action_space.sample())

            s, reward, done, info = env.step(action)
            next_state = env.discretise_state(s)

            if next_state not in q_table:
                q_table[next_state] = [0 for _ in range(env.action_space.n)]
            
            q_table[state][action] = q_table[state][action] + learning*(reward +
                discount*max(q_table[next_state]) - q_table[state][action])
            
            state = next_state
            total_reward += reward
        
        if epsilon > 0:
            epsilon -= epsilon_decay
        
        rewards.append(total_reward)

        if (e + 1) % 100 == 0:
            average_reward = np.mean(rewards)
            rewards = []
            print(f'Episode {e + 1} - Average Reward: {average_reward}')
    env.close()

    return q_table

env = TrafficIntersection(
    '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/env/sumo/test/traffic.net.xml',
    '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/env/sumo/test/traffic.rou.xml',
    '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/env/sumo/test/traffic.add.xml',
    gui=False,
    max_dur=2500)
q_table = QLearning(env, 0.5, 0.85, 0.8, 200)