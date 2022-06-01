from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv

from env.traffic_intersection import TrafficIntersection

def learn(
        env: TrafficIntersection,
        learning,
        discount,
        epsilon,
        episodes
    ):
    all_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
    rewards = []
    analyse_every = max(1, episodes / 50)
    state = env.discretise_state(env.reset())
    q_table = defaultdict(lambda: [0 for _ in range(env.action_space.n)])
    q_table[state] = [0 for _ in range(env.action_space.n)]
    stop_decay = episodes / 2
    epsilon_decay = epsilon / stop_decay

    for e in tqdm(range(episodes)):
        done = False
        total_reward = 0
        state = env.discretise_state(env.reset())

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[state])
            else:
                action = int(env.action_space.sample())

            s, reward, done, info = env.step(action)
            next_state = env.discretise_state(s)

            if next_state not in q_table:
                q_table[next_state] = [0 for _ in range(env.action_space.n)]
            
            q_table[state][action] = (1-learning) * q_table[state][action] + learning*(reward +
                discount*max(q_table[next_state]))
            
            state = next_state
            total_reward += reward
        
        if epsilon > 0 and episodes < stop_decay:
            epsilon -= epsilon_decay
        
        rewards.append(total_reward)

        if (e + 1) % analyse_every == 0:
            average_reward = np.mean(rewards)
            all_rewards['ep'].append(e + 1)
            all_rewards['avg'].append(average_reward)
            all_rewards['max'].append(np.max(rewards))
            all_rewards['min'].append(np.min(rewards))
            rewards = []
            print(f'Episode {e + 1} - Average Reward: {average_reward}')
    env.close()

    return q_table, all_rewards

def test(
        env: TrafficIntersection,
        q_table
    ):
    state = env.discretise_state(env.reset())
    done = False
    avg = 0
    while not done:
        action = np.argmax(q_table[state])
        s, reward, done, info = env.step(action)
        state = env.discretise_state(s)
        avg += reward
    print(f"Final average reward {avg}")

def save(q_table):
    with open('model.csv', 'w') as f:
        writer = csv.writer(f)

        for s in q_table:
            writer.writerow(np.array([i for i in s] + q_table[s]))

def load(env: TrafficIntersection):
    env.reset()
    with open('/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/model.csv', 'r') as f:
        reader = csv.reader(f)
        q_table = {}

        for row in reader:
            if row != []:
                q_table[tuple([int(float(c)) for c in row[:(1 + 1 + len(env.lanes))]])] = [float(c) for c in row[(1 + 1 + len(env.lanes)):]]
        
        return q_table

def plot(rewards, episodes):
    plt.plot(rewards['ep'], rewards['avg'], label="Average")
    plt.plot(rewards['ep'], rewards['max'], label="Max")
    plt.plot(rewards['ep'], rewards['min'], label="Min")
    plt.legend(loc=4)
    plt.ylabel("Episodes")
    plt.xlabel("Reward")
    plt.title(f"Rewards for {episodes} episodes")
    plt.grid(True)
    plt.show()

env = TrafficIntersection(
    '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/env/sumo/test/traffic.net.xml',
    '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/env/sumo/test/traffic.rou.xml',
    '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/src/env/sumo/test/traffic.add.xml',
    gui=False,
    max_dur=2500)

episodes = 100
q_table, rewards = learn(env, 0.1, 0.95, 1, episodes)
plot(rewards, episodes)
test(env, q_table)

#q_table = load(env)