from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv

from env.traffic_intersection import TrafficIntersection

EVALUATION = True
PROJECT_PATH = '/Users/holliesteinman/Documents/Uni/Year4/Sem1/Machine Learning/Assignment 2/'

def learn(
        env: TrafficIntersection,
        learning: float=0.1,
        discount: float=0.99,
        epsilon: float=1,
        stop_decay_percent: float=0.5,
        episodes: float=300
    ):
    # initialise variables
    all_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
    rewards = []
    analyse_every = max(1, episodes / 25)
    state = env.discretise_state(env.reset())
    q_table = defaultdict(lambda: [0 for _ in range(env.action_space.n)])
    q_table[state] = [0 for _ in range(env.action_space.n)]
    stop_decay = int(episodes * stop_decay_percent)
    epsilon_decay = epsilon / stop_decay

    # loop through episodes
    for e in tqdm(range(episodes)):
        done = False
        total_reward = 0
        state = env.discretise_state(env.reset())

        # select actions and step
        while not done:
            # exploit if random < 1 - epsilon
            if np.random.random() < 1 - epsilon:
                action = np.argmax(q_table[state])
            else:
                # else explore
                action = int(env.action_space.sample())

            # step with optimal action & discretise state
            s, reward, done, _ = env.step(action)
            next_state = env.discretise_state(s)

            # add state to q table
            if next_state not in q_table:
                q_table[next_state] = [0 for _ in range(env.action_space.n)]
            
            # q function
            q_table[state][action] = (1-learning) * q_table[state][action] + learning*(reward +
                discount*max(q_table[next_state]))
            
            # set state & reward
            state = next_state
            total_reward += reward
        
        # decay epsilon
        if epsilon > 0 and e < stop_decay:
            epsilon -= epsilon_decay
        
        rewards.append(total_reward)

        # output episode & averages
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
        q_table,
        episodes
    ):
    rewards = []
    steps = []
    for e in tqdm(range(episodes)):
        step = 0
        state = env.discretise_state(env.reset())
        done = False
        avg = 0
        while not done:
            action = np.argmax(q_table[state])
            s, reward, done, info = env.step(action)
            state = env.discretise_state(s)
            avg += reward
            step += 1
        rewards.append(avg)
        steps.append(step)
    print(f"Average steps: {np.average(steps)}, final average reward {np.average(rewards)}")

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

episodes = 1000
q_table, rewards = learn(env, 0.1, 0.95, 1,  0.75, episodes)
plot(rewards, episodes)
test(env, q_table, 10)