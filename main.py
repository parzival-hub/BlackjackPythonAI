import pickle
import random
from collections import defaultdict
from copy import deepcopy

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

env = gym.make('Blackjack-v1')
filename = 'powerVisualize'
maxEpisodes = 25000
evalSteps = int(maxEpisodes / 10)
useDict = ''
evaluate = True
only_show_tables = False
basic_strategy = {(4, 1, False): (0, 1),
                  (4, 2, False): (0, 1),
                  (4, 3, False): (0, 1),
                  (4, 4, False): (0, 1),
                  (4, 5, False): (0, 1),
                  (4, 6, False): (0, 1),
                  (4, 7, False): (0, 1),
                  (4, 8, False): (0, 1),
                  (4, 9, False): (0, 1),
                  (4, 10, False): (0, 1),
                  (5, 1, False): (0, 1),
                  (5, 2, False): (0, 1),
                  (5, 3, False): (0, 1),
                  (5, 4, False): (0, 1),
                  (5, 5, False): (0, 1),
                  (5, 6, False): (0, 1),
                  (5, 7, False): (0, 1),
                  (5, 8, False): (0, 1),
                  (5, 9, False): (0, 1),
                  (5, 10, False): (0, 1),
                  (6, 1, False): (0, 1),
                  (6, 2, False): (0, 1),
                  (6, 3, False): (0, 1),
                  (6, 4, False): (0, 1),
                  (6, 5, False): (0, 1),
                  (6, 6, False): (0, 1),
                  (6, 7, False): (0, 1),
                  (6, 8, False): (0, 1),
                  (6, 9, False): (0, 1),
                  (6, 10, False): (0, 1),
                  (7, 1, False): (0, 1),
                  (7, 2, False): (0, 1),
                  (7, 3, False): (0, 1),
                  (7, 4, False): (0, 1),
                  (7, 5, False): (0, 1),
                  (7, 6, False): (0, 1),
                  (7, 7, False): (0, 1),
                  (7, 8, False): (0, 1),
                  (7, 9, False): (0, 1),
                  (7, 10, False): (0, 1),
                  (8, 1, False): (0, 1),
                  (8, 2, False): (0, 1),
                  (8, 3, False): (0, 1),
                  (8, 4, False): (0, 1),
                  (8, 5, False): (0, 1),
                  (8, 6, False): (0, 1),
                  (8, 7, False): (0, 1),
                  (8, 8, False): (0, 1),
                  (8, 9, False): (0, 1),
                  (8, 10, False): (0, 1),
                  (9, 1, False): (0, 1),
                  (9, 2, False): (0, 1),
                  (9, 3, False): (0, 1),
                  (9, 4, False): (0, 1),
                  (9, 5, False): (0, 1),
                  (9, 6, False): (0, 1),
                  (9, 7, False): (0, 1),
                  (9, 8, False): (0, 1),
                  (9, 9, False): (0, 1),
                  (9, 10, False): (0, 1),
                  (10, 1, False): (0, 1),
                  (10, 2, False): (0, 1),
                  (10, 3, False): (0, 1),
                  (10, 4, False): (0, 1),
                  (10, 5, False): (0, 1),
                  (10, 6, False): (0, 1),
                  (10, 7, False): (0, 1),
                  (10, 8, False): (0, 1),
                  (10, 9, False): (0, 1),
                  (10, 10, False): (0, 1),
                  (11, 1, False): (0, 1),
                  (11, 2, False): (0, 1),
                  (11, 3, False): (0, 1),
                  (11, 4, False): (0, 1),
                  (11, 5, False): (0, 1),
                  (11, 6, False): (0, 1),
                  (11, 7, False): (0, 1),
                  (11, 8, False): (0, 1),
                  (11, 9, False): (0, 1),
                  (11, 10, False): (0, 1),
                  (12, 1, False): (0, 1),
                  (12, 1, True): (0, 1),
                  (12, 2, False): (0, 1),
                  (12, 2, True): (0, 1),
                  (12, 3, False): (0, 1),
                  (12, 3, True): (0, 1),
                  (12, 4, False): (1, 0),
                  (12, 4, True): (0, 1),
                  (12, 5, False): (1, 0),
                  (12, 5, True): (0, 1),
                  (12, 6, False): (1, 0),
                  (12, 6, True): (0, 1),
                  (12, 7, False): (0, 1),
                  (12, 7, True): (0, 1),
                  (12, 8, False): (0, 1),
                  (12, 8, True): (0, 1),
                  (12, 9, False): (0, 1),
                  (12, 9, True): (0, 1),
                  (12, 10, False): (0, 1),
                  (12, 10, True): (0, 1),
                  (13, 1, False): (0, 1),
                  (13, 1, True): (0, 1),
                  (13, 2, False): (1, 0),
                  (13, 2, True): (0, 1),
                  (13, 3, False): (1, 0),
                  (13, 3, True): (0, 1),
                  (13, 4, False): (1, 0),
                  (13, 4, True): (0, 1),
                  (13, 5, False): (1, 0),
                  (13, 5, True): (0, 1),
                  (13, 6, False): (1, 0),
                  (13, 6, True): (0, 1),
                  (13, 7, False): (0, 1),
                  (13, 7, True): (0, 1),
                  (13, 8, False): (0, 1),
                  (13, 8, True): (0, 1),
                  (13, 9, False): (0, 1),
                  (13, 9, True): (0, 1),
                  (13, 10, False): (0, 1),
                  (13, 10, True): (0, 1),
                  (14, 1, False): (0, 1),
                  (14, 1, True): (0, 1),
                  (14, 2, False): (1, 0),
                  (14, 2, True): (0, 1),
                  (14, 3, False): (1, 0),
                  (14, 3, True): (0, 1),
                  (14, 4, False): (1, 0),
                  (14, 4, True): (0, 1),
                  (14, 5, False): (1, 0),
                  (14, 5, True): (0, 1),
                  (14, 6, False): (1, 0),
                  (14, 6, True): (0, 1),
                  (14, 7, False): (0, 1),
                  (14, 7, True): (0, 1),
                  (14, 8, False): (0, 1),
                  (14, 8, True): (0, 1),
                  (14, 9, False): (0, 1),
                  (14, 9, True): (0, 1),
                  (14, 10, False): (0, 1),
                  (14, 10, True): (0, 1),
                  (15, 1, False): (0, 1),
                  (15, 1, True): (0, 1),
                  (15, 2, False): (1, 0),
                  (15, 2, True): (0, 1),
                  (15, 3, False): (1, 0),
                  (15, 3, True): (0, 1),
                  (15, 4, False): (1, 0),
                  (15, 4, True): (0, 1),
                  (15, 5, False): (1, 0),
                  (15, 5, True): (0, 1),
                  (15, 6, False): (1, 0),
                  (15, 6, True): (0, 1),
                  (15, 7, False): (0, 1),
                  (15, 7, True): (0, 1),
                  (15, 8, False): (0, 1),
                  (15, 8, True): (0, 1),
                  (15, 9, False): (0, 1),
                  (15, 9, True): (0, 1),
                  (15, 10, False): (0, 1),
                  (15, 10, True): (0, 1),
                  (16, 1, False): (0, 1),
                  (16, 1, True): (0, 1),
                  (16, 2, False): (1, 0),
                  (16, 2, True): (0, 1),
                  (16, 3, False): (1, 0),
                  (16, 3, True): (0, 1),
                  (16, 4, False): (1, 0),
                  (16, 4, True): (0, 1),
                  (16, 5, False): (1, 0),
                  (16, 5, True): (0, 1),
                  (16, 6, False): (1, 0),
                  (16, 6, True): (0, 1),
                  (16, 7, False): (0, 1),
                  (16, 7, True): (0, 1),
                  (16, 8, False): (0, 1),
                  (16, 8, True): (0, 1),
                  (16, 9, False): (0, 1),
                  (16, 9, True): (0, 1),
                  (16, 10, False): (0, 1),
                  (16, 10, True): (0, 1),
                  (17, 1, False): (1, 0),
                  (17, 1, True): (0, 1),
                  (17, 2, False): (1, 0),
                  (17, 2, True): (0, 1),
                  (17, 3, False): (1, 0),
                  (17, 3, True): (0, 1),
                  (17, 4, False): (1, 0),
                  (17, 4, True): (0, 1),
                  (17, 5, False): (1, 0),
                  (17, 5, True): (0, 1),
                  (17, 6, False): (1, 0),
                  (17, 6, True): (0, 1),
                  (17, 7, False): (1, 0),
                  (17, 7, True): (0, 1),
                  (17, 8, False): (1, 0),
                  (17, 8, True): (0, 1),
                  (17, 9, False): (1, 0),
                  (17, 9, True): (0, 1),
                  (17, 10, False): (1, 0),
                  (17, 10, True): (0, 1),
                  (18, 1, False): (1, 0),
                  (18, 1, True): (0, 1),
                  (18, 2, False): (1, 0),
                  (18, 2, True): (1, 0),
                  (18, 3, False): (1, 0),
                  (18, 3, True): (1, 0),
                  (18, 4, False): (1, 0),
                  (18, 4, True): (1, 0),
                  (18, 5, False): (1, 0),
                  (18, 5, True): (1, 0),
                  (18, 6, False): (1, 0),
                  (18, 6, True): (1, 0),
                  (18, 7, False): (1, 0),
                  (18, 7, True): (1, 0),
                  (18, 8, False): (1, 0),
                  (18, 8, True): (1, 0),
                  (18, 9, False): (1, 0),
                  (18, 9, True): (0, 1),
                  (18, 10, False): (1, 0),
                  (18, 10, True): (0, 1),
                  (19, 1, False): (1, 0),
                  (19, 1, True): (1, 0),
                  (19, 2, False): (1, 0),
                  (19, 2, True): (1, 0),
                  (19, 3, False): (1, 0),
                  (19, 3, True): (1, 0),
                  (19, 4, False): (1, 0),
                  (19, 4, True): (1, 0),
                  (19, 5, False): (1, 0),
                  (19, 5, True): (1, 0),
                  (19, 6, False): (1, 0),
                  (19, 6, True): (1, 0),
                  (19, 7, False): (1, 0),
                  (19, 7, True): (1, 0),
                  (19, 8, False): (1, 0),
                  (19, 8, True): (1, 0),
                  (19, 9, False): (1, 0),
                  (19, 9, True): (1, 0),
                  (19, 10, False): (1, 0),
                  (19, 10, True): (1, 0),
                  (20, 1, False): (1, 0),
                  (20, 1, True): (1, 0),
                  (20, 2, False): (1, 0),
                  (20, 2, True): (1, 0),
                  (20, 3, False): (1, 0),
                  (20, 3, True): (1, 0),
                  (20, 4, False): (1, 0),
                  (20, 4, True): (1, 0),
                  (20, 5, False): (1, 0),
                  (20, 5, True): (1, 0),
                  (20, 6, False): (1, 0),
                  (20, 6, True): (1, 0),
                  (20, 7, False): (1, 0),
                  (20, 7, True): (1, 0),
                  (20, 8, False): (1, 0),
                  (20, 8, True): (1, 0),
                  (20, 9, False): (1, 0),
                  (20, 9, True): (1, 0),
                  (20, 10, False): (1, 0),
                  (20, 10, True): (1, 0),
                  (21, 1, False): 0,
                  (21, 1, True): (1, 0),
                  (21, 2, False): (1, 0),
                  (21, 2, True): (1, 0),
                  (21, 3, False): (1, 0),
                  (21, 3, True): (1, 0),
                  (21, 4, False): (1, 0),
                  (21, 4, True): (1, 0),
                  (21, 5, False): (1, 0),
                  (21, 5, True): (1, 0),
                  (21, 6, False): (1, 0),
                  (21, 6, True): (1, 0),
                  (21, 7, False): (1, 0),
                  (21, 7, True): (1, 0),
                  (21, 8, False): (1, 0),
                  (21, 8, True): (1, 0),
                  (21, 9, False): (1, 0),
                  (21, 9, True): (1, 0),
                  (21, 10, False): (1, 0),
                  (21, 10, True): 0}


def get_epsilon(N_state_count, N_zero=40):
    return max(0.05, N_zero / (N_zero + N_state_count))


def get_action(Q, state, state_count, action_size):
    random_action = random.randint(0, action_size - 1)
    best_action = np.argmax(Q[state])
    epsilon = get_epsilon(state_count)
    return np.random.choice([best_action, random_action], p=[1. - epsilon, epsilon])


def evaluate_policy(Q, episodes=500000):
    wins = 0
    for _ in range(episodes):
        state = env.reset()

        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action=action)

        if reward > 0:
            wins += 1
    res = wins / episodes
    return res


def get_avg_epsilon(state_count):
    res = 0
    for key in state_count:
        res += get_epsilon(state_count[key])
    return res / len(state_count)


def initWithZeros():
    return np.zeros(env.action_space.n)


def monte_carlo(episodes, evaluate=False, gamma=1., Q=defaultdict(initWithZeros)):
    state_count = defaultdict(float)
    state_action_count = defaultdict(float)

    # for keeping track of our policy evaluations (we'll plot this later)
    evaluations = []
    strategyTables = []

    for i in range(episodes + 1):
        # evaluating a policy is slow going, so let's only do this every 1000 games
        if i == episodes:  # if i % evalSteps == 0:
            if evaluate:
                evaluation = evaluate_policy(Q)
                evaluations.append(evaluation)
            else:
                evaluation = 0
            strategyTables.append(deepcopy(Q))
            if i != 0:
                print('Episode: %d, win rate: %f, avg epsilon: %f' % (i, evaluation, get_avg_epsilon(state_count)))
        # Speicher für die States in denen wir während der Runde waren
        episode = []

        # neue Runde starten
        state = env.reset()
        done = False

        # spielen bis Runde beendet
        while not done:
            # state_count für epsilon berechnung erhöhen
            state_count[state] += 1

            # Wählen einer neuen Aktion
            action = get_action(Q, state, state_count[state], env.action_space.n)

            # nächster Schritt
            new_state, reward, done, _ = env.step(action=action)

            # Speichern was passiert ist
            episode.append((state, action, reward))
            state = new_state

        # Belohnung
        b = 0
        gamma = 0.5
        # durch die States der Runde gehen und entsprechend belohnen
        for s, a, r in reversed(episode):
            b = r + b * gamma
            state_action_count[(s, a)] += 1
            Q[s][a] += (b - Q[s][a]) / state_action_count[(s, a)]

    return Q, evaluations, strategyTables


def playRound(Q, n):
    for rounds in range(n):
        print('New Round')
        prev_observation = env.reset()
        print(prev_observation)
        done = False
        while not done:
            env.render()
            action = np.argmax(Q[prev_observation])
            print(action)
            observation, reward, done, info = env.step(action)
            print(observation)
            prev_observation = observation


def q_table_to_dataframe(Q):
    policy = [['0' for _ in range(10)] for _ in range(18)]
    policy_color = [['0' for _ in range(10)] for _ in range(18)]

    for key in Q:
        if key[2]:
            continue

        hand_sum = key[0]
        dealer_upcard = key[1]
        best_action = np.argmax(Q[key])

        old = True
        if old:
            # wenn eins dann Ass also 11
            if dealer_upcard == 1:
                policy[hand_sum - 4][
                    11 - 2] = 'S' if best_action == 0 else 'H' if best_action == 1 else np.random.choice(
                    ['S', 'H'])
                policy_color[hand_sum - 4][
                    11 - 2] = 'r' if best_action == 0 else 'g' if best_action == 1 else np.random.choice(
                    ['r', 'g'])
            else:
                policy[hand_sum - 4][dealer_upcard - 2] = 'S' if best_action == 0 else 'H'
                policy_color[hand_sum - 4][dealer_upcard - 2] = 'r' if best_action == 0 else 'g'
        else:
            policy[hand_sum - 4][dealer_upcard - 2] = str(round(Q[key][0], 2)) + ' ' + str(
                round(Q[key][1], 2)) + '\n' + str()
            policy_color[hand_sum - 4][dealer_upcard - 2] = 'r' if best_action == 0 else 'g'

    df = pd.DataFrame(policy, index=range(4, 22), columns=range(2, 12))
    df_color = pd.DataFrame(policy_color, index=range(4, 22), columns=range(2, 12))
    return df, df_color


def show_strategy_table(Q):
    df, df_color = q_table_to_dataframe(Q)
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.set_axis_off()
    ax.axis('tight')
    ax.table(cellText=df.values, cellColours=df_color.values, cellLoc="center", rowLabels=df.index,
             colLabels=df.columns, loc='center')
    fig.tight_layout()
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.show()


def write_to_file(file, obj):
    with open(file, 'wb') as myfile:
        pickle.dump(obj, myfile)


def read_file(file):
    with open(file, 'rb') as myfile:
        return pickle.load(myfile)


if only_show_tables:
    strategyTables = read_file('strategies_' + filename + '.list')
    for strategy in strategyTables:
        show_strategy_table(strategy)
    quit()

if len(useDict) > 0:
    Q_mc = read_file(useDict)
    Q_mc, evaluations, strategyTables = monte_carlo(Q=Q_mc, episodes=maxEpisodes, evaluate=evaluate)
else:
    Q_mc, evaluations, strategyTables = monte_carlo(episodes=maxEpisodes, evaluate=evaluate)

for strategy in strategyTables:
    show_strategy_table(strategy)

# playRound(Q_mc, 10)
if evaluate:
    plt.plot([i * evalSteps for i in range(len(evaluations))], evaluations)
    plt.xlabel('episode')
    plt.ylabel('win rate')
    plt.show()

# write to save file
# mydatetime = str(datetime.datetime.now()).replace(':', '-')
write_to_file('dict_' + filename + '.dict', Q_mc)
write_to_file('strategies_' + filename + '.list', strategyTables)
