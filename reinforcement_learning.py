from astra import Astra
import numpy as np

LR = 1e-3
goal_steps = 200
initial_games = 10000
score_requirement = 0

def some_random_games_first():
    for episode in range(5):
        env = Astra()
        env.reset()
        for t in range(goal_steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break


def initial_population():
    env = Astra()
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            oct_move = env.get_oct_move_action()
            second_move = None
            if oct_move%env.n_move == env.shuffle:
                second_move = env.move_sample(oct_move)

            observation, reward, done, info = env.step(oct_move, second_move)

            if len(prev_observation) > 0:
                game_memory.append([oct_move, second_move])

            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append(data)
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)



initial_population()