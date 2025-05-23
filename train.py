from agent import Agent
from game import SnakeGame
import matplotlib.pyplot as plt
import torch
import os

MAX_GAMES = 1000 # just set a max amount of games to train on
MODEL_PATH = "results/model.pth"
PLOT_PATH = "results/training.png"

# make sure that the model dir exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def plot_and_save(scores, mean_scores):
    plt.figure()
    plt.title('Training Results')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig(PLOT_PATH)
    plt.close()

def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        agent.model.load_state_dict(torch.load(MODEL_PATH))
        agent.model.eval()

    while agent.n_games < MAX_GAMES:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                torch.save(agent.model.state_dict(), MODEL_PATH)
                print("Model saved!")

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

    # save final plot
    plot_and_save(scores, mean_scores)

if __name__ == '__main__':
    train()