import sys
sys.path.append("../")
sys.path.append("../")

import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
import os
import tensorboard_evaluation
# import Evaluation
# from tensorboard_evaluation import *
from agent.networks import MLP
from utils import EpisodeStats
import torch


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random action>    do_training == True => train agent
    """

    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic, env='CartPole')
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats

def train_online(env, agent, num_episodes=1000, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = tensorboard_evaluation.Evaluation(os.path.join(tensorboard_dir, "train"), name='CartPoleRuns', stats=>
     # training

    best_eval_reward = 0.0
    for i in range(num_episodes):
        print("episode: ",i)
        stats = run_episode(env, agent, deterministic=False, do_training=True, rendering=True)
        tensorboard.write_episode_data(i, eval_dict={  "episode_reward" : stats.episode_reward,
                                                                "a_0" : stats.get_action_usage(0),
                                                                "a_1" : stats.get_action_usage(1)})

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_tr>        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...


        # store model.

        if i % eval_cycle == 0:
            avg_eval_reward = 0.0
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
                avg_eval_reward = (avg_eval_reward*j + stats.episode_reward)/(j + 1)
            print("Evaluation Reward for {:.4f}".format(avg_eval_reward))
            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                agent.save(os.path.join(model_dir, "dqn_agent_best.pt"))
                print("Model saved")
        agent.save(os.path.join(model_dir, "dqn_agent_final.pt"))

    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5   # evaluate on 5 episodes
    eval_cycle = 20         # evaluate every 10 episodes

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutiv>



    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    # TODO:
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    Q_network = MLP(state_dim, num_actions).to(device)
    Q_target = MLP(state_dim=4, action_dim=2).to(device)
    # Q_network.to(device)
    # Q_target.to(device)

    # print(next(Q_network.parameters()).device)

    dqn = DQNAgent(Q_network, Q_target, num_actions, device = device, history_length=100000)

    train_online(env=env, agent=dqn)