import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys
import os
import gym

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

import csv
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import Box, Discrete

DIFFUSION_MODEL = None
DIFFUSION_CONSTS = {}



def parse_args():
    parser = argparse.ArgumentParser("robust multi-agent actor-critic experiments for multiagent environments")
    parser.add_argument("--server-name", type=str, default="MyServer", help="type in what host machine the experiments run on")
    parser.add_argument("--do-not-use-send-email", action="store_true", default=True)
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries in the game")
    parser.add_argument("--good-policy", type=str, default="rmaac", help="policy for good agents in the game")
    parser.add_argument("--adv-policy", type=str, default="rmaac", help="policy of adversaries in the game")
    parser.add_argument("--noise-policy", type=str, default="rmaac", help="policy of state perturbation adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Agent Adam optimizer")
    parser.add_argument("--lr-adv", type=float, default=1e-3, help="learning rate for State Perturbation Adversary Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--variant", type=str, default="maddpg-none",
                        choices=["maddpg-none", "maddpg-earnie", "maddpg-act_adv", "maddpg-obs_adv", "m3ddpg"],
                        help="single entrypoint variant to run")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./model", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    parser.add_argument("--run-id", type=int, default=0, help="ID of the run for multiple seeds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    parser.add_argument("--mode", choices=["train", "test", "collect_diffusion", "train_diffusion"],
                        default="train",
                        help="Run mode: train / test / collect_diffusion / train_diffusion")
    parser.add_argument("--adv-type", choices=["none", "obs", "act", "both"], default="obs", help="Adversary Type: 'none' clean training, 'obs' state uncertainity, 'act' action uncertainity, 'both' action+state uncertainity")
    parser.add_argument("--num-test-runs", type=int, default=3, help="Number of test runs to perform when in test mode")

    # State Perturbation
    parser.add_argument("--noise-type", type=str, default="Linear", help="Linear, Gaussian")
    parser.add_argument("--act-noise", type=float, default=1, help="std for Gaussian noise")
    parser.add_argument("--noise-variance", type=float, default=1, help="variance of gaussian noise, 0.1, 0.2, 0.5, 1, 2, 3")
    parser.add_argument("--constraint-epsilon", type=float, default=0.5, help="the constraint parameter: 0.1, 0.25, 0.5, 0.75, 1, 2")

    # --- Robustness settings ---
    parser.add_argument("--noise-factor", type=str, default="state", choices=["none", "state", "reward"],
                        help="where to apply noise (state/reward/none)")
    parser.add_argument("--test-noise-type", type=str, default="gauss", choices=["gauss", "shift", "uniform"],
                        help="type of noise distribution")
    parser.add_argument("--noise-mu", type=float, default=0, help="mean for Gaussian noise")
    parser.add_argument("--noise-sigma", type=float, default=0, help="std for Gaussian noise")
    parser.add_argument("--noise-shift", type=float, default=0.05, help="shift noise magnitude")
    parser.add_argument("--uniform-low", type=float, default=-0.1, help="low bound for uniform noise")
    parser.add_argument("--uniform-high", type=float, default=0.1, help="high bound for uniform noise")
    parser.add_argument("--llm-disturb-interval", type=int, default=5, help="steps between disturbances")
    parser.add_argument("--num-test-episodes", type=int, default=800, help="number of testing episodes")
        # --- LLM-guided adversary ---
    parser.add_argument("--llm-guide", type=str, default="adversary", choices=["none", "adversary"],
                        help="enable LLM-guided perturbations")
    parser.add_argument("--llm-guide-type", type=str, default="stochastic",
                        choices=["stochastic", "uniform", "constraint"],
                        help="LLM adversarial perturbation type")

    # --- M3DDPG adversarial perturbation ---
    parser.add_argument("--adv-eps", type=float, default=1e-3,
                        help="M3DDPG adversarial perturbation magnitude for good agents")
    parser.add_argument("--adv-eps-s", type=float, default=1e-5,
                        help="M3DDPG adversarial perturbation magnitude for adversary agents")

    # --- EARNIE regularization ---
    parser.add_argument("--use-ernie", action="store_true", default=False,
                        help="enable EARNIE-style adversarial regularization in the actor loss")
    parser.add_argument("--lambda-ernie", type=float, default=0.05,
                        help="weight for the EARNIE regularization term")
    parser.add_argument("--perturb-epsilon", type=float, default=0.001,
                        help="max adversarial observation perturbation for EARNIE")
    parser.add_argument("--perturb-alpha", type=float, default=0.001,
                        help="step size for EARNIE perturbation ascent")
    parser.add_argument("--perturb-num-steps", type=int, default=3,
                        help="number of EARNIE inner-loop ascent steps")

    # --- DDPM action denoiser ---
    parser.add_argument("--diffusion-horizon", type=int, default=25,
                        help="trajectory length H used when training the diffusion model")
    parser.add_argument("--diffusion-steps", type=int, default=100,
                        help="number of forward diffusion steps T")
    parser.add_argument("--diffusion-batch-size", type=int, default=64)
    parser.add_argument("--diffusion-epochs", type=int, default=50)
    parser.add_argument("--diffusion-lr", type=float, default=1e-4)
    parser.add_argument("--diffusion-data-path", type=str, default="./diffusion_data.npz",
                        help="path to save/load (states, actions) trajectories for the denoiser")
    parser.add_argument("--diffusion-model-path", type=str, default="./diffusion_model.pt",
                        help="path to save/load the trained DDPM denoiser")
    parser.add_argument("--skip-diffusion", action="store_true", default=False,
                        help="disable DDPM denoising during the test sweep")
    parser.add_argument("--t-start-list", type=int, nargs="*", default=[20, 40, 60],
                        help="reverse-diffusion start steps to sweep at test time")

    return parser.parse_args()


def apply_variant_defaults(arglist):
    if arglist.variant == "maddpg-none":
        arglist.adv_type = "none"
        arglist.use_ernie = False
    elif arglist.variant == "maddpg-earnie":
        arglist.adv_type = "none"
        arglist.use_ernie = True
    elif arglist.variant == "maddpg-act_adv":
        arglist.adv_type = "act"
        arglist.use_ernie = False
    elif arglist.variant == "maddpg-obs_adv":
        arglist.adv_type = "obs"
        arglist.use_ernie = False
    elif arglist.variant == "m3ddpg":
        arglist.adv_type = "none"
        arglist.use_ernie = False
    else:
        raise ValueError("Unsupported variant: {}".format(arglist.variant))

    if arglist.variant in ("maddpg-none", "maddpg-earnie"):
        arglist.good_policy = "maddpg"
        arglist.adv_policy = "maddpg"
        arglist.noise_policy = "maddpg"
    elif arglist.variant in ("maddpg-act_adv", "maddpg-obs_adv"):
        arglist.good_policy = "rmaac"
        arglist.adv_policy = "rmaac"
        arglist.noise_policy = "rmaac"
    elif arglist.variant == "m3ddpg":
        arglist.good_policy = "mmmaddpg"
        arglist.adv_policy = "mmmaddpg"
        arglist.noise_policy = "mmmaddpg"

    return arglist


API_KEY = ""

def gpt_call(prompt):
    # url = "https://api.openai.com/v1/chat/completions"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": "Bearer {}".format(API_KEY)
    # }
    # data = {
    #     "model": "gpt-3.5-turbo",  # updated supported model
    #     "messages": [
    #         {"role": "system", "content": "You are an adversarial perturbation generator for robust RL. Output only the revised observation as a Python list."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     "temperature": 0.7,
    #     "max_tokens": 200
    # }

    # try:
    #     response = requests.post(url, headers=headers, data=json.dumps(data))
    #     result = response.json()

    #     if "error" in result:
    #         print("OpenAI API error:", result["error"])
    #         return None

    #     if "choices" not in result or len(result["choices"]) == 0:
    #         print("OpenAI API returned no choices:", result)
    #         return None

    #     # gpt-3.5-turbo returns content here:
    #     return result["choices"][0]["message"]["content"].strip()

    # except Exception as e:
    #     print("GPT call failed:", str(e))
    #     return None

    return None

def get_total_action_dim(env):
    """
    Return total joint action dimension across all agents.
    Handles both Box and Discrete action spaces.
    """
    total = 0
    for i in range(env.n):
        space = env.action_space[i]
        if isinstance(space, Box):
            total += int(np.prod(space.shape))
        elif isinstance(space, Discrete):
            total += space.n
        else:
            raise NotImplementedError(
                "Unsupported action space type: {}".format(type(space))
            )
    return total

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    # print("===========policy_num_output==========", num_outputs)
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def mlp_model_adv(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None, constraint_epsilon = 0.5):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return tf.clip_by_value(out, clip_value_min=-constraint_epsilon, clip_value_max=constraint_epsilon)



def get_noise(adv_action_n, var):
    noise = []
    for i in range(len(adv_action_n)):
        noise.append(np.random.normal(adv_action_n[i], var)) 
    return np.array(noise)


def resolve_checkpoint(arglist, load_dir=None, exp_name=None):
    resolved_load_dir = arglist.load_dir if load_dir is None else load_dir
    if resolved_load_dir == "":
        resolved_load_dir = arglist.save_dir

    resolved_exp_name = arglist.exp_name if exp_name is None else exp_name
    return resolved_load_dir, resolved_exp_name

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    p_model = mlp_model
    q_model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        policy_name = arglist.bad_policy if arglist.variant == "m3ddpg" else None
        trainers.append(trainer(
            "r_agent_%d" % i, p_model, q_model, obs_shape_n, env.observation_space, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg'),
            ADV=(arglist.variant == "m3ddpg"),
            policy_name=policy_name,
            variant=arglist.variant))
    for i in range(num_adversaries, env.n):
        policy_name = arglist.good_policy if arglist.variant == "m3ddpg" else None
        trainers.append(trainer(
            "r_agent_%d" % i, p_model, q_model, obs_shape_n, env.observation_space, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg'),
            ADV=(arglist.variant == "m3ddpg"),
            policy_name=policy_name,
            variant=arglist.variant))
    return trainers


def get_adversaries(env, obs_shape_n, arglist):
    adversaries = []

    if arglist.adv_type == "none" or arglist.variant == "m3ddpg":
        return adversaries

    def mlp_model_adv_arg(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None, constraint_epsilon = arglist.constraint_epsilon):
        # This model takes as input an observation and returns values of all actions
        # print("===========adv_num_output==========", num_outputs)
        with tf.variable_scope(scope, reuse=reuse):
            out = input
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
            return tf.clip_by_value(out, clip_value_min=-constraint_epsilon, clip_value_max=constraint_epsilon)
    
    p_model = mlp_model_adv_arg
    q_model = mlp_model
    adversary = MADDPGAgentTrainer
    # observation_space = [gym.spaces.Discrete(env.observation_space[i].shape[0]) for i in range(env.n)]
    # action_space = [gym.spaces.Discrete(env.action_space[i].shape[0]) for i in range(env.n)]

    if arglist.adv_type == "obs":
        act_space = env.observation_space
    elif arglist.adv_type == "act":
        act_space = env.action_space
    else:
        combined_spaces = []
        for obs_space, act_space in zip(env.observation_space, env.action_space):
            # Observation dimension
            if isinstance(obs_space, gym.spaces.Box):
                obs_dim = np.prod(obs_space.shape)
                obs_low = obs_space.low
                obs_high = obs_space.high
            else:
                raise ValueError("Observation space must be Box")

            # Action dimension
            if isinstance(act_space, gym.spaces.Box):
                act_dim = np.prod(act_space.shape)
                act_low = act_space.low
                act_high = act_space.high
            elif isinstance(act_space, gym.spaces.Discrete):
                act_dim = act_space.n
                # represent discrete actions as one-hot continuous vector ∈ [0,1]
                act_low = np.zeros(act_dim)
                act_high = np.ones(act_dim)
            else:
                raise ValueError("Unsupported action space type:")

            # Combine both into a single Box
            low = np.concatenate([obs_low.flatten(), act_low])
            high = np.concatenate([obs_high.flatten(), act_high])
            combined = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            combined_spaces.append(combined)
            act_space = combined_spaces

    for i in range(env.n):
        adversaries.append(adversary(
            "r_adversary_%d" % i, p_model, q_model, obs_shape_n, env.observation_space, act_space, i, arglist,
            local_q_func=(arglist.noise_policy=='ddpg'), ADV=True))
    return adversaries


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # print("env.action_space is {} ".format(env.action_space))
        # print("env.observation_space is {} ".format(env.observation_space))
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        # print("obs_shape_n is {} ".format(obs_shape_n))
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        # print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        
        adversaries = get_adversaries(env, obs_shape_n, arglist)
        # print('Using noise policy {}'.format(arglist.noise_policy))
        # print('There is {} adversaries'.format(str(len(adversaries))))


        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info

        
        adversary_rewards = [[0.0] for _ in range(env.n)]  # individual adversary reward
        adversary_inf = [[[]]] # placeholder for benchmarking info

        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            
            adv_action_n = [adv.action(obs) for adv, obs in zip(adversaries,obs_n)]
            # adv_action_n = np.array(adv_action_n)
            # adv_action_n = np.clip(adv_action_n, a_min = -arglist.constraint_epsilon, a_max = arglist.constraint_epsilon)
            adv_action_n = [np.clip(a, -arglist.constraint_epsilon, arglist.constraint_epsilon) for a in adv_action_n]
            # print("==========adversory actions=======", adv_action_n)
            if arglist.adv_type == "none":
                disturbed_obs_n = obs_n
                action_n = [agent.action(obs) for agent, obs in zip(trainers, disturbed_obs_n)]
                disturbed_action_n = [np.clip(a, -1.0, 1.0) for a in action_n]
            elif arglist.noise_type == "Linear":
                # print("Use Linear Noise")
                disturbed_obs_n = [act+obs for act, obs in zip(adv_action_n,obs_n)]
            elif arglist.noise_type == "Gaussian":
                # print("Use Gaussian Noise")
                noise_n = get_noise(adv_action_n = adv_action_n, var = arglist.noise_variance)
                disturbed_obs_n = [act+obs for act, obs in zip(noise_n,obs_n)]
            else:
                print("No noise")
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,disturbed_obs_n)]
            # action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            
            for i, adv in enumerate(adversaries):
                adv.experience(obs_n[i], adv_action_n[i], -rew_n[i], new_obs_n[i], done_n[i], terminal)
            for i, rew in enumerate(rew_n):
                adversary_rewards[i][-1] += -rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                
                adversary_inf.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished agent benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                # loss = agent.update(trainers, train_step)
                loss = agent.update(trainers+adversaries, train_step)
            
            
            adv_loss = None
            for adv in adversaries:
                adv.preupdate()
            for adv in adversaries:
                # adv_loss = adv.update(adversaries, train_step)
                adv_loss = adv.update(adversaries+trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


def train_multiple_runs(arglist, seed_list):
    """
    Run RMA-AC multiple times with different seeds and save concatenated rewards
    in a single CSV. Includes per-agent rewards.
    """
    all_rewards = {}       # key: run_id, value: list of mean episode rewards
    all_agent_rewards = {} # key: run_id, value: list of lists (per agent)

    for run_id, seed in enumerate(seed_list):
        print("\n=== Starting run {} with seed {} ===".format(run_id, seed))

        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        tf.set_random_seed(seed)

        arglist.run_id = run_id
        arglist.seed = seed

        tf.reset_default_graph()   # reset TF graph
        max_mean_ep_reward = None
        with U.single_threaded_session():
            # Create environment
            env = make_env(arglist.scenario, arglist, arglist.benchmark)
            # print("env.action_space is {} ".format(env.action_space))
            # print("env.observation_space is {} ".format(env.observation_space))
            # Create agent trainers
            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            # print("obs_shape_n is {} ".format(obs_shape_n))
            num_adversaries = min(env.n, arglist.num_adversaries)
            trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
            # print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

            
            adversaries = get_adversaries(env, obs_shape_n, arglist)
            # print('Using noise policy {}'.format(arglist.noise_policy))
            # print('There is {} adversaries'.format(str(len(adversaries))))


            # Initialize
            U.initialize()

            # Load previous results, if necessary
            if arglist.load_dir == "":
                arglist.load_dir = arglist.save_dir
            if arglist.display or arglist.restore or arglist.benchmark:
                print('Loading previous state...')
                U.load_state(arglist.load_dir, exp_name=arglist.exp_name)

            episode_rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
            final_ep_rewards = []  # sum of rewards for training curve
            final_ep_ag_rewards = []  # agent rewards for training curve
            agent_info = [[[]]]  # placeholder for benchmarking info

            
            adversary_rewards = [[0.0] for _ in range(env.n)]  # individual adversary reward
            adversary_inf = [[[]]] # placeholder for benchmarking info

            saver = tf.train.Saver()
            obs_n = env.reset()
            episode_step = 0
            train_step = 0
            t_start = time.time()

            print('Starting iterations...')
            while len(episode_rewards) <= arglist.num_episodes:
                
                adv_out_n = [adv.action(obs) for adv, obs in zip(adversaries,obs_n)]
                # adv_action_n = np.array(adv_action_n)
                # adv_action_n = np.clip(adv_action_n, a_min = -arglist.constraint_epsilon, a_max = arglist.constraint_epsilon)
                adv_out_n = [np.clip(a, -arglist.constraint_epsilon, arglist.constraint_epsilon) for a in adv_out_n]
                # print("==========adversory actions=======", adv_action_n)

                disturbed_obs_n = []
                disturbed_action_n = []

                if arglist.adv_type == "none":
                    disturbed_obs_n = obs_n
                    action_n = [agent.action(obs) for agent, obs in zip(trainers, disturbed_obs_n)]
                    disturbed_action_n = [np.clip(a, -1.0, 1.0) for a in action_n]
                elif arglist.adv_type == "obs":
                    if arglist.noise_type == "Linear":
                        # print("Use Linear Noise")
                        disturbed_obs_n = [act+obs for act, obs in zip(adv_out_n,obs_n)]
                    elif arglist.noise_type == "Gaussian":
                        # print("Use Gaussian Noise")
                        noise_n = get_noise(adv_action_n = adv_out_n, var = arglist.noise_variance)
                        disturbed_obs_n = [act+obs for act, obs in zip(noise_n,obs_n)]
                    else:
                        print("No noise")
                    action_n = [agent.action(obs) for agent, obs in zip(trainers,disturbed_obs_n)]
                    disturbed_action_n = action_n
                elif arglist.adv_type == "act":
                    # get action
                    disturbed_obs_n = obs_n
                    action_n = [agent.action(obs) for agent, obs in zip(trainers,disturbed_obs_n)]
                    # --- apply Δact from adversary ---
                    # disturbed_action_n = [np.clip(0.8*a + 0.2*da, -1, 1)  # clip to env limits
                            # for a, da in zip(action_n, disturbed_action_n)]
                    # action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                    # environment step
                    if arglist.noise_type == "Linear":
                        # print("Use Linear Noise")
                        disturbed_action_n = [act+obs for act, obs in zip(adv_out_n,action_n)]
                    elif arglist.noise_type == "Gaussian":
                        # print("Use Gaussian Noise")
                        noise_n = get_noise(adv_action_n = adv_out_n, var = arglist.noise_variance)
                        disturbed_action_n = [act+obs for act, obs in zip(noise_n,action_n)]
                    else:
                        print("No noise")
                else:
                    for i, adv_out in enumerate(adv_out_n):
                        # --- get dimensions ---
                        
                        # --- Get observation and action dimensions safely ---
                        space_obs = env.observation_space[i] if isinstance(env.observation_space, list) else env.observation_space
                        space_act = env.action_space[i] if isinstance(env.action_space, list) else env.action_space

                        if isinstance(space_obs, gym.spaces.Box):
                            obs_dim = space_obs.shape[0]
                        else:
                            raise ValueError("Unsupported observation space type: {}".format(type(space_obs)))

                        if isinstance(space_act, gym.spaces.Box):
                            act_dim = space_act.shape[0]
                        elif isinstance(space_act, gym.spaces.Discrete):
                            act_dim = space_act.n
                        else:
                            raise ValueError("Unsupported action space type: {}".format(type(space_act)))


                        # --- slice the adversary output ---
                        delta_obs = adv_out[:obs_dim]
                        delta_act = adv_out[obs_dim:]

                        # --- reshape in case flattened ---
                        delta_obs = np.reshape(delta_obs, obs_n[i].shape)
                        delta_act = np.reshape(delta_act, (act_dim,))

                        # --- add perturbations ---
                        disturbed_obs = obs_n[i] + delta_obs
                        disturbed_action = None  # will be applied after agent action

                        disturbed_obs_n.append(disturbed_obs)
                        disturbed_action_n.append(delta_act)  # store Δact for later
                    

                    # Optional debug print
                    # print(f"[ADV{i}] adv_out={adv_out.shape}, Δobs={delta_obs.shape}, Δact={delta_act.shape}")
                    action_n = [agent.action(obs) for agent, obs in zip(trainers,disturbed_obs_n)]
                    # --- apply Δact from adversary ---
                    disturbed_action_n = [np.clip(0.8*a + 0.2*da, -1, 1)  # clip to env limits
                            for a, da in zip(action_n, disturbed_action_n)]

                
                prev_obs_n = obs_n
                new_obs_n, rew_n, done_n, info_n = env.step(disturbed_action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                # collect experience
                for i, agent in enumerate(trainers):
                    agent.experience(prev_obs_n[i], disturbed_action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                obs_n = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew

                
                for i, adv in enumerate(adversaries):
                    adv.experience(prev_obs_n[i], adv_out_n[i], -rew_n[i], new_obs_n[i], done_n[i], terminal)
                for i, rew in enumerate(rew_n):
                    adversary_rewards[i][-1] += -rew

                if done or terminal:
                    obs_n = env.reset()
                    episode_step = 0
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])
                    
                    adversary_inf.append([[]])

                # increment global step counter
                train_step += 1

                # for benchmarking learned policies
                if arglist.benchmark:
                    for i, info in enumerate(info_n):
                        agent_info[-1][i].append(info_n['n'])
                    if train_step > arglist.benchmark_iters and (done or terminal):
                        file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                        print('Finished agent benchmarking, now saving...')
                        with open(file_name, 'wb') as fp:
                            pickle.dump(agent_info[:-1], fp)
                        break
                    continue

                # for displaying learned policies
                if arglist.display:
                    time.sleep(0.1)
                    env.render()
                    continue

                # update all trainers, if not in display or benchmark mode
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    # loss = agent.update(trainers, train_step)
                    loss = agent.update(trainers+adversaries, train_step)
                
                
                adv_loss = None
                for adv in adversaries:
                    adv.preupdate()
                for adv in adversaries:
                    # adv_loss = adv.update(adversaries, train_step)
                    adv_loss = adv.update(adversaries+trainers, train_step)

                # save model, display training output
                if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                    U.save_state(arglist.save_dir, saver=saver, exp_name=arglist.exp_name)
                    # print statement depends on whether or not there are adversaries
                    if num_adversaries == 0:
                        print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    else:
                        print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                            [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                    t_start = time.time()
                    # Keep track of final episode reward
                    mean_episode_reward = np.mean(episode_rewards[-arglist.save_rate:])
                    if max_mean_ep_reward is None or max_mean_ep_reward < mean_episode_reward:
                        max_mean_ep_reward = mean_episode_reward
                        U.save_state(arglist.save_dir, saver=saver, exp_name=arglist.exp_name+"best")
                    final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                    for rew in agent_rewards:
                        final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                    # print("Run {} | steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    #     run_id, train_step, len(episode_rewards), mean_episode_reward, round(time.time()-t_start, 3)))
                    t_start = time.time()

                # saves final episode reward for plotting training curve later
                # if len(episode_rewards) > arglist.num_episodes:
                #     rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                #     with open(rew_file_name, 'wb') as fp:
                #         pickle.dump(final_ep_rewards, fp)
                #     agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                #     with open(agrew_file_name, 'wb') as fp:
                #         pickle.dump(final_ep_ag_rewards, fp)
                #     print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                #     break

        # store rewards for this run
        all_rewards[run_id] = final_ep_rewards
        all_agent_rewards[run_id] = final_ep_ag_rewards
        print("=== Finished run {} ===".format(run_id))

    # --- Save all runs to single CSV (mean rewards only) ---
    os.makedirs(arglist.plots_dir, exist_ok=True)
    exp_name = arglist.exp_name if arglist.exp_name is not None else "default_exp"
    csv_file = os.path.join(arglist.plots_dir, exp_name + "_all_runs_mean.csv")

    max_len = max(len(r) for r in all_rewards.values())

    # pad shorter runs
    for rid in all_rewards:
        if len(all_rewards[rid]) < max_len:
            all_rewards[rid] += [all_rewards[rid][-1]] * (max_len - len(all_rewards[rid]))

    # write CSV
    header = ["episode"] + ["run_{}".format(rid) for rid in sorted(all_rewards.keys())]

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(max_len):
            row = [(i+1) * arglist.save_rate]  # episode number
            for rid in sorted(all_rewards.keys()):
                row.append(all_rewards[rid][i])
            writer.writerow(row)

    print("Saved concatenated mean episode rewards for all runs to {}".format(csv_file))


def testWithoutP(arglist, load_dir=None, exp_name=None):
    tf.reset_default_graph()
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.adv_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        load_dir, exp_name = resolve_checkpoint(arglist, load_dir, exp_name)
        print('Loading trained model from {}'.format(load_dir))
        U.load_state(load_dir, exp_name=exp_name)

        # Parameters for testing
        n_episodes = arglist.num_test_episodes
        max_episode_len = arglist.max_episode_len

        all_rewards = []
        print('Starting testing...')

        for ep in range(n_episodes):
            obs_n = env.reset()
            episode_reward = np.zeros(env.n)
            for step in range(max_episode_len):
                # get actions from trained policies
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                new_obs_n, rew_n, done_n, _ = env.step(action_n)

                episode_reward += rew_n
                obs_n = new_obs_n

                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            # print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))
        return np.mean(np.sum(all_rewards, axis=1))


def testRobustnessOP(arglist, load_dir=None, exp_name=None):
    tf.reset_default_graph()
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.adv_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        load_dir, exp_name = resolve_checkpoint(arglist, load_dir, exp_name)
        print('Loading trained model from {}'.format(load_dir))
        U.load_state(load_dir, exp_name=exp_name)

        # Testing params
        n_episodes = arglist.num_test_episodes
        max_episode_len = arglist.max_episode_len
        all_rewards = []

        # --- Extra for disruption ---
        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        print('Starting testing with robustness perturbations...')

        for ep in range(n_episodes):
            obs_n = env.reset()

            disrupted_obs_n = []
            for i, obs in enumerate(obs_n):
                disrupted_obs_n.append(apply_observation_disruption(
                    obs, 0, env, arglist
                ))

            obs_n = disrupted_obs_n
            episode_reward = np.zeros(env.n)

            for step in range(max_episode_len):
                # get actions
                
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)

                # === Apply your disruption here ===
                disrupted_obs_n = []
                # print("=================== before perturbation ===========")
                # print(new_obs_n)
                for i, obs in enumerate(new_obs_n):
                    disrupted_obs_n.append(apply_observation_disruption(
                        obs, rew_n[i], env, arglist
                    ))

                # print("======disturbed_obs============",disrupted_obs_n)

                # print("=================== after perturbation ===========")
                # print(np.array(disrupted_obs_n) - np.array(new_obs_n))

                # track reward
                episode_reward += rew_n

                obs_n = disrupted_obs_n
                # print("=====obs_n============", obs_n)
                # print(obs_n)

                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            # print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))
        return np.mean(np.sum(all_rewards, axis=1))


def testRobustnessOA(arglist, load_dir=None, exp_name=None):
    tf.reset_default_graph()
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.adv_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        load_dir, exp_name = resolve_checkpoint(arglist, load_dir, exp_name)
        print('Loading trained model from {}'.format(load_dir))
        U.load_state(load_dir, exp_name=exp_name)

        # Testing params
        n_episodes = arglist.num_test_episodes
        max_episode_len = arglist.max_episode_len
        all_rewards = []

        # --- Extra for disruption ---
        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        print('Starting testing with robustness perturbations...')

        for ep in range(n_episodes):
            obs_n = env.reset()

            disrupted_obs_n = []
            for i, obs in enumerate(obs_n):
                disrupted_obs_n.append(apply_observation_disruption(
                    obs, 0, env, arglist
                ))

            obs_n = disrupted_obs_n

            episode_reward = np.zeros(env.n)

            for step in range(max_episode_len):
                # --- Apply observation disruption before action selection ---
                obs_n_disrupted = [
                    apply_observation_disruption(obs, 0, env, arglist)
                    for obs in obs_n
                ]

                # --- Get actions from agents ---
                action_n = [
                    agent.action(obs_dis)
                    for agent, obs_dis in zip(trainers, obs_n_disrupted)
                ]

                # --- Apply action disruption ---
                action_n_disrupted = [
                    apply_action_disruption(action, 0, env, arglist)
                    for action in action_n
                ]

                # --- Environment step ---
                new_obs_n, rew_n, done_n, info_n = env.step(action_n_disrupted)

                # --- Track reward ---
                episode_reward += rew_n
                obs_n = new_obs_n

                # --- Render if needed ---
                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            # print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))
        return np.mean(np.sum(all_rewards, axis=1))
    
def testRobustnessAP(arglist, use_denoiser=False, t_start=40, load_dir=None, exp_name=None):
    tf.reset_default_graph()
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.adv_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        load_dir, exp_name = resolve_checkpoint(arglist, load_dir, exp_name)
        print('Loading trained model from {}'.format(load_dir))
        U.load_state(load_dir, exp_name=exp_name)

        # Testing params
        n_episodes = arglist.num_test_episodes
        max_episode_len = arglist.max_episode_len
        all_rewards = []

        # --- Extra for disruption ---
        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        print('Starting testing with robustness perturbations...')

        for ep in range(n_episodes):
            obs_n = env.reset()
            episode_reward = np.zeros(env.n)

            for step in range(max_episode_len):

                # --- Get actions from agents ---
                # action_n = [
                #     agent.action(obs_dis)
                #     for agent, obs_dis in zip(trainers, obs_n)
                # ]

                # # --- Apply action disruption ---
                # action_n_disrupted = [
                #     apply_action_disruption(action, 0, env, arglist)
                #     for action in action_n
                # ]

                # --- clean MADDPG actions ---
                action_n = [
                    agent.action(obs_dis)
                    for agent, obs_dis in zip(trainers, obs_n)
                ]
                # print("=======MADDPG actions:=========")
                # print(action_n)

                # Number of agents
                n_agents = len(action_n)

                # Action dimension per agent (assume all agents have same action dim)
                action_dim_per_agent = [len(a) for a in action_n]

                # print("Number of agents:", n_agents)
                # print("Action dimension per agent:", action_dim_per_agent)

                # --- adversarial noise ---
                action_n_noisy = [
                    apply_action_disruption(action, 0, env, arglist)
                    for action in action_n
                ]
                action_n_clean = action_n_noisy

                # --- optional DDPM denoising ---
                if use_denoiser:
                    action_vec_noisy = concat_actions(action_n_noisy)
                    state_vec = np.concatenate(obs_n, axis=0)
                    action_vec_clean = diffusion_denoise_action(action_vec_noisy, state_vec, t_start=t_start)
                    action_n_clean = split_actions(action_vec_clean, n_agents, action_dim_per_agent)

                # --- env step ---
                new_obs_n, rew_n, done_n, info_n = env.step(action_n_clean)

                # --- Track reward ---
                episode_reward += rew_n
                obs_n = new_obs_n

                # --- Render if needed ---
                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            # print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))
        return np.mean(np.sum(all_rewards, axis=1))



class TrajectoryDiffusion(nn.Module):
    """DDPM denoiser for joint action trajectories. x:[B,H,Da], cond:[B,Ds]."""
    def __init__(self, horizon, action_dim, cond_dim, hidden_dim=256):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(horizon * action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim))

    def forward(self, x_noisy, t, cond):
        B = x_noisy.shape[0]
        t_emb = self.time_mlp(t.float().unsqueeze(-1) / 1000.0)
        c_emb = self.cond_mlp(cond)
        h = torch.cat([x_noisy.reshape(B, -1), t_emb + c_emb], dim=-1)
        return self.net(h).view(B, self.horizon, self.action_dim)


def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


def q_sample(x0, t, eps, alphas_bar):
    a_bar = alphas_bar[t].view(-1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps


def collect_diffusion_data(arglist):
    """Roll out the trained policy and save clean (state, action) trajectories."""
    tf.reset_default_graph()
    H = arglist.diffusion_horizon
    with U.single_threaded_session():
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        U.initialize()
        load_dir, exp_name = resolve_checkpoint(arglist)
        U.load_state(load_dir, exp_name=exp_name)
        print("[Diffusion] Collecting from checkpoint: {}".format(exp_name))

        state_dim = sum(int(np.prod(s)) for s in obs_shape_n)
        action_dim = get_total_action_dim(env)
        print("[Diffusion] state_dim={}, action_dim={}, horizon={}".format(state_dim, action_dim, H))

        state_trajs, action_trajs = [], []
        for ep in range(arglist.num_episodes):
            obs_n = env.reset()
            ep_states, ep_actions = [], []
            for _ in range(arglist.max_episode_len):
                state_vec = np.concatenate(obs_n, axis=0)
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                action_vec = np.concatenate(action_n, axis=0)
                ep_states.append(state_vec)
                ep_actions.append(action_vec)
                obs_n, _, done_n, _ = env.step(action_n)
                if all(done_n):
                    break
            ep_states = np.asarray(ep_states, dtype=np.float32)
            ep_actions = np.asarray(ep_actions, dtype=np.float32)
            if ep_states.shape[0] < H:
                continue
            state_trajs.append(ep_states[:H])
            action_trajs.append(ep_actions[:H])
            if (ep + 1) % 500 == 0:
                print("[Diffusion] Collected {} episodes".format(ep + 1))

        states = np.stack(state_trajs, axis=0)
        actions = np.stack(action_trajs, axis=0)
        np.savez(arglist.diffusion_data_path, states=states, actions=actions)
        print("[Diffusion] Saved {} trajectories to {}".format(len(state_trajs), arglist.diffusion_data_path))


def train_diffusion(arglist):
    """Train the DDPM denoiser on trajectories from collect_diffusion_data()."""
    data = np.load(arglist.diffusion_data_path)
    states = data["states"]
    actions = data["actions"]
    N, H, Ds = states.shape
    _, _, Da = actions.shape
    print("[Diffusion] Loaded {} trajectories — state_dim={}, action_dim={}".format(N, Ds, Da))

    device = torch.device("cpu")
    states_t = torch.from_numpy(states).float()
    actions_t = torch.from_numpy(actions).float()

    act_mean = actions_t.mean(dim=(0, 1), keepdim=True)
    act_std = actions_t.std(dim=(0, 1), keepdim=True) + 1e-6
    actions_t = (actions_t - act_mean) / act_std

    model = TrajectoryDiffusion(H, Da, Ds).to(device)
    betas, alphas, alphas_bar = make_beta_schedule(arglist.diffusion_steps)
    betas, alphas, alphas_bar = betas.to(device), alphas.to(device), alphas_bar.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=arglist.diffusion_lr)

    for epoch in range(arglist.diffusion_epochs):
        perm = torch.randperm(N)
        states_t, actions_t = states_t[perm], actions_t[perm]
        epoch_loss = 0.0
        for b in range(max(1, N // arglist.diffusion_batch_size)):
            sl = slice(b * arglist.diffusion_batch_size, min(N, (b + 1) * arglist.diffusion_batch_size))
            x0 = actions_t[sl].to(device)
            cond = states_t[sl, 0, :].to(device)
            B = x0.shape[0]
            t = torch.randint(0, arglist.diffusion_steps, (B,), device=device)
            eps = torch.randn_like(x0)
            eps_pred = model(q_sample(x0, t, eps, alphas_bar), t, cond)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * B
        print("[Diffusion] Epoch {}/{} loss={:.6f}".format(epoch + 1, arglist.diffusion_epochs, epoch_loss / N))

    os.makedirs(os.path.dirname(os.path.abspath(arglist.diffusion_model_path)), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "horizon": H, "action_dim": Da, "cond_dim": Ds,
                "diffusion_steps": arglist.diffusion_steps,
                "act_mean": act_mean, "act_std": act_std},
               arglist.diffusion_model_path)
    print("[Diffusion] Saved model to {}".format(arglist.diffusion_model_path))


def load_diffusion_model(arglist):
    global DIFFUSION_MODEL, DIFFUSION_CONSTS
    ckpt = torch.load(arglist.diffusion_model_path, map_location="cpu")
    model = TrajectoryDiffusion(ckpt["horizon"], ckpt["action_dim"], ckpt["cond_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _, alphas, alphas_bar = make_beta_schedule(ckpt["diffusion_steps"])
    betas_full = 1.0 - alphas
    DIFFUSION_MODEL = model
    DIFFUSION_CONSTS = {"betas": betas_full, "alphas": alphas, "alphas_bar": alphas_bar,
                        "act_mean": ckpt["act_mean"], "act_std": ckpt["act_std"],
                        "T": ckpt["diffusion_steps"], "H": ckpt["horizon"]}
    print("[Diffusion] Loaded model from {}".format(arglist.diffusion_model_path))


@torch.no_grad()
def diffusion_denoise_action(noisy_action_vec, state_vec, t_start=40):
    """Denoise a single flat action vector using DDPM reverse diffusion."""
    model = DIFFUSION_MODEL
    C = DIFFUSION_CONSTS
    alphas, alphas_bar = C["alphas"], C["alphas_bar"]

    a = torch.from_numpy(noisy_action_vec).float()
    a = (a - C["act_mean"][0, 0]) / C["act_std"][0, 0]
    x = torch.zeros((1, C["H"], a.shape[0]))
    x[0, 0] = a
    cond = torch.from_numpy(state_vec).float().unsqueeze(0)

    for t in reversed(range(t_start + 1)):
        t_tensor = torch.tensor([t])
        eps_pred = model(x, t_tensor, cond)
        alpha, alpha_bar = alphas[t], alphas_bar[t]
        x0_hat = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
        if t > 0:
            x = torch.sqrt(alpha) * x0_hat + torch.sqrt(1 - alpha) * torch.randn_like(x)
        else:
            x = x0_hat

    clean = x[0, 0] * C["act_std"][0, 0] + C["act_mean"][0, 0]
    return clean.numpy()


def concat_actions(action_n):
    return np.concatenate(action_n, axis=0)


def split_actions(action_vec, n_agents, action_dim_per_agent):
    split, start = [], 0
    for dim in action_dim_per_agent:
        split.append(action_vec[start:start + dim])
        start += dim
    return split


def apply_observation_disruption(observation, reward, env, args):
    obs_orig = np.array(observation, dtype=np.float32)

    # === Apply noise ===
    
    if args.noise_type == "gauss":
        noise = np.random.normal(0, 0.5, size=obs_orig.shape)
        # print(noise)
        obs_orig = obs_orig + noise
    elif args.noise_type == "shift":
        obs_orig = obs_orig + args.noise_shift
    elif args.noise_type == "uniform":
        noise = np.random.uniform(args.uniform_low, args.uniform_high, size=obs_orig.shape)
        obs_orig = obs_orig + noise

        
    return obs_orig


def apply_action_disruption(action, reward, env, args):
    action_orig = np.array(action, dtype=np.float32)
    

    if args.noise_type == "gauss":
        action_orig = action_orig + np.random.normal(0, args.act_noise, size=action_orig.shape)
    elif args.noise_type == "shift":
        action_orig = action_orig + args.noise_shift
    elif args.noise_type == "uniform":
        action_orig = action_orig + np.random.uniform(args.uniform_low, args.uniform_high, size=action_orig.shape)

    return np.clip(action_orig, -1.0, 1.0)

def r2(x):
    return "{:.2f}".format(float(x))




if __name__ == '__main__':
    arglist = parse_args()
    arglist = apply_variant_defaults(arglist)
    # train(arglist)
    if arglist.mode == "train":
        seed_list = [1]  # list of random seeds for multiple runs
        train_multiple_runs(arglist, seed_list)

    elif arglist.mode == "test":
        arglist.noise_type = "gauss"

        use_denoiser = not arglist.skip_diffusion
        if use_denoiser:
            load_diffusion_model(arglist)

        t_start_list = arglist.t_start_list
        act_std_list = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
        csv_filename = "{}_actstd_tstart_sweep.csv".format(arglist.exp_name)
        results = []

        rew_no_noise = testWithoutP(arglist)
        print("Baseline (no noise): {:.3f}".format(rew_no_noise))

        for act_std in act_std_list:
            arglist.act_noise = act_std
            print("\n=== Action noise std = {} ===".format(act_std))

            rew_noisy = testRobustnessAP(arglist, use_denoiser=False)
            print("  Noisy (no denoiser): {:.3f}".format(rew_noisy))

            row = [r2(act_std), r2(rew_no_noise), r2(rew_noisy)]

            if use_denoiser:
                diff_rewards = {}
                for t_start in t_start_list:
                    print("  -> t_start = {}".format(t_start))
                    rew_d = testRobustnessAP(arglist, use_denoiser=True, t_start=t_start)
                    diff_rewards[t_start] = rew_d
                    print("     with denoiser (t_start={}): {:.3f}".format(t_start, rew_d))
                best = max(diff_rewards.values())
                for t_start in t_start_list:
                    row.append(r2(diff_rewards[t_start]))
                row.extend([r2(best),
                             r2(((best - rew_noisy) / abs(rew_noisy)) * 100.0 if rew_noisy else 0.0)])

            results.append(row)

        header = ["action_noise_std", "reward_no_noise", "reward_noise_no_diffusion"]
        if use_denoiser:
            for t_start in t_start_list:
                header.append("reward_with_diff_t{}".format(t_start))
            header += ["best_reward_with_diffusion", "pct_inc_vs_no_diffusion"]

        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)

        print("Saved robustness results to {}".format(csv_filename))

    elif arglist.mode == "collect_diffusion":
        collect_diffusion_data(arglist)

    elif arglist.mode == "train_diffusion":
        train_diffusion(arglist)

# from send_email import *
# if __name__ == '__main__':
#     arglist = parse_args()
#     print(arglist)
#     if arglist.do_not_use_send_email:
#         if not os.path.exists(arglist.save_dir):
#             os.mkdir(arglist.save_dir)
#         train(arglist)
#     else:
#         Emails = ServerEmail(mail_host="smtp.gmail.com", 
#                                     mail_sender="your.email@gmail.com", 
#                                     mail_license="abcdefghijklmnop", 
#                                     mail_receivers="sihong.he@uconn.edu", 
#                                     server_name=arglist.server_name)
#         Emails.send_begin_email(exp_name = arglist.exp_name, arglist = arglist)
#         print("send begin email")
#         try:
#             if not os.path.exists(arglist.save_dir):
#                 os.mkdir(arglist.save_dir)
#             train(arglist)
#             Emails.send_end_email(exp_name = arglist.exp_name, arglist = arglist)
#             print("send finish email")
#         except Exception as e:
#             Emails.send_einfo_email(exp_name = arglist.exp_name, einfo = e)
#             print("send error email")

