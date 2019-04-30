import numpy as np
import statistics as stat
from matplotlib import pyplot as plt
"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""

class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose
        self.obs_size = 5
        self.batch_size = 5 

    def step(self):
        observation = self.environment.observe()
        action = self.agent.act(observation)
        (reward, stop) = self.environment.act(action)
        next_observation = self.agent.next_observation(observation, action)
        return (observation, action, reward, next_observation, stop)

    def loop(self, games, max_iter):
        cumul_reward = 0.0
        rew_hist = []
        track_games = 0
        for g in range(1, games+1):
            print('Track games...', track_games)
            self.agent.reset()
            self.environment.reset()
            for i in range(1, max_iter+1):
                if self.verbose:
                    print("Simulation step {}:".format(i))
                    self.environment.display()
                (obs, act, rew, next_obs, stop) = self.step()
                if stop == None:
                    stop_val = 0
                elif stop != None:
                    stop_val = 1
                cumul_reward += rew

                pos, smell, breeze, charges = obs
                obs = [pos[0], pos[1], smell, breeze, charges]

                obs = np.reshape(obs, [1, self.obs_size])
                next_obs = np.reshape(next_obs, [1, self.obs_size])
                self.agent.remember(obs, act, rew, next_obs, stop_val)

                if self.verbose:
                    print(" ->       observation: {}".format(obs))
                    print(" ->            action: {}".format(act))
                    print(" ->            reward: {}".format(rew))
                    print(" -> cumulative reward: {}".format(cumul_reward))
                    if stop is not None:
                        print(" ->    Terminal event: {}".format(stop))
                    print()
                if stop is not None:
                    rew_hist.append(rew)
                    break
                if i == max_iter:
                    rew_hist.append(rew)
                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)
            if self.verbose:
                print(" <=> Finished game number: {} <=>".format(g))
                print()
            track_games += 1
        print('Max...', max(rew_hist))
        print('Min...', min(rew_hist))
        print('Avg...', stat.mean(rew_hist))
        print('Std Dev...', stat.stdev(rew_hist))
        plt.plot(rew_hist)
        plt.show()

        return cumul_reward

def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))

class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count)
        self.agents = iter_or_loopcall(agent_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [ False for _ in self.environments ]

    def game(self, max_iter):
        rewards = []
        for (agent, env) in zip(self.agents, self.environments):
            agent.reset()
            env.reset()
            game_reward = 0
            for i in range(1, max_iter+1):
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                game_reward += reward
                if stop is not None:
                    break
            rewards.append(game_reward)
        return sum(rewards)/len(rewards)

    def loop(self, games, max_iter):
        cum_avg_reward = 0.0
        for g in range(1, games+1):
            avg_reward = self.game(max_iter)
            if g > 9*games/10:
                cum_avg_reward += avg_reward
            if self.verbose:
                print("Simulation game {}:".format(g))
                print(" ->            average reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward
