import numpy as np
import collections
from collections import deque
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

"""
Contains the definition of the agent that will run in an
environment.
"""

ACT_UP    = 0
ACT_DOWN  = 1
ACT_LEFT  = 2
ACT_RIGHT = 3
ACT_TORCH_UP    = 4
ACT_TORCH_DOWN  = 5
ACT_TORCH_LEFT  = 6
ACT_TORCH_RIGHT = 7

class RandomAgent:
    def __init__(self):
        """
            learning rate value
            gamma value
        """
        self.learning_rate = .01   # learning rate 
        self.obs_size = 5
        self.action_size = 8

        self.memory = deque(maxlen=2000)
        self.gamma = 0.8    # discount rate
        #self.epsilon = 1.0  # exploration rate
        self.epsilon_a = -1/900
        #self.epsilon_min = 0.01
        #self.epsilon_decay = 0.995
        self.model = self._build_model()

        # Episode is defined as each time we starting moving until we die
        self.n_episode = 0

        # Step is defined as the iteration in each game
        self.step = 0
        self.rand_act_count = 0

        self.cumul_reward = 0
        

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(5, input_dim=self.obs_size, activation='relu'))
        #model.add(Dense(5, activation='relu'))
        #model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.argmax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=5, verbose=0)
        #print('Epsilon...', self.epsilon)
        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

    def reset(self):
        self.step = 0
        self.rand_act_count = 0
        self.n_episode += 1
        if self.n_episode == 1000:
            print(self.cumul_reward)
            # print(self.q_table)

    def act(self, observation):
        self.step += 1
        self.epsilon = 0.5

        position, smell, breeze, charges = observation

        observation_vector = [position[0], position[1], smell, breeze, charges]

        #if np.random.rand() <= self.epsilon:
        if np.random.uniform() >= self.epsilon_a * self.epsilon + 1:
            observation_vector = np.asarray(observation_vector)
            observation_vector = np.reshape(observation_vector, [1, self.obs_size])
            act_values = self.model.predict(observation_vector)
        else:

            self.rand_act_count += 1
            #return random.randrange(self.action_size)
            if smell and charges > 0:
                return np.random.randint(0,8)
            else:
                return np.random.randint(0,4)
        return np.argmax(act_values[0])  # returns action

    def next_observation(self, observation, action):
        position, smell, breeze, charges = observation
        if action >= 4 and charges > 0:
            charges -= 1
            next_position_x, next_position_y = self.next_position(position, action)
            next_observation = (next_position_x, next_position_y, smell, breeze, charges)
        elif action >= 4 and charges == 0:
            position_x = position[0]
            position_y = position[1]
            next_observation = (position_x, position_y, smell, breeze, charges)
        else:
            next_position_x, next_position_y = self.next_position(position, action)
            next_observation = (next_position_x, next_position_y, smell, breeze, charges)
        
        return next_observation

    def next_position(self, position, action):
        x, y = position
        if action == 1 and y <= 2:
            y += 1
        elif action == 2 and y >= 1:
            y -= 1
        elif action == 3 and x >= 1:
            x -= 1
        elif action == 4 and x <= 2:
            x += 1
        return x, y
            

Agent = RandomAgent
