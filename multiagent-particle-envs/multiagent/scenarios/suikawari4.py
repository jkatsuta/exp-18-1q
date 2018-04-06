import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_landmarks = 1
        self.len_memory = 3
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
        # speaker
        world.agents[0].movable = False
        # watermelon splitter
        world.agents[1].silent = True
        # add a watermelon
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]  # splitter
        world.agents[0].goal_b = world.landmarks[0]  # watermelon
        # coloring
        world.agents[0].color = np.array([0.25, 0.25, 0.25])
        world.agents[1].color = np.array([1.0, 0.5, 0.5])
        world.landmarks[0].color = np.array([0.15, 0.65, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        self.world = world

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist2

    @staticmethod
    def onehot(comms):
        if not isinstance(comms[0], np.ndarray):
            comms = [comms]
        one_hot_comms = []
        for c in comms:
            c_onehot = np.zeros_like(c)
            c_onehot[np.argmax(c)] = 1
            one_hot_comms.append(c_onehot)
        return one_hot_comms

    def observation(self, agent, world):
        # speaker's action (= communication) log
        comm_log = self.world.action_trajectory[0][-self.len_memory:]
        if len(comm_log) < self.len_memory:
            for i in range(self.len_memory - len(comm_log)):
                comm_log.insert(i, np.zeros_like(comm_log[0]))
        comm_log_onehot = self.onehot(comm_log)
        # speaker
        if not agent.movable:
            # get the watermelon position of splitter's reference frame
            a = world.agents[0]
            goal_pos = [a.goal_b.state.p_pos - a.goal_a.state.p_pos]
            return np.concatenate(goal_pos + comm_log_onehot)
        # watermelon splitter
        if agent.silent:
            # communication from speaker to splitter
            return np.concatenate(comm_log_onehot)
