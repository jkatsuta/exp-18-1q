import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 6
        num_watermelon = 1
        num_speaker = 1
        num_splitter = 2
        # add agents
        world.agents = [Agent() for i in range(num_speaker + num_splitter)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
        # speaker
        world.agents[0].movable = False
        # watermelon splitter
        world.agents[1].silent = True
        world.agents[2].silent = True
        # add a watermelon
        world.landmarks = [Landmark() for i in range(num_watermelon)]
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
        # for agent in world.agents:
            # agent.goal_a = None
            # agent.goal_b = None
        # want listener to go to the goal landmark
        # world.agents[0].goal_a = world.agents[1]  # splitter
        # world.agents[0].goal_b = world.landmarks[0]  # watermelon
        # coloring
        world.agents[0].color = np.array([0.25, 0.25, 0.25])
        world.agents[1].color = np.array([1.0, 0.5, 0.5])
        world.agents[2].color = np.array([0.5, 0.5, 1.0])
        world.landmarks[0].color = np.array([0.15, 0.65, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.movable:
            return self.splitter_reward(agent, world)
        else:
            return self.speaker_reward(agent, world)

    def splitter_reward(self, agent, world):
        # squared distance from splitter to watermelon
        watermelon = world.landmarks[0]
        v_diff = watermelon.state.p_pos - agent.state.p_pos
        dist2 = np.sum(np.square(v_diff))
        return -dist2

    def speaker_reward(self, agent, world):
        rew = 0.
        for other in world.agents:
            if other is agent:
                continue
            rew += self.splitter_reward(other, world)
        return rew

    def observation(self, agent, world):
        # speaker
        if not agent.movable:
            # get the watermelon position of splitter's reference frame
            watermelon = world.landmarks[0]
            goals = []
            for other in world.agents:
                if other is agent:
                    continue
                ref_goal_pos = watermelon.state.p_pos - other.state.p_pos
                goals.append(ref_goal_pos)
                goals.append(other.state.p_vel)
            return np.concatenate(goals)
        # watermelon splitter
        if agent.silent:
            # communication from speaker to splitter
            chars = world.agents[0].state.c
            # convert analog to digital
            chars_digital = np.zeros_like(chars)
            chars_digital[np.argmax(chars)] = 1
            return np.concatenate([chars_digital])
