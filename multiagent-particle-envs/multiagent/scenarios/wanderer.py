import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def set_params(self):
        self.dim_c = 2
        self.n_agents = 1
        self.PENALTY_WEIGHT = 0.0
        self.VISIBLE_WEIGHT = 0.1
        self.L_PROB = 1.0

    def get_world(self):
        pass

    def make_world(self):
        world = self.get_world()
        self.set_params()
        world.dim_c = self.dim_c
        n_landmark = 1  # should be 1

        # add agents
        world.agents = [Agent() for i in range(self.n_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.trade = True
            agent.size = 0.1
        # add landmarks
        world.landmarks = [Landmark() for i in range(n_landmark)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.025
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            agent.color[i % 3] += 0.5 * (i + 1)
            agent.state.energy = 0.0
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    @staticmethod
    def dist2(x, y):
        return np.sum(np.square(x - y))

    def reward(self, agent, world):
        dist2 = self.dist2(agent.state.p_pos, world.landmarks[0].state.p_pos)
        return -dist2 + self.PENALTY_WEIGHT * agent.state.energy

    @staticmethod
    def mask_vector(dv, visible_radius, l_prob):
        def calc_prob(dl, visible_radius):
            eps = 1e-8  # avoid zero divistion
            dl_in_vrad = dl / (visible_radius + eps)
            # return 0.5 * (1. + 0.5 ** ((dl_in_vrad - 1) / L_PROB))
            return 0.5 ** ((dl_in_vrad - 1) / l_prob)

        assert len(dv.shape) == 1
        ndim = dv.shape[0]
        dl = np.sqrt(sum(dv**2))

        if dl <= visible_radius:
            return dv

        p = calc_prob(dl, visible_radius)
        # mask_sign = 2 * (np.random.rand(ndim) <= p).astype(np.int) - 1
        # mask = np.random.rand(ndim) * mask_sign
        mask = (np.random.rand(ndim) <= p).astype(np.int)
        return dv * mask

    def observation(self, agent, world):
        visible_area = abs(self.VISIBLE_WEIGHT * agent.state.energy)
        visible_radius =\
            -1 * np.sign(agent.state.energy) * np.sqrt(visible_area / np.pi)

        entity_pos = []
        for entity in world.landmarks:
            dv = agent.state.p_pos - entity.state.p_pos
            proc_dv = self.mask_vector(dv, visible_radius, self.L_PROB)
            entity_pos.append(proc_dv)

        # multi-agent case
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)


class World2(World):
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise
            # added by JK
            if agent.trade:
                assert self.dim_c == 2  # A:-1, B:0
                if np.argmax(agent.state.c) == 0:
                    agent.state.energy -= 1.0


class World3(World):
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise
            # added by JK
            if agent.trade:
                assert self.dim_c == 3  # A:-1, B:+1, C:0
                if np.argmax(agent.state.c) == 0:
                    agent.state.energy -= 1.0
                elif np.argmax(agent.state.c) == 1:
                    agent.state.energy += 1.0
