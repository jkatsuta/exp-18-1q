import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import wanderer


class Scenario(wanderer.Scenario):
    def set_params(self):
        self.n_agents = 1
        self.PENALTY_WEIGHT = 0.0  # energy
        self.PENALTY_WEIGHT2 = 0.0  # attention
        self.VISIBLE_WEIGHT = 0.1
        self.L_PROB = 1.0

    def get_world(self):
        world = wanderer.World3()
        world.dim_c = 3
        return world

    def reward(self, agent, world):
        dist2 = self.dist2(agent.state.p_pos, world.landmarks[0].state.p_pos)
        rew = -dist2
        rew += self.PENALTY_WEIGHT * agent.state.energy
        rew += self.PENALTY_WEIGHT2 * agent.state.attention
        return rew

    def observation(self, agent, world):
        agent.state.visible_radius = self.calc_visible_radius(agent)

        entity_pos = []
        for entity in world.landmarks:
            dv = agent.state.p_pos - entity.state.p_pos
            proc_dv = self.mask_vector(dv, agent.state.visible_radius, self.L_PROB)
            entity_pos.append(proc_dv)

        # multi-agent case
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            # only in the B case, agnet can get others' position
            if np.argmax(agent.state.c) == 1:
                vec_to_other = other.state.p_pos - agent.state.p_pos
            else:
                vec_to_other = np.zeros_like(other.state.p_pos)
            other_pos.append(vec_to_other)
        return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)
