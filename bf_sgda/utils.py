import numpy as np
class RandomMapping:
    def __init__(self, pids, num_players, with_replacement=False):
        self.pids = pids.copy()
        self.num_players = num_players

        self.with_replacement = with_replacement or (len(pids) < num_players)

        self.mappings = {}
    def __call__(self, agent_id, episode, worker, **kwargs):
        """
        avoids self-playing scenarios
        """

        if episode.episode_id not in self.mappings:
            self.mappings[episode.episode_id] = list(np.random.choice(self.pids, self.num_players, replace=self.with_replacement))

        mapping = self.mappings[episode.episode_id]
        policy_id = mapping.pop()
        if len(mapping) == 0:
            del self.mappings[episode.episode_id]

        return policy_id