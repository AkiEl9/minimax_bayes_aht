import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Any, TypedDict

from gymnasium.core import ObsType
from matplotlib import pyplot as plt
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import networkx as nx


class StateTreeNode:
    def __init__(self, index):
        self.index = index
        self.children = [None, None, None, None]

class BinaryTreeNode:
    def __init__(self, index):
        self.index = index
        self.children = [None, None]

# Build the tree
def build_tree(depth):
    if depth == 0:
        return None
    root = StateTreeNode(0)
    for i in range(4):
        root.children[i] = build_tree_recursive(root.index * 4 + i + 1, depth - 1)
    return root

def build_tree_recursive(index=0, depth=5):
    if depth == 0:
        return StateTreeNode(index)
    node = StateTreeNode(index)
    for i in range(4):
        node.children[i] = build_tree_recursive(index * 4 + i + 1, depth - 1)
    return node

def build_binary_tree_recursive(index=0, depth=5):
    if depth == 0:
        return BinaryTreeNode(index)
    node = BinaryTreeNode(index)
    for i in range(2):
        node.children[i] = build_binary_tree_recursive(index * 2 + i + 1, depth - 1)
    return node


def navigate_tree(node, actions):
    idx = actions[0] + actions[1] * 2
    return node.children[idx]




class RepeatedPrisonersDilemmaEnv(MultiAgentEnv):
    def __init__(self, episode_length: int):

        self.episode_length = episode_length
        self.current_step = 0
        self.max_reward = 5
        self._agent_ids = {0, 1}
        self.max_score = 5 * episode_length
        self.n_actions = 2

        self.action_space = spaces.Dict(
            {
                i: spaces.Discrete(2) for i in self._agent_ids
            }
        )

        self.observation_space = spaces.Dict(
            {
                i: spaces.Discrete(1 + sum([4**(t) for t in range(episode_length)])) for i in self._agent_ids
            }
        )

        self.tree = build_tree_recursive(depth=episode_length)
        self.opp_tree = build_tree_recursive(depth=episode_length)

        self.transition_function = np.zeros(
            (self.observation_space[0].n, self.action_space[0].n, self.action_space[0].n, self.observation_space[0].n), dtype=np.float32
        )
        self.reward_function = np.zeros((self.observation_space[0].n, self.action_space[0].n, self.action_space[0].n), dtype=np.float32)
        self.gamma = 1.
        self.s0 = 0
        self.s_terminal = self.observation_space[0].n - 1

        self.tit_for_tat = np.full((self.observation_space[0].n, self.action_space[0].n), dtype=np.float32,
                                   fill_value=0)
        self.tit_for_tat[0, 0] = 1
        self.cooperate_then_defect = np.full((self.observation_space[0].n, self.action_space[0].n), dtype=np.float32,
                                             fill_value=0)
        self.defect_then_cooperate = np.full((self.observation_space[0].n, self.action_space[0].n), dtype=np.float32,
                                             fill_value=0)
        self.cooperate_then_defect[0, 0] = 1
        self.defect_then_cooperate[0, 1] = 1


        self.curr_state_to_opp_state = {0:0}

        def setup_rec(node=self.tree, opp_node=self.opp_tree,  depth=self.episode_length,
                      opp_already_defected=False, opp_already_cooperated=False):
            for action1 in range(self.action_space[0].n):
                for action2 in range(self.action_space[0].n):
                    idx = action1 + 2 * action2
                    opp_idx = action2 + 2 * action1
                    next_node = node.children[idx]
                    next_opp_node = opp_node.children[opp_idx]
                    self.curr_state_to_opp_state[next_node.index] = next_opp_node.index

                    if action1 == 0 and action2 == 0:
                        r = self.max_reward - 1
                    elif action1 == 0 and action2 == 1:
                        r = 0
                    elif action1 == 1 and action2 == 0:
                        r = self.max_reward
                    else:
                        r = 1

                    self.reward_function[node.index, action1, action2] = r

                    if depth > 1:
                        self.tit_for_tat[next_node.index, action2] = 1
                        opp_already_defected = opp_already_defected or action2 == 1
                        opp_already_cooperated = opp_already_cooperated or action2 == 0
                        self.cooperate_then_defect[next_node.index, int(opp_already_defected)] = 1
                        self.defect_then_cooperate[next_node.index, 1 - int(opp_already_cooperated)] = 1

                        self.transition_function[node.index, action1, action2, next_node.index] = 1
                        setup_rec(next_node, next_opp_node, depth-1, opp_already_defected, opp_already_cooperated)
                    else:
                        self.transition_function[node.index, action1, action2, self.s_terminal] = 1

        setup_rec()

        self.curr_nodes = [None, None]

        super(RepeatedPrisonersDilemmaEnv, self).__init__()


    def render_policy(self, policy, name):
        assert policy.shape == self.tit_for_tat.shape
        our_color = (0.5 * np.array([102., 154., 250.]) + 0.5 * np.array([222., 222., 222.])) / 255

        def add_edges(G, node, game_node, pos, edge_labels, node_colors, x=0, y=0, layer=1, parent=None, label=None,
                      depth=self.episode_length*2, last_action=None):

            if node is None or (last_action is not None and policy[game_node.index, last_action] < 0.05):
                return pos


            G.add_node(id(node))
            pos[id(node)] = (x, y)
            if layer == 6:
                node_colors[id(node)] = "lightgray"
            else:
                node_colors[id(node)] = "lightcoral" if layer % 2 == 0 else our_color
            if parent:
                weight = 1
                color = our_color if layer % 2 == 0 else "lightcoral"
                if last_action is not None:
                    weight = policy[game_node.index, last_action] * 2

                G.add_edge(id(parent), id(node), color=color, weight=weight)
                if label is not None and weight > 0.1:
                    edge_labels[(id(parent), id(node))] = label

            if last_action is not None and policy[game_node.index, last_action] == 0:
                return pos

            for action in range(self.action_space[0].n):
                # layer is pair if opp, else self
                next_node = node.children[action]

                if layer % 2 == 0:
                    if depth > 1:
                        next_game_node = game_node.children[last_action + 2 * action]

                        if action == 0:
                            pos = add_edges(G, next_node, next_game_node, pos, edge_labels, node_colors, x - 1 / 2**layer, y - 1, layer + 1, node,
                                            f"C", depth-1)

                        elif action == 1:
                            pos = add_edges(G, next_node, next_game_node, pos, edge_labels, node_colors,  x + 1 / 2**layer, y - 1, layer + 1, node,
                                            f"D", depth-1)
                else:

                    if depth > 1:

                        if action == 0:
                            ll = f"{policy[game_node.index, 0]:.1f}"
                            if policy[game_node.index, 0] >= policy[game_node.index, 1]:
                                ll = r"C($\mathbf{" + ll + "}$)"
                            else:
                                ll = "C($" + ll + "$)"

                            next_x = x - 1/ 2**layer
                            next_x = next_x
                            pos = add_edges(G, next_node, game_node, pos, edge_labels, node_colors, next_x, y - 1, layer + 1, node,
                                            ll, depth-1, action)

                        elif action == 1:
                            ll = f"{policy[game_node.index, 1]:.1f}"
                            if policy[game_node.index, 0] <= policy[game_node.index, 1]:
                                ll = "D($\mathbf{" + ll + "}$)"
                            else:
                                ll = "D($" + ll + "$)"

                            next_x = x + 1/ 2**layer
                            next_x = next_x
                            pos = add_edges(G, next_node, game_node, pos, edge_labels, node_colors, next_x, y - 1, layer + 1, node,
                                            ll, depth-1, action)

            return pos

        G = nx.DiGraph()
        pos = {}
        edge_labels = {}
        node_colors = {}

        # Start adding edges from the root
        add_edges(G, build_binary_tree_recursive(0, self.episode_length*2), self.tree,
                  pos, edge_labels, node_colors)

        edges = G.edges()
        colors = [G[u][v]['color'] for u, v in edges]
        weights = [G[u][v]['weight'] for u, v in edges]

        nx.draw(G, pos, with_labels=False, node_size=30, node_color=[node_colors[n] for n in G.nodes()], font_size=10, font_weight="bold",
                arrows=False, edge_color=colors, width=weights)

        # Draw edge labels (i.e., probabilities)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='k', rotate=True,
                                     font_size=4)

        # Show the plot
        plt.savefig(f"{name}.png", dpi=300)
        plt.clf()
        plt.cla()





    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.curr_nodes = [self.tree, self.tree]
        return self._get_observation()

    def step(self, actions: dict):

        # Update the state based on actions
        for i, curr_node in enumerate(self.curr_nodes):
            a = actions.values()
            if i == 1:
                a = reversed(a)
            self.curr_nodes[i] = navigate_tree(curr_node, list(a))

        reward = self._calculate_reward(actions)

        # Move to the next step
        self.current_step += 1

        # Check if it's the last step of the episode
        done = self.current_step == self.episode_length

        obs = {
            i: 0 for i in self._agent_ids
        } if done else self._get_observation()

        return obs, reward, done, done, {}


    def _get_observation(self):

        return {
            i: curr_node.index for i, curr_node in enumerate(self.curr_nodes)
        }

    def _calculate_reward(self, actions: dict) -> Tuple[float, float]:
        # Calculate the reward based on the actions

        if actions[0] == 0 and actions[1] == 0:
            return (self.max_reward - 1, self.max_reward - 1)  # Both players cooperate
        elif actions[0] == 0 and actions[1] == 1:
            return (0, self.max_reward)  # Player 1 cooperates, player 2 defects
        elif actions[0] == 1 and actions[1] == 0:
            return (self.max_reward, 0)  # Player 1 defects, player 2 cooperates
        else:
            return (1, 1)  # Both players defect


if __name__ == "__main__":

    # Example usage:
    episode_length = 3
    env = RepeatedPrisonersDilemmaEnv(episode_length)

    env.render_policy(env.tit_for_tat, "titfortat")

    # Reset the environment
    # obs = env.reset()
    #
    # for _ in range(episode_length):
    #     # Take random actions for both players
    #     print(obs)
    #     actions = env.action_space.sample()
    #     # Step through the environment
    #     obs, reward, done, _, _ = env.step({
    #         0:1,
    #         1:1
    #     })
    #     #print(f"Step: {env.current_step}, Actions: {actions}, Reward: {reward}, Done: {done}")
    #
    # # Close the environment
    # env.close()
