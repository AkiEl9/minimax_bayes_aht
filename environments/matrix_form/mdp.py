from typing import List, Tuple

import numpy as np

def compute_multiagent_mdp(transition_function, reward_function, policy, opp_index_dict, joint_rewards=(1., 0.)):
    # two players
    # transition_function (state, action1, action2, next_state)
    state_dim = transition_function.shape[0]
    action_dim =  transition_function.shape[1]
    single_agent_transition_function = np.zeros((state_dim, action_dim, state_dim))
    single_agent_reward_function = np.zeros((state_dim, action_dim))
    self_rewards, coop_rewards = joint_rewards

    for state in range(state_dim):
        for action1 in range(action_dim):
            for action2 in range(action_dim):
                for next_state in range(state_dim):

                    single_agent_transition_function[state, action1, next_state] += (
                            policy[opp_index_dict[state], action2] * transition_function[state, action1, action2, next_state]
                    )

                single_agent_reward_function[state, action1] += (
                        self_rewards * reward_function[state, action1, action2] * policy[opp_index_dict[state], action2]
                        + coop_rewards *  policy[opp_index_dict[state], action2] * reward_function[opp_index_dict[state], action2, action1]
                )

                # TODO :ok, check for prisoners
                #print("equal?", reward_function[state, action2, action1], reward_function[opp_index_dict[state], action2, action1])
                #input()

    #single_agent_transition_function /= single_agent_transition_function.sum(axis=2, keepdims=True).sum(axis=1, keepdims=True)



    return single_agent_transition_function, single_agent_reward_function


def induce_mdp_from_scenario(transition_function, reward_function, main_policy, scenario):
    """

    :param transition_function: original POMDP transition function
    :param reward_function: original POMDP reward function
    :param background_policy: policies that are considered background
    :param reward_weights: the reward shaping of each policy
    :return:
    """




if __name__ == "__main__":

    players = 2
    actions = 2
    states = 5

    transition_function = np.random.random((states,) + (actions,)*players + (states,))
    normalization = np.sum(transition_function, axis=0, keepdims=True)
    for i in range(players):
        normalization = np.sum(normalization, axis=i+1, keepdims=True)
    transition_function /= normalization
    policy = np.random.random((states, actions))
    policy /= np.sum(policy, axis= 1, keepdims=True)

    single_agent_tf_1 = compute_multiagent_mdp(transition_function, policy)
    print(single_agent_tf_1.shape, transition_function.shape)
    print(np.sum(single_agent_tf_1[:, :, 0]))