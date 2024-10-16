from typing import Tuple

from meltingpot_lib.meltingpot.utils.policies import policy
from ray.rllib.policy.policy import Policy
import dm_env


class RayModelPolicy(policy.Policy[policy.State]):
  """Policy wrapping an RLLib model for inference.

  Note: Currently only supports a single input, batching is not enabled
  """

  def __init__(self,
               policy_id: str,
               policy: Policy
               ) -> None:

    self._prev_action = 0
    self._policy_id = policy_id
    self._policy = policy

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""
    observations = {
        key: value
        for key, value in timestep.observation.items()
        if (
                'WORLD' not in key
            and
                'INFO' not in key
            and
                'POSITION' not in key
        )
    }
    observations["scenario"] = 0

    action, state, _ = self._policy.compute_single_action(
        observations,
        prev_state,
        prev_action=self._prev_action,
        prev_reward=timestep.reward
    )

    self._prev_action = action
    return action, state

  def initial_state(self) -> policy.State:
    """See base class."""
    self._prev_action = 0
    return self._policy.get_initial_state()

  def close(self) -> None:
    """See base class."""


  def __repr__(self):
    return f"MPPolicy[{self._policy_id}]"
