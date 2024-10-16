import itertools
from typing import Any, Iterable, Sized
from itertools import combinations_with_replacement
from dataclasses import dataclass, field
import numpy as np

# @dataclass
# class Scenario:
#     name: str = ""
#     num_focal_players: int = 0
#     background_players: Sized[Any] = field(default_factory=tuple)
#     num_background_players: int = field(init=False)
#
#     def __post_init__(self):
#         self.num_background_players = len(self.background_players)
#         if self.name == "":
#             self.name = f"unnamed_focal={self.num_focal_players}_background={self.num_background_players}"
#
# class ScenarioSet:
#     def __init__(self, team_size:int, background_population):
#         self.scenarios = []
#
#         for num_copies in range(1, team_size + 1):
#
#             num_background = team_size - num_copies
#
#             if num_background == 0:
#                 self.scenarios.append(Scenario(num_focal_players=num_copies))
#             else:
#                 possible_background_players = list(combinations_with_replacement(background_population, r=num_background))
#                 for background_players in possible_background_players:
#                     self.scenarios.append(Scenario(num_focal_players=num_copies, background_players=background_players))
#
#         self.size = len(self.scenarios)
#
#     def sample(self, distribution=None, size=None):
#         return np.random.choice(self.scenarios, size=size, p=distribution)
#
#     def __iter__(self):
#         self.a = 0
#         return self
#
#     def __next__(self):
#         if self.a < self.size:
#             x = self.scenarios[self.a]
#             self.a += 1
#             return x
#         else:
#             raise StopIteration
#
#     def __len__(self):
#         return self.size


from tabular_population.bg_population import BackgroundPopulation


class ScenarioFactory:
    def __init__(self, environment):
        self.env = environment
        self.num_players = 2

    def __call__(self, num_copies):
        background_policies = BackgroundPopulation(self.env)
        background_policies.build_randomly(self.num_players - 1 - num_copies)

        return Scenario(num_copies=num_copies, background_policies=background_policies)

    def get_test_scenarios(self, test_set):
        bg_pop = test_set.policies
        scenarios = []
        for num_copies in range(1, self.num_players + 1):
            for policies in itertools.combinations_with_replacement(list(bg_pop), self.num_players - num_copies):
                bg_policies = BackgroundPopulation(self.env)
                if len(policies)> 0:
                    bg_policies.policies = np.stack(policies)
                scenarios.append(Scenario(num_copies, bg_policies))
        return scenarios


class Scenario:

    def __init__(self, num_copies, background_policies : BackgroundPopulation):
        self.num_copies = num_copies
        self.background_policies = background_policies
