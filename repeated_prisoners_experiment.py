import os
import pickle
import time
from collections import defaultdict
from typing import List
import numpy as np
from tqdm import tqdm

from tabular_population.deterministic import DeterministicPoliciesPopulation
from probabilities.learnable_distribution import LearnableDistribution
from environments.matrix_form.repeated_prisoners import RepeatedPrisonersDilemmaEnv
from hps import compute_theoretical_learning_rates
from policies.policy import Policy
from policies.tabular_policy import TabularPolicy
from rl_algorithms.policy_iteration import PolicyIteration
from rl_algorithms.policy_gradient import PolicyGradient
import argparse
import multiprocessing as mp

from evaluation.pyplot_utils import make_grouped_boxplot, make_grouped_plot, plot_distribution
from scenarios.tabular_scenarios import Scenario, ScenarioFactory


def main(policy_lr, prior_lr, lambda_, clip, n_seeds=1, episode_length=10, n_steps=1000,
         env_seed=0):

    approaches = [
        dict(
            scenario_distribution_optimization="minimax_regret",
            use_regret=True,
            policy_lr=policy_lr,
            prior_lr=prior_lr,
            n_steps=n_steps,
            clip=clip,
        ),
        dict(
            scenario_distribution_optimization="maximin_utility",
            use_regret=False,
            policy_lr=policy_lr,
            prior_lr=prior_lr,
            n_steps=n_steps,
            clip=clip,

        ),
        dict(
            scenario_distribution_optimization="uniform",
            use_regret=False,
            policy_lr=policy_lr,
            prior_lr=0.,
            n_steps=n_steps,
            clip=clip,
        ),
        dict(
            scenario_distribution_optimization="self_play",
            use_regret=False,
            policy_lr=policy_lr,
            prior_lr=0.,
            n_steps=n_steps,
            self_play=True,
            clip=clip,
        ),
        dict(
            scenario_distribution_optimization="fictitious_play",
            use_regret=False,
            policy_lr=policy_lr,
            prior_lr=0.,
            n_steps=n_steps,
            self_play=True,
            clip=clip,
        ),
        dict(
            scenario_distribution_optimization="random",
            use_regret=False,
            policy_lr=0.,
            prior_lr=0.,
            n_steps=2,
            clip=clip,
        ),
        dict(
            scenario_distribution_optimization="tft",
            use_regret=False,
            policy_lr=0.,
            prior_lr=0.,
            n_steps=2,
            clip=clip,
        ),
        dict(
            scenario_distribution_optimization="cud",
            use_regret=False,
            policy_lr=0.,
            prior_lr=0.,
            n_steps=2,
            clip=clip,
        ),
    ]

    name = f"prisoners_n_steps={n_steps}_env_seed={env_seed}_lr={policy_lr:.0E}_beta_lr={prior_lr:.0E}"

    all_jobs = []
    for seed in range(n_seeds):
        seeded_configs = [{
            "seed": seed,
            "lambda_": lambda_,
            "episode_length": episode_length,
            "run_name": name,
            "scenario_distribution_optimization": approach["scenario_distribution_optimization"],
            "job": prisoners_experiment_with_config #if idx < len(approaches)-1
            #else repeated_prisoners_best_solution_with_config
        } for approach in approaches]

        for config, approach in zip(seeded_configs, approaches):
            config.update(**approach)
        all_jobs.extend(seeded_configs)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # results[approach][train/test][metric][data list]

    with mp.Pool(os.cpu_count(), maxtasksperchild=1) as p:
        for ret, config in tqdm(
                zip(p.imap(run_job, all_jobs), all_jobs), total=len(all_jobs)
        ):
            ret["config"] = config
            approach = config["scenario_distribution_optimization"]
            for train_or_test, data in ret.items():
                for k, v in data.items():
                    results[approach][train_or_test][k].append(v)

    train_grouped_data = {}
    test_grouped_data_regret = {}
    test_grouped_data_utility = {}


    for approach, metrics in results.items():
        config = metrics.pop("config")

        run_data = {}

        for metric, data in metrics["train"].items():
            stacked = np.stack(data)
            meaned = np.mean(stacked, axis=0)
            run_data[metric] = meaned

        train_grouped_data[approach] = run_data

        # Test

        run_data_r = {}
        run_data_u = {}


        for metric, data in metrics["test"].items():

            data_r = [d["regret"] for d in data]

            data_u = [d["utility"] for d in data]

            stacked_r = np.stack(data_r)
            stacked_u = np.stack(data_u)
            run_data_r[metric] = np.mean(stacked_r, axis=0)
            run_data_u[metric] = np.mean(stacked_u, axis=0)

        test_grouped_data_regret[approach] = run_data_r
        test_grouped_data_utility[approach] = run_data_u

    for policy, test_sets in test_grouped_data_utility.items():
        for test_set, scores in test_sets.items():
            if "beta^*" in test_set:
                print(policy, test_set, "p=", np.mean(scores))
            else:
                print(policy, test_set, "p=", np.mean(scores), "U-=",np.min(scores))
    for policy, test_sets in test_grouped_data_regret.items():
        for test_set, scores in test_sets.items():
            if "beta^*" in test_set:
                pass
            else:
                print(policy, test_set, "R+=",np.max(scores))

    make_grouped_plot(train_grouped_data, name=f"train_{name}")
    make_grouped_boxplot(test_grouped_data_utility, name=f"boxplot_utility_{name}", whiskers=(0, 50), plot_type="utility")
    make_grouped_boxplot(test_grouped_data_regret, name=f"boxplot_regret_{name}", whiskers=(50, 100), plot_type="regret")


def run_job(config):
    job = config.pop("job")
    return job(config)

def prisoners_experiment_with_config(config):
    return prisoners_experiment(**config)

def prisoners_experiment(
        policy_lr=1e-3,
        prior_lr=1e-3,
        use_regret=False,
        self_play=False,
        lambda_=0.5,
        clip=None,
        seed=0,
        episode_length=1,
        n_steps=1000,
        run_name="",
        scenario_distribution_optimization="",
        **kwargs):

    environment = RepeatedPrisonersDilemmaEnv(episode_length=episode_length)

    robust_policy = Policy(environment)

    if scenario_distribution_optimization == "tft":
        robust_policy.action_logits = environment.tit_for_tat * 10000
    elif scenario_distribution_optimization == "cud":
        robust_policy.action_logits = environment.cooperate_then_defect * 10000
    else:
        robust_policy.initialize_uniformly()

    if scenario_distribution_optimization == "fictitious_play":
        smoothed_pi = True
        pi_history_length = 1

    else:
        smoothed_pi = False
        pi_history_length = 64

    rng = np.random.default_rng(seed)
    bg_population = DeterministicPoliciesPopulation(environment)

    num_test_policies = 512
    if episode_length == 1:
        #  with one defecting, one cooperating policy
        policies = np.zeros((2, bg_population.n_states, bg_population.n_actions))
        policies[0, :-1, 0] = 1.  # cooperating
        policies[1, :-1, 1] = 1.  # defecting
        bg_population.policies = policies
    else:

        # one defecting, one cooperating, tit for tat, cooperating then defecting
        policies = np.zeros((9, bg_population.n_states, bg_population.n_actions))
        policies[0, :, 0] = 1. # cooperating
        policies[1, :, 1] = 1. # defecting
        policies[2, :, :] = environment.tit_for_tat
        policies[3, :, :] = 1 - environment.tit_for_tat
        policies[4, :, :] = environment.tit_for_tat
        policies[5, :, :] = 1 - environment.tit_for_tat

        # tit for tat with initial defect
        policies[4, 0, :] = 0, 1
        policies[5, 0, :] = 0, 1

        policies[6, :, :] = environment.cooperate_then_defect
        policies[7, :, :] = environment.defect_then_cooperate

        # random
        policies[8, :, :] = 0.5


        policies[:, :, 1] = 1 - policies[:, :, 0]
        bg_population.policies = policies


    test_populations = [DeterministicPoliciesPopulation(environment)
                        for _ in range(10)]
    for i, test_population in enumerate(test_populations, 1):
        epsilon_net_width = i * 0.1
        test_policies = np.zeros((num_test_policies, bg_population.n_states, bg_population.n_actions))
        for pid in range(num_test_policies):
            base_id = rng.integers(0, len(policies))
            for s in range(bg_population.n_states):
                old_probs = policies[base_id, s]
                p = np.random.random()
                new_probs = old_probs.copy()
                new_probs[:] = p, 1-p
                while np.sum(np.abs(new_probs - old_probs)) >= epsilon_net_width:
                    p = np.random.random()
                    new_probs[:] = p, 1 - p
                test_policies[pid, s] = new_probs

        test_population.policies = test_policies

    algo = PolicyGradient(robust_policy, environment, epsilon=episode_length, learning_rate=policy_lr, lambda_=lambda_,
                           clip=clip)

    # belief over worst teammate policy (all bg individuals and our self)
    belief = LearnableDistribution(len(bg_population.policies) + 1, learning_rate=prior_lr)
    #belief.initialize_randomly()
    if self_play:
        belief.initialize_certain(belief.dim - 1)
    else:
        #belief.initialize_randomly()print
        belief.initialize_uniformly()


    vfs = []
    regret_scores = []
    vf_scores = []

    # Compute best responses for regret
    best_response_vfs = np.empty((len(bg_population.policies) + 1, robust_policy.n_states), dtype=np.float32)
    priors = []
    priors.append(belief().copy())
    best_responses = {}


    for p_id in range(len(bg_population.policies) + 1):
        print(p_id)
        best_response = TabularPolicy(environment)
        best_response.initialize_uniformly()

        policy_history = [
            best_response.get_probs(),
            best_response.get_probs()
        ]

        p_algo = PolicyIteration(best_response, environment, learning_rate=1, epsilon=episode_length)
        if p_id < len(bg_population.policies):
            scenario = bg_population.policies[p_id], (1, 0)
        else:
            scenario = best_response.get_probs(), (0.5, 0.5)
        for i in range(episode_length * 5):
            policy_history.append(best_response.get_probs())
            old_best_response = policy_history.pop(0)

            if p_id == len(bg_population.policies):
                scenario = old_best_response, (0.5, 0.5)
            vf = p_algo.policy_evaluation_for_scenario(scenario)
            p_algo.policy_improvement_for_scenario(scenario, vf)

            if np.allclose(old_best_response, best_response.get_probs()):
                break

        vf = p_algo.policy_evaluation_for_scenario(scenario)

        best_response_vfs[p_id] = vf
        best_responses[p_id] = best_response

    regrets = []
    worst_case_regrets = []
    policy_history = [robust_policy.get_probs()]

    for i in range(n_steps):


        if len(policy_history) > pi_history_length:
            last_pi = policy_history.pop(0)
        else:
            last_pi = policy_history[-1]
        if smoothed_pi:
            smooth_pi = i * last_pi / (i+1) + robust_policy.get_probs() / (i+1)
            policy_history.append(smooth_pi)
        else:
            policy_history.append(robust_policy.get_probs())

        previous_robust_policy = policy_history[-1]

        expected_vf, vf = algo.policy_evaluation_for_prior(bg_population, belief)

        vf_s0 = vf[:, environment.s0]

        all_regrets = best_response_vfs - vf

        regret_s0 = all_regrets[:, environment.s0]

        if use_regret:
            algo.exact_pg(bg_population, belief, vf, previous_copy=previous_robust_policy)
            belief.update_prior(regret_s0, regret=True)
        else:
            algo.exact_pg(bg_population, belief, vf, previous_copy=previous_robust_policy)
            belief.update_prior(vf_s0, regret=False)

        vfs.append(expected_vf[environment.s0])
        regrets.append(np.sum(regret_s0 * belief()))

        vf_scores.append(np.mean(vf_s0))
        regret_scores.append(np.mean(regret_s0))
        worst_case_regrets.append(np.max(regret_s0))

        print(f"--- Iteration {i} ---")

        priors.append(belief(smooth=True))
        a_probs = robust_policy.get_probs()

    algo_p = PolicyGradient(robust_policy, environment, epsilon=episode_length)
    expected_vf, vf = algo_p.policy_evaluation_for_prior(bg_population, belief)
    vf_s0 = vf[:, environment.s0]
    all_regrets = best_response_vfs - vf
    regret_s0 = all_regrets[:, environment.s0]

    minimax_worst_case_distribution_path = (
            run_name + "minimax_regret"
            + "worst_case_distribution.pkl")
    maximin_worst_case_distribution_path = (
            run_name + "maximin_utility"
            + "worst_case_distribution.pkl")

    plot_distribution([priors], timesteps=None, name="prior_overtime_" + scenario_distribution_optimization + run_name + ".png",
                      scenarios=["Cooperating", "Defecting", "Tit-for-Tat", "Tat-for-Tit",
                                 "Tit-for-Tat (defect first)", "Tat-for-Tit (defect first)",
                                 "Cooperating-until-Defected",
                                 "Defecting-until-Cooperated", "Random"])

    if scenario_distribution_optimization == "minimax_regret":
        worst_case_distribution = belief(smooth=True)
        with open(minimax_worst_case_distribution_path, "wb+") as f:
            pickle.dump(worst_case_distribution, f)
    elif scenario_distribution_optimization == "maximin_utility":
        worst_case_distribution = belief(smooth=True)
        with open(maximin_worst_case_distribution_path, "wb+") as f:
            pickle.dump(worst_case_distribution, f)

    while not (os.path.exists(minimax_worst_case_distribution_path)
    and os.path.exists(maximin_worst_case_distribution_path)
    ):
        time.sleep(1)
    time.sleep(np.random.random())

    with open(minimax_worst_case_distribution_path, "rb") as f:
        minimax_worst_case_distribution = pickle.load(f)
    with open(maximin_worst_case_distribution_path, "rb") as f:
        maximin_worst_case_distribution_path = pickle.load(f)

    # EVALUATION

    test_results = {
        r"\[\Sigma(\mathcal{B}^\text{train})\]": {"utility": vf_s0, "regret": regret_s0},
        #r"\[\Sigma^\text{self-play}\]": {"utility": vf_s0[-1:], "regret": regret_s0[-1:]}
    }

    samples = np.random.choice(len(minimax_worst_case_distribution), 10_000, p=maximin_worst_case_distribution_path)
    test_results[r"\[ \beta^*_U \]"] = {
        "utility": vf_s0[samples],
        "regret": regret_s0[samples]
    }

    samples = np.random.choice(len(minimax_worst_case_distribution), 10_000, p=minimax_worst_case_distribution)
    test_results[r"\[ \beta^*_R \]"] = {
        "utility": vf_s0[samples],
        "regret" : regret_s0[samples]
    }

    environment.render_policy(robust_policy.get_probs(), scenario_distribution_optimization)
    sf = ScenarioFactory(environment)

    for i, test_pop in enumerate(test_populations):
        print("Running evaluation...", i)
        test_set = sf.get_test_scenarios(test_pop)
        test_values, test_regrets = evaluate_on_test_set(robust_policy, environment, test_set)
        test_results[r"$\Sigma(\mathcal{B^\text{test}})$ " + str(i*0.1)] = {
            "utility": test_values,
            "regret" : test_regrets
        }

    return {
        "train": {
            "regret": regret_scores,
            "learned distribution regret": regrets,
            "worst-case regret": worst_case_regrets,
        },
        "test": test_results
    }

def evaluate_on_test_set(policy, environment, test_set: List[Scenario]):
    algo_p = PolicyGradient(policy, environment, epsilon=environment.episode_length)
    vfs = []
    regrets = []
    for test_scenario in test_set:

        if test_scenario.num_copies == 2:
            scenario = policy.get_probs(), (0.5, 0.5)
        else:
            scenario = test_scenario.background_policies.policies[0], (1, 0)

        main_policy_vf = algo_p.policy_evaluation_for_scenario(scenario)

        best_response = TabularPolicy(environment)
        best_response.initialize_uniformly()
        p_algo = PolicyIteration(best_response, environment, learning_rate=1, epsilon=environment.episode_length)
        policy_history = [
            best_response.get_probs(),
            best_response.get_probs()
        ]
        if test_scenario.num_copies == 2:

            for i in range(environment.episode_length * 5):
                policy_history.append(best_response.get_probs())
                old_best_response = policy_history.pop(0)
                scenario = old_best_response, (0.5, 0.5)
                best_response_vf = p_algo.policy_evaluation_for_scenario(scenario)
                p_algo.policy_improvement_for_scenario(scenario, best_response_vf)

                if np.allclose(old_best_response, best_response.get_probs()):
                    break
        else:
            scenario = test_scenario.background_policies.policies[0], (1, 0)

            for i in range(environment.episode_length * 5):
                policy_history.append(best_response.get_probs())
                old_best_response = policy_history.pop(0)
                best_response_vf = p_algo.policy_evaluation_for_scenario(scenario)
                p_algo.policy_improvement_for_scenario(scenario, best_response_vf)

                if np.allclose(old_best_response, best_response.get_probs()):
                    break
        vf_s0 = main_policy_vf[environment.s0]
        regret_s0 = best_response_vf[environment.s0] - vf_s0

        vfs.append(vf_s0)
        regrets.append(regret_s0)
    return np.array(vfs), np.array(regrets)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='repeated_prisoners_experiment',
    )
    parser.add_argument("--policy_lr", type=float, default=1e-2)
    parser.add_argument("--prior_lr", type=float, default=1e-2)
    parser.add_argument("--use_regret", type=bool, default=False)
    parser.add_argument("--lambda_", type=float, default=0.)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--sp", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=2)
    parser.add_argument("--n_seeds", type=int, default=1)

    parser.add_argument("--env_seed", type=int, default=0.)
    parser.add_argument("--auto_hps", action='store_true')
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--clip", type=float, default=0.)

    args = parser.parse_args()

    if args.auto_hps:
        dummy_env = RepeatedPrisonersDilemmaEnv(args.episode_length)
        policy_lr, prior_lr = compute_theoretical_learning_rates(dummy_env, args.epsilon)
    else:
        policy_lr = args.policy_lr
        prior_lr = args.prior_lr


    main(
        policy_lr,
        prior_lr,
        args.lambda_,
        args.clip if args.clip > 0 else None,
        args.n_seeds,
        args.episode_length,
        args.n_steps,
    )
