"""
This module implements three UCB policies for deciding which feature to sample an edge from next.
Once a feature is decided, the edge with smallest/largest value of the feature is retrieved
from the sample pool. The reward of an arm pull is 0 if the sampled edge is correct, and
otherwise 1, and this is already reflected by the 'solution_incorrect' column.

We save the first rank of each chosen edge to a 'bandit_rank' column. If an edge is "re-sampled"
using a different feature, we reward the bandit arm, but do not save the rank again. We do this
on the assumption that edges may be "clustered" around this arm, and we want to explore this feature
for a bit, in case there are more, previously unseen, incorrect edges in the cluster.

We continue sampling until each edge in the solution has been sampled at least once. This means the
total number of rounds we might play is equal to the number of edges in
the solution multiplied by the number of features we are testing.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_count_arm_played(played_ranks, arm_name, t, gamma=1):
    """Count the number of times an arm has been played up to round t.

    When gamma=1 this is just a plain count. When gamma<1, this is the
    discounted count using gamma^(t-s) where s is the round the arm was
    played.

    Parameters
    ----------
    played_ranks : Dict[str, List[int]]
        rounds each arm was played at
    arm_name : str
        name of the arm we are counting plays for. Must be a valid value 
        in the 'bandit_arm' column of df.
    t : int
        number of total rounds played until now
    gamma : int, optional
        discount between 0 and 1, by default 1
    """
    arm_played = played_ranks[arm_name]
    return np.sum(gamma ** (t-np.array(arm_played)))

def get_reward_for_arm(rewards, played_ranks, arm_name, t, gamma=1):
    """Get the reward for an arm up to round t.

    When gamma=1 this is just a plain sum. When gamma<1, this is the
    discounted sum using gamma^(t-s) where s is the round the arm was
    played and found in the 'bandit_rank' column and possibly the
    'bandit_resample_rank' column of df.

    Parameters
    ----------
    rewards : Dict[str, List[int]]
        rewards for each arm pull
    played_ranks : Dict[str, List[int]]
        rounds each arm was played at
    arm_name : str
        name of the arm we are counting plays for. Must be a valid value 
        in the 'bandit_arm' column of df.
    t : int
        number of total rounds played until now
    gamma : float, optional
        discount between 0 and 1, by default 1
    """
    arm_rewards = rewards[arm_name]
    ranks = played_ranks[arm_name]
    return np.sum(np.asarray(arm_rewards) * (gamma ** (t - np.asarray(ranks))))

def get_confidence_bound_for_arm(Nt, eta_t, B=1, epsilon=2):
    """Get the confidence bound for an arm up to round t.

    When gamma=1, B=1 and epsilon=2 this is standard UCB1. When gamma<1,
    this is discounted UCB and B should be maximum reward * 2.

    Parameters
    ----------
    Nt : int
        number of times arm has been played
    eta_t: int
        disounted count of all arms played so far
    B : int, optional
        maximum reward of arm, by default 1
    epsilon : int, optional
        some appropriate constant, by default 2
    """
    if eta_t < 1:
        return np.inf
    to_sqrt = (epsilon * np.log(eta_t)) / Nt
    ct = B * np.sqrt(to_sqrt)
    return ct

def get_ucb_for_arm(Nt, discounted_reward, eta_t, B=1, epsilon=2):
    """Get the UCB for an arm at round t.

    When gamma=1, B=1 and epsilon=2 this is standard UCB1. When gamma<1,
    this is discounted UCB and B should be maximum reward * 2.

    Parameters
    ----------
    Nt : float
        discounted number of times arm has been played
    discounted_reward : float
        discounted sum of rewards for this arm
    eta_t: float
        discounted total number of times arms have been played
    t : float
        number of total rounds played so far
    B : float, optional
        maximum reward of arm, by default 1
    epsilon : float, optional
        some appropriate constant, by default 2
    gamma : float, optional
        discount between 0 and 1, by default 1
    """
    if Nt < 1:
        return np.inf
    average_reward = discounted_reward / Nt

    confidence_bound = get_confidence_bound_for_arm(Nt, eta_t, B, epsilon)

    ucb = average_reward + confidence_bound
    return ucb

def get_arm_to_play(
        bandit_arms,
        discounted_arm_played,
        discounted_arm_rewards,
        B=1,
        epsilon=2,
        gamma=1
):
    """Get the arm to play at round t.

    Parameters
    ----------
    bandit_arms: List[str]
        list of column names in df to use as bandit arms
    discounted_arm_played : Dict[str, float]
        discounted number of times each arm has been played
    discounted_arm_rewards : Dict[str, float]
        discounted sum of rewards for each arm
    t : float
        number of total rounds played so far
    B : float, optional
        maximum reward of arm, by default 1
    epsilon : float, optional
        some appropriate constant, by default 2
    gamma : float, optional
        discount between 0 and 1, by default 1
    """
    ucb_values = {}
    for arm in bandit_arms:
        discounted_arm_played[arm] *= gamma
        discounted_arm_rewards[arm] *= gamma
    eta_t = sum(discounted_arm_played.values())
    for arm in bandit_arms:
        ucb_values[arm] = get_ucb_for_arm(
            discounted_arm_played[arm],
            discounted_arm_rewards[arm],
            eta_t,
            B,
            epsilon
        )
        if np.isnan(ucb_values[arm]):
            print('nan')
    arm_to_play = max(ucb_values, key=ucb_values.get)
    discounted_arm_played[arm_to_play] += 1
    return arm_to_play

def initialize_bandit(df, bandit_arms, ascending_sort):
    """Initialize the bandit columns in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe of edges we are sampling from
    bandit_arms : List[str]
        list of column names in df to use as bandit arms
    """
    feature_ranked = {
        bandit_arms[i]:
            df[bandit_arms[i]].sort_values(ascending=ascending_sort[i]) for i in range(len(bandit_arms))
    }

    played_ranks = {
        bandit_arms[i]: [] for i in range(len(bandit_arms))
    }
    rewards = {
        bandit_arms[i]: [] for i in range(len(bandit_arms))
    }

    next_index = {
        bandit_arms[i]: 0 for i in range(len(bandit_arms))
    }
    t = 1
    # play each arm once in "arbitrary" order
    for bandit_arm in bandit_arms:
        while df.loc[feature_ranked[bandit_arm].index[next_index[bandit_arm]], 'bandit_rank'] != -1:
            next_index[bandit_arm] += 1
        edge = feature_ranked[bandit_arm].index[next_index[bandit_arm]]
        df.loc[edge, 'bandit_rank'] = t
        df.loc[edge, 'bandit_arm'] = bandit_arm
        played_ranks[bandit_arm].append(t)
        rewards[bandit_arm].append(int(df.loc[edge, 'solution_incorrect']))
        t += 1
        next_index[bandit_arm] += 1
    return feature_ranked, played_ranks, rewards, next_index, t

def rank_edges_by_ucb(
        df,
        bandit_arms,
        ascending_sort,
        B=1,
        epsilon=2,
        gamma=1
    ):
    """Rank all edges for sampling using UCB arm draws.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe of edges we are sampling from
    B : int, optional
        maximum reward of arm, by default 1
    epsilon : int, optional
        some appropriate constant, by default 2
    gamma : float, optional
        discount between 0 and 1, by default 1
    """
    feature_ranked, played_ranks, rewards, next_index, t = initialize_bandit(df, bandit_arms, ascending_sort)
    discounted_arm_played = {
        arm: get_count_arm_played(played_ranks, arm, t, gamma) for arm in bandit_arms
    }
    discounted_arm_rewards = {
        arm: get_reward_for_arm(rewards, played_ranks, arm, t, gamma) for arm in bandit_arms
    }

    unsampled = (df.bandit_rank == -1).sum()
    while unsampled > 0:
        arm = get_arm_to_play(
            bandit_arms,
            discounted_arm_played,
            discounted_arm_rewards,
            B,
            epsilon,
            gamma
        )
        edge = feature_ranked[arm].index[next_index[arm]]
        next_index[arm] += 1
        if df.loc[edge, 'bandit_rank'] == -1:
            df.loc[edge, 'bandit_rank'] = t
            df.loc[edge, 'bandit_arm'] = arm
        discounted_arm_rewards[arm] += int(df.loc[edge, 'solution_incorrect'])
        t += 1
        unsampled = (df.bandit_rank == -1).sum()

if __name__ == '__main__':
    all_df_pth = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/ducb_w_resolve/'

    all_df_pths = [
        os.path.join(all_df_pth, f) for f in os.listdir(all_df_pth) if f.endswith('.csv')
        and f.startswith('Fluo-N2DL-HeLa_01')
    ]

    for pth in tqdm(all_df_pths):
        ds_name = os.path.basename(pth).replace('_all_edges_with_target_ws_fa_fe.csv', '')
        ds_df = pd.read_csv(pth)
        ds_df['bandit_rank'] = -1
        ds_df['bandit_arm'] = 'None'
        sol_df = ds_df[(ds_df.flow > 0) & (ds_df.u != -1) & (ds_df.u != -3)]

        b = 2
        gamma = 1 - (1 / (4 * np.sqrt(2 * ds_df.shape[0])))
        epsilon = 1/2
        print('Processing', ds_name, 'with gamma', gamma)
        rank_edges_by_ucb(
            sol_df,
            bandit_arms=["cost", "softmax_entropy", "sensitivity_diff", "softmax", "parental_softmax"],
            ascending_sort=[False, False, True, True, True],
            B=b,
            epsilon=epsilon,
            gamma=gamma
        )
        ds_df.loc[sol_df.index, 'bandit_rank'] = sol_df.bandit_rank
        ds_df.loc[sol_df.index, 'bandit_arm'] = sol_df.bandit_arm
        ds_df.to_csv(pth, index=False)
        print(f'Wrote {pth}')
        print('#' * 40)