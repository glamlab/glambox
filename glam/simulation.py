import numpy as np
import pandas as pd
from scipy.stats import invgauss


def simulate_subject(parameters, values, gaze, n_repeats=1, subject=0, boundary=1, error_weight=0.05, error_range=(0, 5000)):

    n_trials, n_items = values.shape

    rts = np.zeros(n_trials * n_repeats) * np.nan
    choices = np.zeros(n_trials * n_repeats) * np.nan
    trial_idx = np.zeros(n_trials * n_repeats) * np.nan
    repeat_idx = np.zeros(n_trials * n_repeats) * np.nan

    running_idx = 0

    for trial in range(n_trials):

        for repeat in range(n_repeats):

            choice, rt = simulate_trial(parameters, values[trial], gaze[trial],
                                        boundary=boundary,
                                        error_weight=error_weight,
                                        error_range=error_range)

            rts[running_idx] = rt
            choices[running_idx] = choice
            trial_idx[running_idx] = trial
            repeat_idx[running_idx] = repeat

            running_idx += 1

    df = pd.DataFrame(dict(subject=np.ones(n_trials*n_repeats) * subject,
                           trial=trial_idx,
                           repeat=repeat_idx,
                           choice=choices,
                           rt=rts))

    for i in range(n_items):
        df['item_value_{}'.format(i)] = np.repeat(values[:, i], n_repeats)
        df['gaze_{}'.format(i)] = np.repeat(gaze[:, i], n_repeats)

    return df


def simulate_trial(parameters, values, gaze, boundary=1, error_weight=0.05, error_range=(0, 5000)):
    v, gamma, s, tau, t0 = parameters
    n_items = len(values)

    if np.random.uniform(0, 1) < error_weight:
        rt = int(np.random.uniform(*error_range))
        choice = np.random.choice(n_items)

    else:
        drifts = expdrift(v, tau, gamma, values, gaze)

        FPTs = np.zeros(n_items) * np.nan

        for i in range(n_items):
            mu = boundary / drifts[i]
            lam = (boundary / s)**2
            FPTs[i] = invgauss.rvs(mu=mu/lam, scale=lam)

        choice = np.argmin(FPTs)
        rt = int(np.round(np.min(FPTs) + t0))

    return choice, rt


def expdrift(v, tau, gamma, values, gaze):
    n_items = len(values)

    absolute = gaze * values + (1. - gaze) * gamma * values
    relative = np.zeros(n_items)

    for i in range(n_items):
        others = np.arange(n_items)[np.arange(n_items) != i].astype(int)
        relative[i] = absolute[i] - np.max(absolute[others])

    scaled = v * 10 / (1 + np.exp(-tau*relative))

    return scaled
