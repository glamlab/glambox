import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import despine


def plot_fit(data, predictions, prediction_labels=None):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    plot_rt_by_difficulty(data, predictions,
                          xlims=(1.5, 6.5), xlabel_skip=2,
                          ax=axs[0],
                          prediction_labels=prediction_labels)
    plot_pleft_by_left_minus_mean_others(data, predictions,
                                         xlabel_skip=4, xlims=[-6, 6.5], xlabel_start=0,
                                         ax=axs[1])
    plot_pleft_by_left_gaze_advantage(data, predictions,
                                      ax=axs[2])
    plot_corpleft_by_left_gaze_advantage(data, predictions,
                                         ax=axs[3])

    # Labels
    for label, ax in zip(list('ABCD'), axs.ravel()):
        ax.text(-0.15, 1.175, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    fig.tight_layout()

    return fig, axs


def add_difficulty(df):
    """
    Compute trial difficulties and add to DataFrame.

    Maximum value - mean other values.
    In the binary case, this reduces to abs(v0 - v1).

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for item_value_i
    """

    # infer number of items
    value_cols = ([col for col in df.columns
                   if col.startswith('item_value_')])

    values = df[value_cols].values
    values_sorted = np.sort(values, axis=1)
    difficulty = values_sorted[:, -1] - np.mean(values_sorted[:, :-1], axis=1)

    df['difficulty'] = difficulty

    return df.copy()


def plot_rt_by_difficulty(data, predictions=None, ax=None, xlims=(1.5, 8.5), xlabel_skip=2, prediction_labels=None):
    """
    Plot SI1 Data with model predictions
    a) RT by difficulty

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    add_labels = False
    if (prediction_labels is not None):
        if len(prediction_labels) == len(predictions):
            add_labels = True
        else:
            raise ValueError('Number of prediction labels does not match number of prediction datasets.')

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_difficulty(df)

        # Compute summary statistics
        subject_means = df.groupby(['subject', 'difficulty']).rt.mean()
        means = subject_means.groupby('difficulty').mean()[xlims[0]:xlims[1]]
        sems = subject_means.groupby('difficulty').sem()[xlims[0]:xlims[1]]

        x = np.arange(len(means))

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            ax.bar(x, means,
                   linewidth=1, edgecolor='k', facecolor='w',
                   width=0.5)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color='k')

        else:  # plot predictions
            if add_labels:
                ax.plot(x, means, '--o', markerfacecolor='none', label=prediction_labels[i-1])
            else:
                ax.plot(x, means, '--o', markerfacecolor='none')

    ylim = np.mean(np.concatenate([a['rt'].ravel() for a in [data]+predictions]))
    ax.set_ylim(0, ylim*2)
    ax.set_xlabel('Max. rating – mean other ratings')
    ax.set_ylabel('Reaction time (ms)')
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(means.index.values[::xlabel_skip])
    if add_labels:
        ax.legend(loc='upper left', fontsize=12)

    despine()


def add_left_minus_mean_others(df):
    """
    Compute relative value of left item and add to DataFrame.

    Left rating – mean other ratings
    In the binary case, this reduces to v0 - v1.

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for item_value_i
    """

    # infer number of items
    value_cols = ([col for col in df.columns
                   if col.startswith('item_value_')])

    values = df[value_cols].values
    left_minus_mean_others = values[:, 0] - np.mean(values[:, 1:], axis=1)

    df['left_minus_mean_others'] = left_minus_mean_others

    return df.copy()


def plot_pleft_by_left_minus_mean_others(data, predictions=None, ax=None, xlims=[-5, 5], xlabel_skip=2, xlabel_start=1, prediction_labels=None):
    """
    Plot SI1 Data with model predictions
    b) P(left chosen) by left rating minus mean other rating

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    n_items = len([col for col in data.columns
                   if col.startswith('item_value_')])

    add_labels = False
    if (prediction_labels is not None):
        if len(prediction_labels) == len(predictions):
            add_labels = True
        else:
            raise ValueError('Number of prediction labels does not match number of prediction datasets.')

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_left_minus_mean_others(df)
        df['left_chosen'] = df['choice'] == 0

        # Compute summary statistics
        subject_means = df.groupby(
            ['subject', 'left_minus_mean_others']).left_chosen.mean()
        means = subject_means.groupby('left_minus_mean_others').mean()[
            xlims[0]:xlims[1]]
        sems = subject_means.groupby('left_minus_mean_others').sem()[
            xlims[0]:xlims[1]]

        x = np.arange(len(means))

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            ax.plot(x, means, color='k', linewidth=1, ls='--')
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color='k')

        else:  # plot predictions
            if add_labels:
                ax.plot(x, means, '--', markerfacecolor='none', label=prediction_labels[i-1])
            else:
                ax.plot(x, means, '--', markerfacecolor='none')

    ax.axhline(1 / n_items, linestyle='--', color='k', linewidth=1, alpha=0.5)

    ax.set_xlabel('Left rating – mean other ratings')
    ax.set_ylabel('P(left chosen)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x[xlabel_start::xlabel_skip])
    ax.set_xticklabels(means.index.values[xlabel_start::xlabel_skip])
    if add_labels:
        ax.legend(loc='upper left', fontsize=12)

    despine()


def add_left_gaze_advantage(df):
    """
    Compute gaze advantage of left item and add to DataFrame.

    Left relative gaze – mean other relative gaze
    In the binary case, this reduces to g0 - g1.

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for gaze_i
    """

    # infer number of items
    gaze_cols = ([col for col in df.columns
                  if col.startswith('gaze_')])

    gaze = df[gaze_cols].values
    left_gaze_advantage = gaze[:, 0] - np.mean(gaze[:, 1:], axis=1)

    df['left_gaze_advantage'] = left_gaze_advantage

    return df.copy()


def plot_pleft_by_left_gaze_advantage(data, predictions=None, ax=None, n_bins=8, xlabel_skip=2, prediction_labels=None):
    """
    Plot SI1 Data with model predictions
    c) P(left chosen) by left gaze minus mean other gaze

    x-axis label indicate left bound of interval.

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    add_labels = False
    if (prediction_labels is not None):
        if len(prediction_labels) == len(predictions):
            add_labels = True
        else:
            raise ValueError('Number of prediction labels does not match number of prediction datasets.')

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_left_gaze_advantage(df)
        bins = np.linspace(-1, 1, n_bins+1)
        df['left_gaze_advantage_bin'] = pd.cut(df['left_gaze_advantage'],
                                               bins=bins, include_lowest=True,
                                               labels=bins[:-1])
        df['left_chosen'] = df['choice'] == 0

        # Compute summary statistics
        subject_means = df.groupby(
            ['subject', 'left_gaze_advantage_bin']).left_chosen.mean()
        means = subject_means.groupby('left_gaze_advantage_bin').mean()
        sems = subject_means.groupby('left_gaze_advantage_bin').sem()

        x = np.arange(len(means))

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            ax.bar(x, means,
                   linewidth=1, edgecolor='k', facecolor='w',
                   width=0.5)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color='k')

        else:  # plot predictions
            if add_labels:
                ax.plot(x, means, '--o', markerfacecolor='none', label=prediction_labels[i-1])
            else:
                ax.plot(x, means, '--o', markerfacecolor='none')

    ax.set_xlabel('Left gaze – mean other gaze')
    ax.set_ylabel('P(left chosen)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(means.index.values[::xlabel_skip])
    if add_labels:
        ax.legend(loc='upper left', fontsize=12)

    despine()


def add_left_relative_value(df):
    """
    Compute relative value of left item.

    Left item value – mean other item values
    In the binary case, this reduces to v0 - v1.

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for gaze_i
    """

    # infer number of items
    # relative value left
    value_cols = ([col for col in df.columns
                   if col.startswith('item_value_')])
    values = df[value_cols].values
    relative_value_left = values[:, 0] - np.mean(values[:, 1:])
    df['left_relative_value'] = relative_value_left

    return df.copy()


def add_corrected_choice_left(df):
    """
    Compute corrected choice left

    Corrected choice ~ (choice==left) - p(choice==left | left relative item value)

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for gaze_i
    """

    # recode choice
    df['left_chosen'] = df['choice'].values == 0

    # left relative value
    df = add_left_relative_value(df)

    # compute p(choice==left|left relative value)
    subject_value_psychometric = df.groupby(
        ['subject', 'left_relative_value']).left_chosen.mean()
    # place in dataframe
    for s, subject in enumerate(df['subject'].unique()):
        subject_df = df[df['subject'] == subject].copy()
        df.loc[df['subject'] == subject, 'p_choice_left_given_value'] = subject_value_psychometric[
            subject][subject_df['left_relative_value'].values].values

    # compute corrected choice left
    df['corrected_choice_left'] = df['left_chosen'] - \
        df['p_choice_left_given_value']

    return df.copy()


def plot_corpleft_by_left_gaze_advantage(data, predictions=None, ax=None, n_bins=8, xlabel_skip=2, prediction_labels=None):
    """
    Plot SI1 Data with model predictions
    c) Corrected P(choice==left) by left gaze minus mean other gaze
    Corrected P(choice==left) ~ P(choice==left | left final gaze adv.) - P(choice==left | left relative value)

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    add_labels = False
    if (prediction_labels is not None):
        if len(prediction_labels) == len(predictions):
            add_labels = True
        else:
            raise ValueError('Number of prediction labels does not match number of prediction datasets.')

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        # recode choice
        df['left_chosen'] = df['choice'].values == 0
        # left final gaze advantage
        df = add_left_gaze_advantage(df)
        gaze_bins = np.linspace(-1, 1, n_bins+1)
        df['left_gaze_advantage_bin'] = pd.cut(df['left_gaze_advantage'],
                                               bins=gaze_bins, include_lowest=True,
                                               labels=gaze_bins[:-1])
        df['left_chosen'] = df['choice'] == 0
        # corrected choice
        df = add_corrected_choice_left(df)

        # Compute summary statistics
        subject_means = df.groupby(
            ['subject', 'left_gaze_advantage_bin']).corrected_choice_left.mean()
        means = subject_means.groupby('left_gaze_advantage_bin').mean()
        sems = subject_means.groupby('left_gaze_advantage_bin').sem()
        x = np.arange(len(means))

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            ax.bar(x, means,
                   linewidth=1, edgecolor='k', facecolor='w',
                   width=0.5)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color='k')

        else:  # plot predictions
            if add_labels:
                ax.plot(x, means, '--o', markerfacecolor='none', label=prediction_labels[i-1])
            else:
                ax.plot(x, means, '--o', markerfacecolor='none')

    ax.set_xlabel('Left gaze – mean other gaze')
    ax.set_ylabel('Corrected P(left chosen)')
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(means.index.values[::xlabel_skip])
    ax.set_ylim(-1.05, 1.05)
    if add_labels:
        ax.legend(loc='upper left', fontsize=12)

    despine()
