#!/usr/bin/python
import numpy as np
import pandas as pd
from pymc3.stats import hpd, summary
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table


def cm2inch(*tupl):
    """
    Convert a value or tuple of values from cm
    to inches.

    Source: https://stackoverflow.com/a/22787457

    Input
    ---
    tupl : float, int or tuple of arbitrary size
        Values to convert

    Returns
    ---
    Converted values in inches.
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def add_difficulty(df, bins=7, return_bins=False):
    """
    Add trial difficulty (defined as the difference 
    between the maximum trial value and the maximum
    of all others) to a response dataframe

    Input
    ---
    df : dataframe
        response data

    bins : int or array_like, optional
        defining the bins to use when computing
        the value difference,
        if an int is given, this many bins will be
        created,
        defaults to 7

    return_bins : bool, optional
        whether or not to return the bins

    Returns
    ---
    copy of df (and bins if return_bins=True)
    """

    # infer number of items
    value_cols = ([col for col in df.columns if col.startswith('item_value_')])

    values = df[value_cols].values
    values_sorted = np.sort(values, axis=1)
    difficulty = values_sorted[:, -1] - np.max(values_sorted[:, :-1], axis=1)

    if isinstance(bins, (int, float)):
        bins = np.linspace(np.min(difficulty), np.max(difficulty), bins)
        bins = np.round(bins, 2)
    difficulty_binned = pd.cut(difficulty, bins)
    df['difficulty'] = bins[difficulty_binned.codes]

    if not return_bins:
        return df.copy()
    else:
        return df.copy(), bins


def plot_rt_by_difficulty(bar_data,
                          line_data=None,
                          ax=None,
                          xlims=None,
                          ylims=None,
                          xlabel_skip=2,
                          bins=7,
                          fontsize=7,
                          line_labels=None,
                          line_colors=None,
                          line_markers=None,
                          line_ls=None,
                          line_alphas=None,
                          line_lws=None):
    """
    Plot: RT ~ (max value - max value othres)

    Input
    ---
    bar_data : dataframe
        response data to plot as bars

    line_data : list of dataframes, optional
        response data to plot as colored lines

    ax : matplotlib axis, optional
        matplotlib axis object to use for plottig,
        if None, new axis is created,
        defaults to None

    xlims : tuple, optional
        limits for x-axis

    ylims : tuple, optional
        limits for y-axis

    x_label_skip : int, optional
        how many x-ticklabels to skip,
        defaults to 2 (indicating that every
        second x-ticklabel is plotted)

    bins : int or array_like, optional
        x-bins to use for plotting,
        if an int is given, this many 
        bins will be created,
        defaults to 7

    fontsize : int, optional
        fontsize for plotting,
        defaults to 7

    line_labels : array_like, strings, optional
        legend labels for line_data

    line_colors : array_like, strings, optional
        line colors to use for line_data
        (see matplotlib documentation)

    line_markers : array_like, strings, optional
        line markers to use for line_data
        (see matplotlib documentation)

    line_ls : array_like, strings, optional
        line styles to use for line_data
        (see matplotlib documentation)

    line_alphas : array_like, floats, optional
        alpha levels to use for lines of line_data

    line_lws : array_like, floats, optional
        width of lines of line_data

    Returns
    ---
    matplotlib axis object
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if line_data is None:
        dataframes = [bar_data]
    elif isinstance(line_data, list):
        dataframes = [bar_data] + line_data
    else:
        dataframes = [bar_data] + [line_data]

    if line_lws is None:
        line_lws = [1 for i in range(len(dataframes) - 1)]

    if line_colors is None:
        line_colors = ['C{}'.format(i) for i in range(len(dataframes) - 1)]

    if line_ls is None:
        line_ls = ['-' for i in range(len(dataframes) - 1)]

    if line_markers is None:
        line_markers = ['o' for i in range(len(dataframes) - 1)]

    if line_alphas is None:
        line_alphas = [0.75 for i in range(len(dataframes) - 1)]

    add_labels = False
    if (line_labels is not None):
        if len(line_labels) == len(line_data):
            add_labels = True
        else:
            raise ValueError(
                'Number of prediction labels does not match number of prediction datasets.'
            )

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        if i == 0:
            df, bins = add_difficulty(df, bins=bins, return_bins=True)
        else:
            df = add_difficulty(df, bins=bins)

        # Compute summary statistics
        subject_means = df.groupby(['subject', 'difficulty']).rt.mean()
        means = subject_means.groupby('difficulty').mean()
        sems = subject_means.groupby('difficulty').sem()
        # make sure all

        x = means.index.values
        if xlims is None:
            xlims = [x.min(), x.max()]
        if x.min() < xlims[0]:
            xlims[0] = x.min()
        if x.max() > xlims[1]:
            xlims[1] = x.max()

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            barwidth = (xlims[1] - xlims[0]) / x.size * 0.6
            ax.bar(x,
                   means,
                   linewidth=1,
                   edgecolor='k',
                   facecolor='w',
                   width=barwidth)
            ax.vlines(x, means - sems, means + sems, linewidth=1, color='k')

            # adapt bardwidth
            xlims[0] -= (1.5 * barwidth)
            xlims[1] += (1.5 * barwidth)

        else:  # plot predictions
            if add_labels:
                ax.plot(x,
                        means,
                        marker=line_markers[i - 1],
                        markerfacecolor=line_colors[i - 1],
                        color=line_colors[i - 1],
                        ls=line_ls[i - 1],
                        label=line_labels[i - 1],
                        alpha=line_alphas[i - 1],
                        lw=line_lws[i - 1])
            else:
                ax.plot(x,
                        means,
                        marker=line_markers[i - 1],
                        markerfacecolor=line_colors[i - 1],
                        color=line_colors[i - 1],
                        ls=line_ls[i - 1],
                        alpha=line_alphas[i - 1],
                        lw=line_lws[i - 1])

    if ylims is None:
        ylims = np.mean(np.concatenate(
            [a['rt'].ravel() for a in dataframes])) * 2
    ax.set_ylim(ylims)
    ax.set_xlabel('Max. value –\nmax. value others', fontsize=fontsize)
    ax.set_ylabel('Response time (s)', fontsize=fontsize)
    ax.set_xlim(xlims)
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(means.index.values[::xlabel_skip])
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize * 0.7, frameon=False)
    sns.despine(ax=ax)

    return ax


def add_value_minus_max_others(df, bins=7, return_bins=False):
    """
    Add trial difference between item's value
    and maximum value of all other items in a trial
    to response data

    Input
    ---
    df : dataframe
        response data

    bins : int or array_like, optional
        defining the bins to use when computing
        the value difference,
        if an int is given, this many bins will be
        created,
        defaults to 7

    return_bins : bool, optional
        whether or not to return the bins

    Returns
    ---
    copy of df (and bins if return_bins=True)
    """

    # infer number of items
    value_cols = ([col for col in df.columns if col.startswith('item_value_')])
    n_items = len(value_cols)

    values = df[value_cols].values
    values_minus_max_others = np.zeros_like(values)

    for t in np.arange(values.shape[0]):
        for i in np.arange(n_items):
            values_minus_max_others[t, i] = values[t, i] - \
                np.max(values[t, np.arange(n_items) != i])

    if isinstance(bins, (int, float)):
        # n_bins = np.min(
        #     [np.unique(values_minus_max_others.ravel()).size, bins])
        bins = np.linspace(np.min(values_minus_max_others.ravel()),
                           np.max(values_minus_max_others.ravel()), bins)
        bins = np.round(bins, 2)
    values_minus_max_others_binned = pd.cut(values_minus_max_others.ravel(),
                                            bins)
    values_minus_max_others_binned = bins[
        values_minus_max_others_binned.codes]
    values_minus_max_others_binned = values_minus_max_others_binned.reshape(
        values_minus_max_others.shape)

    for i in np.arange(n_items):
        df['value_minus_max_others_{}'.format(
            i)] = values_minus_max_others_binned[:, i]

    if not return_bins:
        return df.copy()
    else:
        return df.copy(), bins


def add_value_minus_mean_others(df, bins=7, return_bins=False):
    """
    Add trial difference between item's value
    and mean value of all other items in a trial
    to response data

    Input
    ---
    df : dataframe
        response data

    bins : int or array_like, optional
        defining the bins to use when computing
        the value difference,
        if an int is given, this many bins will be
        created,
        defaults to 7

    return_bins : bool, optional
        whether or not to return the bins

    Returns
    ---
    copy of df (and bins if return_bins=True)
    """

    # infer number of items
    value_cols = ([col for col in df.columns if col.startswith('item_value_')])
    n_items = len(value_cols)

    values = df[value_cols].values
    values_minus_mean_others = np.zeros_like(values)

    for t in np.arange(values.shape[0]):
        for i in np.arange(n_items):
            values_minus_mean_others[t, i] = values[t, i] - \
                np.mean(values[t, np.arange(n_items) != i])

    if isinstance(bins, (int, float)):
        # n_bins = np.min(
        #     [np.unique(values_minus_mean_others.ravel()).size, bins])
        bins = np.linspace(np.min(values_minus_mean_others.ravel()),
                           np.max(values_minus_mean_others.ravel()), bins)
        bins = np.round(bins, 2)
    values_minus_mean_others_binned = pd.cut(values_minus_mean_others.ravel(),
                                            bins)
    values_minus_mean_others_binned = bins[
        values_minus_mean_others_binned.codes]
    values_minus_mean_others_binned = values_minus_mean_others_binned.reshape(
        values_minus_mean_others.shape)

    for i in np.arange(n_items):
        df['value_minus_mean_others_{}'.format(
            i)] = values_minus_mean_others_binned[:, i]

    if not return_bins:
        return df.copy()
    else:
        return df.copy(), bins


def plot_pchoose_by_value_minus_max_others(bar_data,
                                           line_data=None,
                                           ax=None,
                                           bins=7,
                                           xlims=None,
                                           ylims=None,
                                           xlabel_skip=2,
                                           fontsize=7,
                                           line_labels=None,
                                           line_colors=None,
                                           line_lws=None,
                                           line_ls=None,
                                           line_alphas=None,
                                           line_markers=None):
    """
    Plot: P(choose best) ~ (item value - max. value othres)

    Input
    ---
    bar_data : dataframe
        response data to plot as bars

    line_data : list of dataframes, optional
        response data to plot as colored lines

    ax : matplotlib axis, optional
        matplotlib axis object to use for plottig,
        if None, new axis is created,
        defaults to None

    bins : int or array_like, optional
        x-bins to use for plotting,
        if an int is given, this many 
        bins will be created,
        defaults to 7

    xlims : tuple, optional
        limits for x-axis

    ylims : tuple, optional
        limits for y-axis

    x_label_skip : int, optional
        how many x-ticklabels to skip,
        defaults to 2 (indicating that every
        second x-ticklabel is plotted)

    fontsize : int, optional
        fontsize for plotting,
        defaults to 7

    line_labels : array_like, strings, optional
        legend labels for line_data

    line_colors : array_like, strings, optional
        line colors to use for line_data
        (see matplotlib documentation)

    line_lws : array_like, floats, optional
        width of lines of line_data

    line_ls : array_like, strings, optional
        line styles to use for line_data
        (see matplotlib documentation)

    line_alphas : array_like, floats, optional
        alpha levels to use for lines of line_data

    line_markers : array_like, strings, optional
        line markers to use for line_data
        (see matplotlib documentation)

    Returns
    ---
    matplotlib axis object
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if line_data is None:
        dataframes = [bar_data]
    elif isinstance(line_data, list):
        dataframes = [bar_data] + line_data
    else:
        dataframes = [bar_data] + [line_data]

    n_items = len(
        [col for col in bar_data.columns if col.startswith('item_value_')])

    if line_lws is None:
        line_lws = [1 for i in range(len(dataframes) - 1)]

    if line_colors is None:
        line_colors = ['C{}'.format(i) for i in range(len(dataframes) - 1)]

    if line_ls is None:
        line_ls = ['-' for i in range(len(dataframes) - 1)]

    if line_markers is None:
        line_markers = ['o' for i in range(len(dataframes) - 1)]

    if line_alphas is None:
        line_alphas = [0.75 for i in range(len(dataframes) - 1)]

    add_labels = False
    if (line_labels is not None):
        if len(line_labels) == len(line_data):
            add_labels = True
        else:
            raise ValueError(
                'Number of prediction labels does not match number of prediction datasets.'
            )

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        if i == 0:
            df, bins = add_value_minus_max_others(df,
                                                  bins=bins,
                                                  return_bins=True)
        else:
            df = add_value_minus_max_others(df, bins=bins)

        # create temporary dataframe
        subjects = df['subject'].values
        value_minus_mean_others = df[[
            'value_minus_max_others_{}'.format(ii) for ii in range(n_items)
        ]].values
        is_choice = np.zeros_like(value_minus_mean_others)
        is_choice[np.arange(is_choice.shape[0]), df['choice'].values.
                  astype(np.int)] = 1

        df_tmp = pd.DataFrame({
            'subject':
            np.repeat(subjects, n_items),
            'value_minus_max_others':
            value_minus_mean_others.ravel(),
            'is_choice':
            is_choice.ravel()
        })

        # Compute summary statistics
        subject_means = df_tmp.groupby(['subject', 'value_minus_max_others'
                                        ]).is_choice.mean()
        means = subject_means.groupby('value_minus_max_others').mean()
        sems = subject_means.groupby('value_minus_max_others').sem()

        x = means.index
        if xlims is None:
            xlims = [x.min(), x.max()]
        if x.min() < xlims[0]:
            xlims[0] = x.min()
        if x.max() > xlims[1]:
            xlims[1] = x.max()

        # subset
        means = means[x <= np.max(xlims)]
        sems = sems[x <= np.max(xlims)]
        x = x[x <= np.max(xlims)]

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            barwidth = (xlims[1] - xlims[0]) / x.size * 0.6
            ax.bar(x,
                   means,
                   linewidth=1,
                   edgecolor='k',
                   facecolor='w',
                   width=barwidth)
            ax.vlines(x, means - sems, means + sems, linewidth=1, color='k')
            xlims[0] -= (1.5 * barwidth)
            xlims[1] += (1.5 * barwidth)

        else:  # plot predictions
            if add_labels:
                ax.plot(x,
                        means,
                        marker=line_markers[i - 1],
                        markerfacecolor=line_colors[i - 1],
                        alpha=line_alphas[i - 1],
                        lw=line_lws[i - 1],
                        ls=line_ls[i - 1],
                        color=line_colors[i - 1])
            else:
                ax.plot(x,
                        means,
                        marker=line_markers[i - 1],
                        markerfacecolor=line_colors[i - 1],
                        alpha=line_alphas[i - 1],
                        lw=line_lws[i - 1],
                        ls=line_ls[i - 1],
                        color=line_colors[i - 1])

    ax.axhline(1 / n_items, linestyle='--', color='k', linewidth=1, alpha=0.75)

    ax.set_xlabel('Item value –\nmax. value others', fontsize=fontsize)
    ax.set_ylabel('P(choose item)', fontsize=fontsize)
    if ylims is None:
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize, frameon=False)
    sns.despine(ax=ax)

    return ax


def add_gaze_advantage(df, bins=7, return_bins=False):
    """
    Add gaze advantage (defined as the difference
    between an item's gaze and the maximum gaze
    of all other) to response data

    Input
    ---
    df : dataframe
        response data

    bins : int or array_like, optional
        defining the bins to use when computing
        the gaze difference,
        if an int is given, this many bins will be
        created,
        defaults to 7

    return_bins : bool, optional
        whether or not to return the bins

    Returns
    ---
    copy of df (and bins if return_bins=True)
    """

    # infer number of items
    gaze_cols = ([col for col in df.columns if col.startswith('gaze_')])
    n_items = len(gaze_cols)

    gaze = df[gaze_cols].values
    gaze_advantage = np.zeros_like(gaze)
    for t in np.arange(gaze.shape[0]):
        for i in range(n_items):
            gaze_advantage[t, i] = gaze[t, i] - \
                np.max(gaze[t, np.arange(n_items) != i])

    if isinstance(bins, (int, float)):
        bins = np.round(np.linspace(-1, 1, bins), 2)

    for i in range(n_items):
        df['gaze_advantage_{}'.format(i)] = gaze_advantage[:, i]
        df['gaze_advantage_binned_{}'.format(i)] = pd.cut(
            df['gaze_advantage_{}'.format(i)],
            bins=bins,
            include_lowest=True,
            labels=bins[:-1])
    if not return_bins:
        return df.copy()
    else:
        return df.copy(), bins


def plot_pchoose_by_gaze_minus_max_others(bar_data,
                                          line_data=None,
                                          bins=7,
                                          ax=None,
                                          xlims=None,
                                          ylims=None,
                                          xlabel_skip=2,
                                          fontsize=7,
                                          line_labels=None,
                                          line_colors=None,
                                          line_lws=None,
                                          line_ls=None,
                                          line_alphas=None,
                                          line_markers=None):
    """
    Plot: P(choose best) ~ (item gaze - max. gaze othres)

    Input
    ---
    bar_data : dataframe
        response data to plot as bars

    line_data : list of dataframes, optional
        response data to plot as colored lines

    bins : int or array_like, optional
        x-bins to use for plotting,
        if an int is given, this many 
        bins will be created,
        defaults to 7

    ax : matplotlib axis, optional
        matplotlib axis object to use for plottig,
        if None, new axis is created,
        defaults to None

    xlims : tuple, optional
        limits for x-axis

    ylims : tuple, optional
        limits for y-axis

    x_label_skip : int, optional
        how many x-ticklabels to skip,
        defaults to 2 (indicating that every
        second x-ticklabel is plotted)

    fontsize : int, optional
        fontsize for plotting,
        defaults to 7

    line_labels : array_like, strings, optional
        legend labels for line_data

    line_colors : array_like, strings, optional
        line colors to use for line_data
        (see matplotlib documentation)

    line_lws : array_like, floats, optional
        width of lines of line_data

    line_ls : array_like, strings, optional
        line styles to use for line_data
        (see matplotlib documentation)

    line_alphas : array_like, floats, optional
        alpha levels to use for lines of line_data

    line_markers : array_like, strings, optional
        line markers to use for line_data
        (see matplotlib documentation)

    Returns
    ---
    matplotlib axis object
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if line_data is None:
        dataframes = [bar_data]
    elif isinstance(line_data, list):
        dataframes = [bar_data] + line_data
    else:
        dataframes = [bar_data] + [line_data]

    n_items = len(
        [col for col in bar_data.columns if col.startswith('item_value_')])

    if line_lws is None:
        line_lws = [1 for i in range(len(dataframes) - 1)]

    if line_colors is None:
        line_colors = ['C{}'.format(i) for i in range(len(dataframes) - 1)]

    if line_ls is None:
        line_ls = ['-' for i in range(len(dataframes) - 1)]

    if line_markers is None:
        line_markers = ['o' for i in range(len(dataframes) - 1)]

    if line_alphas is None:
        line_alphas = [0.75 for i in range(len(dataframes) - 1)]

    add_labels = False
    if (line_labels is not None):
        if len(line_labels) == len(line_data):
            add_labels = True
        else:
            raise ValueError(
                'Number of prediction labels does not match number of prediction datasets.'
            )

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        if i == 0:
            df, bins = add_gaze_advantage(df, bins=bins, return_bins=True)
        else:
            df = add_gaze_advantage(df, bins=bins)

        # create temporary dataframe
        subjects = df['subject'].values
        gaze_minus_mean_others = df[[
            'gaze_advantage_binned_{}'.format(ii) for ii in range(n_items)
        ]].values.astype(np.float)

        is_choice = np.zeros_like(gaze_minus_mean_others)
        is_choice[np.arange(is_choice.shape[0]), df['choice'].values.
                  astype(np.int)] = 1

        df_tmp = pd.DataFrame({
            'subject': np.repeat(subjects, n_items),
            'gaze_advantage': gaze_minus_mean_others.ravel(),
            'is_choice': is_choice.ravel()
        })

        # Compute summary statistics
        subject_means = df_tmp.groupby(['subject',
                                        'gaze_advantage']).is_choice.mean()
        means = subject_means.groupby('gaze_advantage').mean()
        sems = subject_means.groupby('gaze_advantage').sem()

        x = means.index
        if xlims is None:
            xlims = [x.min(), x.max()]
        if x.min() < xlims[0]:
            xlims[0] = x.min()
        if x.max() > xlims[1]:
            xlims[1] = x.max()

        # # subset
        # means = means[x <= np.max(xlims)]
        # sems = sems[x <= np.max(xlims)]
        # x = x[x <= np.max(xlims)]

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            barwidth = (xlims[1] - xlims[0]) / x.size * 0.4
            ax.bar(x,
                   means,
                   linewidth=1,
                   edgecolor='k',
                   facecolor='w',
                   width=barwidth)
            ax.vlines(x, means - sems, means + sems, linewidth=1, color='k')
            xlims[0] -= (1.5 * barwidth)
            xlims[1] += (1.5 * barwidth)

        else:  # plot predictions
            if add_labels:
                ax.plot(x,
                        means,
                        marker=line_markers[i - 1],
                        markerfacecolor=line_colors[i - 1],
                        alpha=line_alphas[i - 1],
                        lw=line_lws[i - 1],
                        ls=line_ls[i - 1],
                        color=line_colors[i - 1])
            else:
                ax.plot(x,
                        means,
                        marker=line_markers[i - 1],
                        markerfacecolor=line_colors[i - 1],
                        alpha=line_alphas[i - 1],
                        lw=line_lws[i - 1],
                        ls=line_ls[i - 1],
                        color=line_colors[i - 1])

    ax.axhline(1 / n_items, linestyle='--', color='k', linewidth=1, alpha=0.75)

    ax.set_xlabel('Item gaze –\nmax. gaze others', fontsize=fontsize)
    ax.set_ylabel('P(choose item)', fontsize=fontsize)
    if ylims is None:
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylim(ylims)
    if xlims is not None:
        ax.set_xlim(xlims)
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize, frameon=False)
    sns.despine(ax=ax)

    return ax


def compute_corrected_choice(df):
    """
    Compute and add corrected choice probability
    to response data; (see manuscript for details)

    Input
    ---
    df : dataframe
        response data

    Returns
    ---
    Copy of df, including corrected_choice column
    """

    # recode choice
    n_items = len([
        c for c in df.columns
        if c.startswith('gaze_') and not ('advantage' in c)
    ])
    is_choice = np.zeros((df.shape[0], n_items))
    is_choice[np.arange(is_choice.shape[0]), df['choice'].values.astype(np.int
                                                                        )] = 1

    if n_items > 2:
        values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
        value_range_others = np.zeros_like(is_choice)
        for t in range(value_range_others.shape[0]):
            for i in range(n_items):
                value_range_others[
                    t, i] = values[t, np.arange(n_items) != i].max(
                ) - values[t, np.arange(n_items) != i].min()

    # relative value
    df = add_value_minus_mean_others(df)
    relative_values = df[[
        'value_minus_mean_others_{}'.format(i) for i in range(n_items)
    ]].values

    df_tmp = pd.DataFrame({
        "subject": np.repeat(df['subject'].values, n_items),
        "relative_value": relative_values.ravel(),
        "is_choice": is_choice.ravel()
    })
    if n_items > 2:
        df_tmp['value_range_others'] = value_range_others.ravel()

    # place in dataframe
    data_out = []
    for s, subject in enumerate(df['subject'].unique()):
        subject_data_tmp = df_tmp[df_tmp['subject'] == subject].copy()
        if n_items > 2:
            X = subject_data_tmp[['relative_value', 'value_range_others']]
            X = sm.add_constant(X)
            y = subject_data_tmp['is_choice']
        else:
            X = subject_data_tmp[['relative_value']]
            # exclude every second entry, bc 2-item case is symmetrical
            X = sm.add_constant(X)[::2]
            y = subject_data_tmp['is_choice'].values[::2]

        logit = sm.Logit(y, X)
        result = logit.fit(disp=0)
        predicted_pchoice = result.predict(X)

        subject_data_tmp['corrected_choice'] = (subject_data_tmp['is_choice'] -
                                                predicted_pchoice)
        data_out.append(subject_data_tmp)

    data_out = pd.concat(data_out)

    return data_out.copy()


def plot_corp_by_gaze_advantage(bar_data,
                                line_data=None,
                                ax=None,
                                bins=7,
                                xlims=None,
                                ylims=None,
                                xlabel_skip=2,
                                fontsize=7,
                                line_labels=None,
                                line_colors=None,
                                line_lws=None,
                                line_ls=None,
                                line_alphas=None,
                                line_markers=None):
    """
    Plot: Corrected p(choose best) ~ (item gaze - max. gaze othres)

    Input
    ---
    bar_data : dataframe
        response data to plot as bars

    line_data : list of dataframes, optional
        response data to plot as colored lines

    ax : matplotlib axis, optional
        matplotlib axis object to use for plottig,
        if None, new axis is created,
        defaults to None

    bins : int or array_like, optional
        x-bins to use for plotting,
        if an int is given, this many 
        bins will be created,
        defaults to 7

    xlims : tuple, optional
        limits for x-axis

    ylims : tuple, optional
        limits for y-axis

    x_label_skip : int, optional
        how many x-ticklabels to skip,
        defaults to 2 (indicating that every
        second x-ticklabel is plotted)

    fontsize : int, optional
        fontsize for plotting,
        defaults to 7

    line_labels : array_like, strings, optional
        legend labels for line_data

    line_colors : array_like, strings, optional
        line colors to use for line_data
        (see matplotlib documentation)

    line_lws : array_like, floats, optional
        width of lines of line_data

    line_ls : array_like, strings, optional
        line styles to use for line_data
        (see matplotlib documentation)

    line_alphas : array_like, floats, optional
        alpha levels to use for lines of line_data

    line_markers : array_like, strings, optional
        line markers to use for line_data
        (see matplotlib documentation)

    Returns
    ---
    matplotlib axis object
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if line_data is None:
        dataframes = [bar_data]
    elif isinstance(line_data, list):
        dataframes = [bar_data] + line_data
    else:
        dataframes = [bar_data] + [line_data]

    if line_lws is None:
        line_lws = [1 for i in range(len(dataframes) - 1)]

    if line_colors is None:
        line_colors = ['C{}'.format(i) for i in range(len(dataframes) - 1)]

    if line_ls is None:
        line_ls = ['-' for i in range(len(dataframes) - 1)]

    if line_markers is None:
        line_markers = ['o' for i in range(len(dataframes) - 1)]

    if line_alphas is None:
        line_alphas = [0.75 for i in range(len(dataframes) - 1)]

    add_labels = False
    if (line_labels is not None):
        if len(line_labels) == len(line_data):
            add_labels = True
        else:
            raise ValueError(
                'Number of prediction labels does not match number of prediction datasets.'
            )

    n_items = len([c for c in bar_data.columns if c.startswith('gaze_')])
    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        # gaze advantage
        if i == 0:
            df, bins = add_gaze_advantage(df, bins=bins, return_bins=True)
        else:
            df = add_gaze_advantage(df, bins=bins)
        gaze_advantages = df[[
            'gaze_advantage_binned_{}'.format(i) for i in range(n_items)
        ]].values
        # corrected choice
        corrected_choice_data = compute_corrected_choice(df)
        corrected_choice_data['gaze_advantage_binned'] = gaze_advantages.ravel(
        )

        # Compute summary statistics
        subject_means = corrected_choice_data.groupby(
            ['subject', 'gaze_advantage_binned']).corrected_choice.mean()
        means = subject_means.groupby('gaze_advantage_binned').mean()
        sems = subject_means.groupby('gaze_advantage_binned').sem()
        x = means.index.values
        if xlims is None:
            xlims = [x.min(), x.max()]
        if x.min() < xlims[0]:
            xlims[0] = x.min()
        if x.max() > xlims[1]:
            xlims[1] = x.max()

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            barwidth = (xlims[1] - xlims[0]) / x.size * 0.4
            ax.bar(x,
                   means,
                   linewidth=1,
                   edgecolor='k',
                   facecolor='w',
                   width=barwidth)
            ax.vlines(x, means - sems, means + sems, linewidth=1, color='k')
            xlims[0] -= (1.5 * barwidth)
            xlims[1] += (1.5 * barwidth)

        else:  # plot predictions
            if add_labels:
                ax.plot(x,
                        means,
                        marker=line_markers[i - 1],
                        markerfacecolor=line_colors[i - 1],
                        alpha=line_alphas[i - 1],
                        lw=line_lws[i - 1],
                        ls=line_ls[i - 1],
                        color=line_colors[i - 1])
            else:
                ax.plot(x,
                        means,
                        marker=line_markers[i - 1],
                        markerfacecolor=line_colors[i - 1],
                        alpha=line_alphas[i - 1],
                        lw=line_lws[i - 1],
                        ls=line_ls[i - 1],
                        color=line_colors[i - 1])

    ax.set_xlabel('Item gaze –\nmax. gaze others', fontsize=fontsize)
    ax.set_ylabel('Corrected\nP(choose item)', fontsize=fontsize)
    ax.set_xticks([-1, -.5, 0, .5, 1.])
    ax.set_xticklabels([-1, -.5, 0, .5, 1.], fontsize=fontsize)
    if ylims is None:
        ax.set_ylim(-1.05, 1.05)
    else:
        ax.set_ylim(ylims)
    if xlims is not None:
        ax.set_xlim(xlims)
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize, frameon=False)
    sns.despine(ax=ax)

    return ax


def extract_range(x, extra=0.25, bound=(None, None)):
    """
    Extract range of x-data

    Input 
    ---
    x : array_like
        x-data

    extra : float, optional
        should be between [0,1],
        defining percentage of x-mean to add / subtract
        to min / max, when copmuting bounds
        e.g. upper bound = np.max(x) + extra * np.mean(x)

    bound : tuple, optional
        if given, these bounds are used

    Returns
    ---
    tuple of bounds
    """

    if bound[0] != None:
        xmin = bound[0]
    else:
        xmean = np.mean(x)
        xmin = np.min(x) - extra * xmean

    if bound[1] != None:
        xmax = bound[1]
    else:
        xmean = np.mean(x)
        xmax = np.max(x) + extra * xmean

    return [xmin, xmax]


def plot_correlation(x,
                     y,
                     xlabel='',
                     ylabel='',
                     title='',
                     ci=0.95,
                     alpha=0.5,
                     size=30,
                     color='red',
                     markercolor='black',
                     marker='o',
                     xticks=None,
                     yticks=None,
                     xticklabels=None,
                     yticklabels=None,
                     xlim=None,
                     ylim=None,
                     annotate=True,
                     annotation_pos=(0.1, 0.1),
                     annotation_halign='left',
                     fontsize_title=7,
                     fontsize_axeslabel=7,
                     fontsize_ticklabels=7,
                     fontsize_annotation=7,
                     regression=True,
                     plot_diagonal=False,
                     return_correlation=False,
                     ax=None):
    """
    Plot correlation between x and y; 
    (scatter-plot and regression line)
    """

    # Defaults
    if ax is None:
        fig, ax = plt.subplots()

    # Axes, ticks, ...
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=fontsize_ticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=fontsize_ticklabels)

    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticklabels)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Scatter (translucent dots with solid outlines)
    ax.scatter(x,
               y,
               marker='o',
               color='none',
               edgecolor=markercolor,
               linewidth=0.5,
               s=size)
    ax.scatter(x,
               y,
               marker='o',
               color=markercolor,
               alpha=alpha,
               linewidth=0,
               s=size)

    if regression:
        # LM fit
        X = sm.add_constant(x)
        lm = sm.OLS(y, X).fit()
        intercept, slope = lm.params
        table, data, columns = summary_table(lm, alpha=1. - ci)
        predicted, mean_ci_lower, mean_ci_upper = data[:,
                                                       np.array([2, 4, 5])].T

        xs = np.linspace(*ax.get_xlim(), 100)
        line = ax.plot(xs, intercept + slope * xs, color=color)
        sort_idx = np.argsort(x)
        ax.fill_between(x[sort_idx],
                        mean_ci_lower[sort_idx],
                        mean_ci_upper[sort_idx],
                        color=color,
                        alpha=0.1)

        # Annotation
        tval = lm.tvalues[-1]
        pval = lm.pvalues[-1]
        if pval < 0.0001:
            p_string = r'$P < 0.0001$'
        else:
            p_string = r'$P = {}$'.format(np.round(pval, 4))
        r = np.sign(tval) * np.sqrt(lm.rsquared)
        annotation = (r'$r = {:.2f}$, '.format(r)) + p_string
        if annotate:
            ax.text(*annotation_pos,
                    annotation,
                    verticalalignment='bottom',
                    horizontalalignment=annotation_halign,
                    transform=ax.transAxes,
                    fontsize=fontsize_annotation)

    # Diagonal
    if plot_diagonal:
        ax.plot([0, 1], [0, 1],
                transform=ax.transAxes,
                color='black',
                alpha=0.5,
                zorder=-10,
                lw=1)

    # Labels
    ax.set_xlabel(xlabel, fontsize=fontsize_axeslabel)
    ax.set_ylabel(ylabel, fontsize=fontsize_axeslabel)
    ax.set_title(title, fontsize=fontsize_title)

    sns.despine(ax=ax)

    if return_correlation:
        return ax, line, annotation
    else:
        return ax


def add_regression_line(ax, intercept, slope, color='darkgray', **kwargs):
    """
    Add a regression line to an axis,
    given its intercept and slope
    """

    xs = np.linspace(*ax.get_xlim(), 100)

    ax.plot(xs, intercept + slope * xs, color=color, **kwargs)

    return ax


def plot_posterior(samples,
                   kind='hist',
                   ref_val=None,
                   precision=2,
                   alpha=0.05,
                   bins=20,
                   burn=0,
                   ax=None,
                   fontsize=7,
                   color='skyblue'):
    """
    Arviz is broken, so we do it ourselves.

    Input:
        samples (TYPE): Description
        kind (str, optional): Description
        ref_val (None, optional): Description
        precision (int, optional): Description
        alpha (float, optional): Description
        burn (int, optional): Description
        ax (None, optional): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    if ax is None:
        fig, ax = plt.subplots()

    if kind == 'hist':
        ax.hist(samples, color=color, bins=bins)
    elif kind == 'kde':
        sns.kdeplot(samples, color=color, ax=ax)
    else:
        raise ValueError("'kind' should be 'hist' or 'kde'.")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Central tendency
    ax.text(x=np.mean(xlim),
            y=(ylim[1] - ylim[0]) * 0.95,
            s='mean = {}'.format(np.round(samples.mean(), precision)),
            ha='center',
            va='center',
            fontsize=fontsize)

    # HPD
    hpdvals = hpd(samples, alpha=alpha)
    ax.fill_between(hpdvals, y1=[0, 0], y2=2 * [(ylim[1] - ylim[0]) * 0.1],
                    color='black', edgecolor='none', lw=0,
                    alpha=0.5, zorder=2)
    # ax.text(x=np.mean(hpdvals),
    #         y=(ylim[1] - ylim[0]) * 0.1,
    #         s='{:.0f}% HPD'.format(100 * (1 - alpha)),
    #         ha='center',
    #         va='center',
    #         fontweight='bold',
    #         fontsize=fontsize)
    for val in hpdvals:
        ax.text(x=val,
                y=(ylim[1] - ylim[0]) * 0.2,
                s='{}'.format(np.round(val, precision)),
                ha='center',
                va='center',
                fontsize=fontsize)

    # Reference Value
    if ref_val is not None:
        ax.axvline(ref_val, color='crimson', linewidth=1, alpha=0.5)
        less = 100 * np.mean(samples < ref_val)
        more = 100 * np.mean(samples > ref_val)
        ax.text(x=np.mean(xlim),
                y=(ylim[1] - ylim[0]) * 0.5,
                s='{:.2f}% < {} < {:.2f}%'.format(less, ref_val, more),
                ha='center',
                va='center',
                fontweight='bold',
                color='crimson',
                fontsize=fontsize)

    ax.set_xlabel('Sample value', fontsize=fontsize)
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    sns.despine(ax=ax, left=True, top=True, right=True)

    return ax


def compare_parameters_hierarchical(model,
                           parameters=['v', 'gamma', 's', 'tau'],
                           comparisons=None,
                           xlimits=dict(v=dict(dist=(0, 1),
                                               delta=(-0.5, 0.5)),
                                        gamma=dict(dist=(-1, 1),
                                                   delta=(-1.5, 1.5)),
                                        s=dict(dist=(0, 0.5),
                                               delta=(-0.1, 0.1)),
                                        tau=dict(dist=(0, 3),
                                                 delta=(-1, 1))),
                           fontsize=7):
    """
    Plot group nodes and comparisons from hierarchical model.

    Input
    ---
    model : glambox.GLAM)
        GLAM model of type 'hierarchical'
    
    parameters : list
        list of parameter names (e.g., ['v', 'gamma'])
    
    comparisons : list, optional
        list of condition pairs (e.g., [('A', 'B')])

    Returns
    ---
        matplotlib fig and {axs}
    """
    parameter_names = {
        'v': 'v',
        'gamma': r'$\gamma$',
        's': r'$\sigma$',
        'tau': r'$\tau$'
    }
    parameter_precisions = {'v': 2, 'gamma': 2, 's': 2, 'tau': 2}

    if comparisons is None:
        comparisons = []
    n_params = len(parameters)
    n_comps = len(comparisons)

    fig = plt.figure(figsize=cm2inch(4.5 * (n_comps + 1), 2.25 * 1.5 * n_params), dpi=330)

    axs = {}

    for p, parameter in enumerate(parameters):

        # Distributions
        axs[(p, 0)] = plt.subplot2grid((n_params, n_comps + 2), (p, 0),
                                       rowspan=1,
                                       colspan=2)

        if model.design[parameter]['dependence'] is not None:
            for condition in model.design[parameter]['conditions']:
                axs[(p, 0)].hist(model.trace[0].get_values(parameter + '_' +
                                                           condition + '_mu'),
                                 label=condition,
                                 bins=np.linspace(*xlimits[parameter]['dist'], 31),
                                 alpha=0.5)
        else:
            axs[(p, 0)].hist(model.trace[0].get_values(parameter + '_mu'),
                             bins=np.linspace(*xlimits[parameter]['dist'], 31),
                             alpha=0.5)
        sns.despine(ax=axs[(p, 0)], top=True, right=True)

        # Labels & Legends
        if model.design[parameter]['dependence'] is not None:
            axs[(p, 0)].legend(frameon=False, fontsize=fontsize)
        axs[(p, 0)].set_title(parameter_names[parameter] + r'$_\mu$', fontsize=fontsize)
        axs[(p, 0)].set_ylabel('Frequency', fontsize=fontsize)
        axs[(p, 0)].set_yticks([])
        axs[(p, 0)].set_xlabel('Sample value', fontsize=fontsize)

        # Comparisons
        for c, comparison in enumerate(comparisons):
            axs[(p, c + 1)] = plt.subplot2grid((n_params, n_comps + 2),
                                               (p, c + 2),
                                               rowspan=1,
                                               colspan=1)
            # Check if parameter has dependence
            if model.design[parameter]['dependence'] is not None:
                # Then, if both conditions are present, plot posterior of the difference
                c0_present = (
                    comparison[0] in model.design[parameter]['conditions'])
                c1_present = (
                    comparison[1] in model.design[parameter]['conditions'])
                if c0_present & c1_present:
                    difference = (
                        model.trace[0].get_values(parameter + '_' +
                                                  comparison[0] + '_mu') -
                        model.trace[0].get_values(parameter + '_' +
                                                  comparison[1] + '_mu'))
                    axs[(p, c + 1)] = plot_posterior(
                        difference,
                        ax=axs[p, c + 1],
                        ref_val=0,
                        color='silver',
                        bins=np.linspace(*xlimits[parameter]['delta'], 16),
                        precision=parameter_precisions[parameter])
                    axs[(p, c + 1)].set_title(comparison[0] + ' - ' +
                                              comparison[1],
                                              fontsize=fontsize)
                    axs[(p, c + 1)].set_xlim(xlimits[parameter]['delta'])
                else:
                    # Otherwise, state that at least one condition is not present.
                    axs[p, c + 1].text(
                        0.5,
                        0.5,
                        ' '.join(['Condition(s) not present:\n'] + [
                            c for c in comparison
                            if c not in model.design[parameter]['conditions']
                        ]),
                        bbox=dict(boxstyle='square',
                                  ec='black',
                                  fc='lightgray',
                                  alpha=0.5),
                        ha='center',
                        va='center',
                        fontsize=fontsize)
                    axs[p, c + 1].axis('off')
            else:
                # Or that the parameter has no dependencies.
                axs[(p, c + 1)].text(0.5,
                                     0.5,
                                     'Parameter has no dependencies.',
                                     bbox=dict(boxstyle='square',
                                               ec='black',
                                               fc='lightgray',
                                               alpha=0.5),
                                     ha='center',
                                     va='center',
                                     fontsize=fontsize)
                axs[p, c + 1].axis('off')
            axs[(p, c + 1)].set_xlabel(
                r'$\delta$' + parameter_names[parameter] + r'$_{\mu}$',
                fontsize=fontsize)

    # Panel Labels
    from string import ascii_uppercase
    for label, ax in zip(list(ascii_uppercase),
                         [axs[(p, 0)] for p in range(n_params)]):
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.text(-0.1,
                1.2,
                label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                va='top')

    # Temporary workaround for overlapping xticklabels
    for label in axs[(0, 0)].get_xticklabels()[::2]:
        label.set_visible(False)

    fig.tight_layout()

    return fig, axs


def compare_parameters_individual(model,
                         parameters,
                         comparisons=None,
                         xlimits=None,
                         fontsize=7):
    """Summary

    Args:
        model (TYPE): Description
        parameters (TYPE): Description
        comparisons (None, optional): Description

    Returns:
        TYPE: Description
    """
    parameter_names = {'v': 'v',
                       'gamma': r'$\gamma$',
                       's': r'$\sigma$',
                       'tau': r'$\tau$'}

    # set default and update xlimits
    if xlimits is None:
        xlimits = {}
    xlimits_default = dict(v=dict(dist=(0, 1),
                                             delta=(-0.5, 0.5)),
                                      gamma=dict(dist=(-1, 1),
                                                 delta=(-1.5, 1.5)),
                                      s=dict(dist=(0, 0.5),
                                             delta=(-0.1, 0.1)),
                                      tau=dict(dist=(0, 3),
                                               delta=(-1, 1)))
    xlimits_default.update(xlimits)
    xlimits = xlimits_default

    if comparisons is None:
        comparisons = []
    n_params = len(parameters)
    n_comps = len(comparisons)

    subjects = model.data['subject'].unique().astype(int)
    summaries = [summary(trace) for trace in model.trace]

    fig = plt.figure(figsize=cm2inch(
        4.5 * (n_comps + 1), 2.25 * 1.5 * n_params), dpi=330)

    axs = {}

    for p, parameter in enumerate(parameters):

        # Distributions
        axs[(p, 0)] = plt.subplot2grid(
            (n_params, n_comps + 2), (p, 0), rowspan=1, colspan=2)

        if model.design[parameter]['dependence'] is not None:
            for c, condition in enumerate(model.design[parameter]['conditions']):
                means = [summaries[i].loc[parameter + '_' + condition + '__0_0', 'mean']
                         for i in subjects]
                hpdlower = [summaries[i].loc[parameter + '_' + condition + '__0_0', 'hpd_2.5']
                            for i in subjects]
                hpdupper = [summaries[i].loc[parameter + '_' + condition + '__0_0', 'hpd_97.5']
                            for i in subjects]
                axs[(p, 0)].scatter(means,
                                    subjects,
                                    #'o',
                                    s=fontsize,
                                    color='C{}'.format(c),
                                    label=condition)
                axs[(p, 0)].hlines(y=subjects, xmin=hpdlower,
                                   xmax=hpdupper, alpha=0.75, zorder=-2)
        else:
            means = [summaries[i].loc[parameter + '__0_0', 'mean']
                     for i in subjects]
            hpdlower = [summaries[i].loc[parameter + '__0_0', 'hpd_2.5']
                        for i in subjects]
            hpdupper = [summaries[i].loc[parameter + '__0_0', 'hpd_97.5']
                        for i in subjects]
            plt.scatter(means,
                        subjects,
                        #'o',
                        s=fontsize)
            axs[(p, 0)].hlines(y=subjects, xmin=hpdlower,
                               xmax=hpdupper, alpha=0.75, zorder=-2)

        sns.despine(ax=axs[(p, 0)], top=True, right=True)

        # Labels & Legends
        if model.design[parameter]['dependence'] is not None:
            axs[(p, 0)].legend(frameon=False,
                               fontsize=fontsize, handletextpad=-0.5, borderpad=-0.5)
        axs[(p, 0)].set_title(parameter_names[parameter], fontsize=fontsize)
        axs[(p, 0)].set_ylabel('Subject', fontsize=fontsize)
        axs[(p, 0)].set_yticks([])
        axs[(p, 0)].set_xlabel('Sample value', fontsize=fontsize)
        axs[(p, 0)].set_xlim(xlimits[parameter]['dist'])

        # Comparisons
        for c, comparison in enumerate(comparisons):
            axs[(p, c + 1)] = plt.subplot2grid((n_params, n_comps + 2),
                                               (p, c + 2), rowspan=1, colspan=1)
            # Check if parameter has dependence
            if model.design[parameter]['dependence'] is not None:
                # Then, if both conditions are present, plot posterior of the difference
                c0_present = (
                    comparison[0] in model.design[parameter]['conditions'])
                c1_present = (
                    comparison[1] in model.design[parameter]['conditions'])
                if c0_present & c1_present:
                    differences = np.array([(model.trace[i].get_values(parameter + '_' + comparison[0]) -
                                             model.trace[i].get_values(parameter + '_' + comparison[1]))
                                            for i in subjects])[:, :, 0, 0]

                    means = np.mean(differences, axis=1)
                    hpdlower, hpdupper = hpd(differences.T).T

                    zero_excluded = np.array([(low > 0) or (high < 0)
                                              for low, high in zip(hpdlower, hpdupper)]).astype(int)
                    colors = np.array(['red', 'green'])[zero_excluded]
                    axs[(p, c + 1)].scatter(means,
                                            subjects,
                                            color=colors,
                                            s=fontsize)
                    axs[(p, c + 1)].hlines(y=subjects,
                                           xmin=hpdlower,
                                           xmax=hpdupper,
                                           alpha=0.75,
                                           zorder=-2)
                    axs[(p, c + 1)].set_title(comparison[0] +
                                              ' - ' + comparison[1], fontsize=fontsize)
                    axs[(p, c + 1)].set_ylabel('Subject', fontsize=fontsize)
                    axs[(p, c + 1)].set_yticks([])

                    axs[(p, c + 1)].axvline(0, color='crimson',
                                            linewidth=1, alpha=0.5)
                    sns.despine(ax=axs[(p, c + 1)], top=True, right=True)
                    axs[(p, c + 1)].tick_params(axis='both', which='major',
                                                labelsize=fontsize)
                    axs[(p, c + 1)].set_xlim(xlimits[parameter]['delta'])
                else:
                    # Otherwise, state that at least one condition is not present.
                    axs[p, c + 1].text(0.5, 0.5, ' '.join(['Condition(s) not present:\n'] +
                                                          [c for c in comparison
                                                           if c not in model.design[parameter]['conditions']]),
                                       fontsize=fontsize,
                                       bbox=dict(boxstyle='square',
                                                 ec='black',
                                                 fc='lightgray',
                                                 alpha=0.5),
                                       ha='center', va='center')
                    axs[p, c + 1].axis('off')
            else:
                # Or that the parameter has no dependencies.
                axs[(p, c + 1)].text(0.5, 0.5, 'Parameter has no dependencies.',
                                     fontsize=fontsize,
                                     bbox=dict(boxstyle='square',
                                               ec='black',
                                               fc='lightgray',
                                               alpha=0.5),
                                     ha='center', va='center')
                axs[p, c + 1].axis('off')
            axs[(p, c + 1)].set_xlabel(r'$\delta$' +
                                       parameter_names[parameter], fontsize=fontsize)
            

    # Panel Labels
    from string import ascii_uppercase
    for label, ax in zip(list(ascii_uppercase), [axs[(p, 0)] for p in range(n_params)]):
        ax.text(-0.1, 1.2, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight='bold', va='top')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)   

    fig.tight_layout()

    return fig, axs
