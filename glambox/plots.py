#!/usr/bin/python

from .analysis import aggregate_subject_level_data
import numpy as np
import pandas as pd
from pymc3.stats import hpd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
import matplotlib.pyplot as plt
import seaborn as sns


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def plot_aggregate(bar_data,
                   line_data=None,
                   line_labels=None,
                   fontsize=7,
                   value_bins=7,
                   gaze_bins=7,
                   limits={
                   'p_choose_best': (0, 1),
                   'rt': (0, None),
                   'corrected_p_choose_best': (-1, 1)
                   }):
    fig, axs = plt.subplots(1, 4, figsize=cm2inch(18, 4.5), dpi=330)

    # add default limits
    for key, lim in zip(['p_choose_best', 'rt', 'corrected_p_choose_best'],
                        [(0,1), (0,None), (-1,1)]):
        if key not in limits.keys():
            limits[key] = lim

    axs[0] = plot_rt_by_difficulty(bar_data,
                                   line_data,
                                   xlabel_skip=2,
                                   ax=axs[0],
                                   line_labels=line_labels,
                                   bins=value_bins,
                                   fontsize=fontsize,
                                   ylims=limits['rt'])
    axs[1] = plot_pchoose_by_value_minus_mean_others(bar_data,
                                            line_data,
                                            xlabel_skip=4,
                                            xlabel_start=0,
                                            ax=axs[1],
                                            bins=value_bins,
                                            fontsize=fontsize,
                                            ylims=limits['p_choose_best'])
    axs[2] = plot_pchoose_by_gaze_minus_mean_others(bar_data,
                                                    line_data,
                                                    ax=axs[2],
                                                    bins=gaze_bins,
                                                    fontsize=fontsize,
                                                    ylims=limits['p_choose_best'])
    axs[3] = plot_corp_by_gaze_advantage(bar_data,
                                         line_data,
                                         ax=axs[3],
                                         bins=gaze_bins,
                                         fontsize=fontsize,
                                         ylims=limits['corrected_p_choose_best'])

    # Labels
    for label, ax in zip(list('ABCD'), axs.ravel()):
        ax.text(-0.5,
                1.05,
                label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                va='top')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        sns.despine(ax=ax)

    fig.tight_layout()

    return fig, axs


def add_difficulty(df, bins=7, return_bins=False):

    # infer number of items
    value_cols = ([col for col in df.columns if col.startswith('item_value_')])

    values = df[value_cols].values
    values_sorted = np.sort(values, axis=1)
    difficulty = values_sorted[:, -1] - np.mean(values_sorted[:, :-1], axis=1)

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
        ylims = np.mean(np.concatenate([a['rt'].ravel() for a in dataframes])) * 2
    ax.set_ylim(ylims)
    ax.set_xlabel('Max. value –\nmean value others', fontsize=fontsize)
    ax.set_ylabel('Response time (s)', fontsize=fontsize)
    ax.set_xlim(xlims)
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(means.index.values[::xlabel_skip])
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize * 0.7, frameon=False)
    sns.despine(ax=ax)

    return ax


def add_value_minus_mean_others(df, bins=7, return_bins=False):

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


def plot_pchoose_by_value_minus_mean_others(bar_data,
                                            line_data=None,
                                            ax=None,
                                            bins=7,
                                            xlims=None,
                                            ylims=None,
                                            xlabel_skip=2,
                                            xlabel_start=1,
                                            fontsize=7,
                                            line_labels=None,
                                            line_colors=None,
                                            line_lws=None,
                                            line_ls=None,
                                            line_alphas=None,
                                            line_markers=None):

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
            df, bins = add_value_minus_mean_others(df,
                                                   bins=bins,
                                                   return_bins=True)
        else:
            df = add_value_minus_mean_others(df, bins=bins)

        # create temporary dataframe
        subjects = df['subject'].values
        value_minus_mean_others = df[[
            'value_minus_mean_others_{}'.format(ii) for ii in range(n_items)
        ]].values
        is_choice = np.zeros_like(value_minus_mean_others)
        is_choice[np.arange(is_choice.shape[0]), df['choice'].values.
                  astype(np.int)] = 1

        df_tmp = pd.DataFrame({
            'subject':
            np.repeat(subjects, n_items),
            'value_minus_mean_others':
            value_minus_mean_others.ravel(),
            'is_choice':
            is_choice.ravel()
        })

        # Compute summary statistics
        subject_means = df_tmp.groupby(['subject', 'value_minus_mean_others'
                                        ]).is_choice.mean()
        means = subject_means.groupby('value_minus_mean_others').mean()
        sems = subject_means.groupby('value_minus_mean_others').sem()

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

    ax.set_xlabel('Item value –\nmean value others', fontsize=fontsize)
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

    # infer number of items
    gaze_cols = ([col for col in df.columns if col.startswith('gaze_')])
    n_items = len(gaze_cols)

    gaze = df[gaze_cols].values
    gaze_advantage = np.zeros_like(gaze)
    for t in np.arange(gaze.shape[0]):
        for i in range(n_items):
            gaze_advantage[t, i] = gaze[t, i] - \
                np.mean(gaze[t, np.arange(n_items) != i])

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


def plot_pchoose_by_gaze_minus_mean_others(bar_data,
                                           line_data=None,
                                           bins=7,
                                           ax=None,
                                           xlims=None,
                                           ylims=None,
                                           xlabel_skip=2,
                                           xlabel_start=1,
                                           fontsize=7,
                                           line_labels=None,
                                           line_colors=None,
                                           line_lws=None,
                                           line_ls=None,
                                           line_alphas=None,
                                           line_markers=None):

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

    ax.set_xlabel('Item gaze –\nmean gaze others', fontsize=fontsize)
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

    ax.set_xlabel('Item gaze –\nmean gaze others', fontsize=fontsize)
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


def plot_node(model,
              parameter,
              comparisons=None,
              fontsize=12,
              alpha=0.5,
              hpd_alpha=0.05,
              plot_histogram=True):

    # determine model type
    model_type = model.type

    # make sure comparisons specified correctly
    if comparisons is not None:
        if not getattr(comparisons, '__iter__', False):
            error_msg = 'comparisons must be iterable (e.g.[(condition(A), condition(B))]).'
            raise ValueError(error_msg)
        else:
            if not np.all([len(c) == 2 for c in comparisons]):
                error_msg = 'Each comparison must be of length 2 (e.g.[(condition(A), condition(B))]).'
                raise ValueError(error_msg)
            else:
                n_comparisons = len(comparisons)

    # extract design for parameter
    if parameter not in model.design.keys():
        error_msg = '"' "{}" '" not part of model parameters.'.format(
            parameter)
        raise ValueError(error_msg)
    parameter_design = model.design[parameter]
    conditions = parameter_design['conditions']
    if conditions is not None:
        n_conditions = len(conditions)
    else:
        n_conditions = 1

    # extract subjects
    subjects = parameter_design[conditions[0]]['subjects']
    try:
        subjects = subjects.astype(np.int)
    except:
        None

    # read out number of traces
    if not getattr(model.trace, '__iter__', False):
        model_traces = [model.trace]
    else:
        model_traces = model.trace
    n_traces = len(model_traces)
    if model_type == 'individual':
        if subjects.size != n_traces:
            error_msg = 'Number of subjects contained in model does not match number of traces.'
            raise ValueError(error_msg)

    # set up figure
    fig = plt.figure(figsize=(4 * (1 + n_comparisons), 2 * n_traces), dpi=330)

    # set up dict for figure axes
    axs = dict()
    for c in range(1 + n_comparisons):
        axs[c] = np.array([])

    # set up array to store [min, max] x-lims per figure column
    xlims = np.zeros((1 + n_comparisons, 2)) * np.nan

    # plot
    for r in range(n_traces):

        # extract trace
        trace = model_traces[r]

        # create & collect axis
        ax = plt.subplot2grid((n_traces, n_comparisons + 2), (r, 0), colspan=2)
        axs[0] = np.append(axs[0], ax)

        # add y-label
        if model_type == 'individual':
            axs[0][-1].set_ylabel('Subject {}'.format(subjects[r], parameter),
                                  fontsize=fontsize)

        # plot condition traces
        for ci, condition in enumerate(conditions):

            # extract trace
            if model_type == 'hierarchical':
                condition_trace = trace['{}_{}_mu'.format(
                    parameter, condition)].ravel()
            else:
                condition_trace = trace['{}_{}'.format(parameter,
                                                       condition)].ravel()

            # update x-lims
            if np.isnan(xlims[0, 0]):
                xlims[0, 0] = np.min(condition_trace)
                xlims[0, 1] = np.max(condition_trace)
            else:
                if np.min(condition_trace) < xlims[0, 0]:
                    xlims[0, 0] = np.min(condition_trace)
                if np.max(condition_trace) > xlims[0, 1]:
                    xlims[0, 1] = np.max(condition_trace)

            # plot trace
            if not plot_histogram:
                trace_hpd = hpd(condition_trace, alpha=hpd_alpha)
                trace_mean = np.mean(condition_trace)
                axs[0][-1].plot(trace_hpd, [ci, ci],
                                lw=3,
                                color='C{}'.format(ci))
                axs[0][-1].scatter(x=trace_mean,
                                   y=ci,
                                   color='C{}'.format(ci),
                                   s=100)
            else:
                if r == 0:
                    contition_label = condition
                else:
                    contition_label = ''
                axs[0][-1].hist(condition_trace,
                                histtype='stepfilled',
                                bins=50,
                                alpha=alpha,
                                label=contition_label)

        # set y-lim
        if not plot_histogram:
            axs[0][-1].set_ylim(-1, n_conditions)
            axs[0][-1].set_yticks(np.arange(n_conditions))
            axs[0][-1].set_yticklabels(conditions, fontsize=fontsize)

        # plot comparisons
        for c, comparison in enumerate(comparisons):

            # create & collect axis
            ax = plt.subplot2grid((n_traces, n_comparisons + 2), (r, 2 + c))
            axs[c + 1] = np.append(axs[c + 1], ax)

            # compute trace difference
            if model_type == 'hierarchical':
                trace_diff = (
                    trace['{}_{}_mu'.format(parameter,
                                            comparison[0])].ravel() -
                    trace['{}_{}_mu'.format(parameter, comparison[1])].ravel())
            else:
                trace_diff = (
                    trace['{}_{}'.format(parameter, comparison[0])].ravel() -
                    trace['{}_{}'.format(parameter, comparison[1])].ravel())

            # update x-lims
            if np.isnan(xlims[c + 1, 0]):
                xlims[c + 1, 0] = np.min(trace_diff)
                xlims[c + 1, 1] = np.max(trace_diff)
            else:
                if np.min(trace_diff) < xlims[c + 1, 0]:
                    xlims[c + 1, 0] = np.min(trace_diff)
                if np.max(trace_diff) > xlims[c + 1, 1]:
                    xlims[c + 1, 1] = np.max(trace_diff)

            # plot trace difference
            if not plot_histogram:
                trace_diff_hpd = hpd(trace_diff, alpha=hpd_alpha)
                trace_diff_mean = np.mean(trace_diff)
                axs[c + 1][-1].plot(trace_diff_hpd, [0, 0], lw=3, color='gray')
                axs[c + 1][-1].scatter(x=trace_diff_mean,
                                       y=0,
                                       color='gray',
                                       s=100)
                hpd_string = '95% HPD:\n[{}, {}]'.format(
                    np.round(trace_diff_hpd[0], 2),
                    np.round(trace_diff_hpd[1], 2))
                axs[c + 1][-1].text(0.5,
                                    0.7,
                                    hpd_string,
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    transform=axs[c + 1][-1].transAxes,
                                    fontsize=fontsize)
                axs[c + 1][-1].set_ylim(-1, n_conditions)
            else:
                plot_posterior(trace_diff,
                               color='gray',
                               histtype='stepfilled',
                               alpha=alpha,
                               alpha_level=hpd_alpha,
                               bins=50,
                               ref_val=0,
                               ax=axs[c + 1][-1])

            # set title
            if r == 0:
                axs[c + 1][-1].set_title('{} - {}'.format(*comparison),
                                         fontsize=fontsize)

    # label x-axis
    if parameter in ['sigma', 'gamma', 'tau']:
        for i in range(1 + n_comparisons):
            if model_type == 'hierarchical':
                axs[i][-1].set_xlabel(r'$\delta$' +
                                      r'$\{}$'.format(parameter) + r'$_{mu}$',
                                      fontsize=fontsize * 1.2)
            else:
                axs[i][-1].set_xlabel(r'$\delta$' + r'$\{}$'.format(parameter),
                                      fontsize=fontsize * 1.2)
    else:
        for i in range(1 + n_comparisons):
            if model_type == 'hierarchical':
                axs[i][-1].set_xlabel(r'$\delta$' + r'${}$'.format(parameter) +
                                      r'$_{mu}$',
                                      fontsize=fontsize)
            else:
                axs[i][-1].set_xlabel(r'$\delta$' + r'${}$'.format(parameter),
                                      fontsize=fontsize)

    # set x-lims
    for i in range(1 + n_comparisons):
        for ax in axs[i]:
            ax.set_xlim(xlims[i, 0], xlims[i, 1])
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            if (model_type != 'individual') or (i > 0):
                ax.set_yticks([])
                ax.set_yticklabels([])
            sns.despine(ax=ax)

    # add legend
    axs[0][0].legend(loc='upper center',
                     frameon=False,
                     fontsize=fontsize,
                     ncol=len(conditions))

    # re-shape axs
    axs = np.concatenate([axs[i][None] for i in range(1 + n_comparisons)],
                         axis=1)

    # autmomatic cleaning
    fig.tight_layout()

    return fig, axs


def plot_individual_node_comparison(model,
                                    parameter,
                                    comparisons,
                                    fontsize=7,
                                    hpd_alpha=0.05):

    # determine model type
    model_type = model.type
    if model_type != 'individual':
        error_msg = 'plot_individual_node_comparison requires model of "' "individual" '" type.'
        raise

    # make sure comparisons specified correctly
    if comparisons is not None:
        if not getattr(comparisons, '__iter__', False):
            error_msg = 'comparisons must be iterable (e.g.[(condition(A), condition(B))]).'
            raise ValueError(error_msg)
        else:
            if not np.all([len(c) == 2 for c in comparisons]):
                error_msg = 'Each comparison must be of length 2 (e.g.[(condition(A), condition(B))]).'
                raise ValueError(error_msg)
            else:
                n_comparisons = len(comparisons)

    # extract design for parameter
    if parameter not in model.design.keys():
        error_msg = '"' "{}" '" not part of model parameters.'.format(
            parameter)
        raise ValueError(error_msg)
    parameter_design = model.design[parameter]
    conditions = parameter_design['conditions']

    # extract subjects
    subjects = parameter_design[conditions[0]]['subjects']
    try:
        subjects = subjects.astype(np.int)
    except:
        None
    n_subjects = subjects.size

    # read out number of traces
    if not getattr(model.trace, '__iter__', False):
        model_traces = [model.trace]
    else:
        model_traces = model.trace
    n_traces = len(model_traces)
    if n_subjects != n_traces:
        error_msg = 'Number of subjects contained in model does not match number of traces.'
        raise ValueError(error_msg)

    # set up figure
    fig, axs = plt.subplots(1,
                            n_comparisons,
                            figsize=cm2inch(18,9*n_comparisons),
                            dpi=300,
                            sharey=True,
                            sharex=True)

    # plot
    for r in range(n_subjects):

        # extract trace
        trace = model_traces[r]

        # plot comparisons
        for c, comparison in enumerate(comparisons):

            if n_comparisons > 1:
                ax = axs[c]
            else:
                ax = axs

            # compute trace difference
            trace_diff = (
                trace['{}_{}'.format(parameter, comparison[0])].ravel() -
                trace['{}_{}'.format(parameter, comparison[1])].ravel())

            # plot trace difference
            trace_diff_hpd = hpd(trace_diff, alpha=hpd_alpha)
            trace_diff_mean = np.mean(trace_diff)
            if (trace_diff_hpd[0] < 0) & (trace_diff_hpd[1] > 0):
                ax.plot(trace_diff_hpd, [r, r], lw=3, color='red')
                ax.scatter(x=trace_diff_mean, y=r, color='red', s=100)
            else:
                ax.plot(trace_diff_hpd, [r, r], lw=3, color='k')
                ax.scatter(x=trace_diff_mean, y=r, color='k', s=100)

            # set title
            ax.set_title('{} - {}'.format(*comparison), fontsize=fontsize)

            # add 0-line
            ax.axvline(0, color='red', ls='--', alpha=0.5)

            # set y-labels
            ax.set_ylim(-1, n_subjects)
            ax.set_yticks(np.arange(n_subjects))
            if c == 0:
                ax.set_yticklabels(subjects, fontsize=fontsize)
                ax.set_ylabel('Subject', fontsize=fontsize)

            # set x-label
            if parameter in ['sigma', 'gamma', 'tau']:
                ax.set_xlabel(r'$\{}$'.format(parameter),
                              fontsize=fontsize * 1.2)
            else:
                ax.set_xlabel(r'${}$'.format(parameter), fontsize=fontsize)

    # autmomatic cleaning
    fig.tight_layout()

    return fig, axs


def extract_range(x, extra=0.25, bound=(None, None)):

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


def plot_individual_differences(data,
                           nbins=20,
                           fontsize=7,
                           regression=True,
                           annotate=True,
                           figsize=cm2inch(18, 7),
                           limits={
                               'p_choose_best': (0, 1),
                               'rt': (0, None),
                               'gaze_influence': (None, None)
                           }):

    if (regression == False) & (annotate == True):
        print('annotate only possible, if regression = True.')

    fig = plt.figure(figsize=figsize, dpi=330)

    ax00 = plt.subplot2grid((7, 3), (0, 0), rowspan=2)
    ax10 = plt.subplot2grid((7, 3), (2, 0), rowspan=5)

    ax01 = plt.subplot2grid((7, 3), (0, 1), rowspan=2)
    ax11 = plt.subplot2grid((7, 3), (2, 1), rowspan=5)

    ax02 = plt.subplot2grid((7, 3), (0, 2), rowspan=2)
    ax12 = plt.subplot2grid((7, 3), (2, 2), rowspan=5)

    # add default limits
    for key, lim in zip(['p_choose_best', 'rt', 'gaze_influence'],
                        [(0,1), (0,None), (None,None)]):
        if key not in limits.keys():
            limits[key] = lim

    # create subject_summary
    n_items = np.int(len([c for c in data.columns if 'item_value_' in c]))
    subject_summary = aggregate_subject_level_data(data, n_items=n_items)

    # extract plotting ranges
    rt_range = extract_range(subject_summary['rt']['mean'], bound=limits['rt'])
    if (rt_range[1] - rt_range[0]) > 3:
        rt_tickstep = 1.5
    else:
        rt_tickstep = 0.75
    rt_ticks = np.arange(rt_range[0], rt_range[1] + rt_tickstep,
                         rt_tickstep)

    best_chosen_range = extract_range(subject_summary['best_chosen']['mean'],
                                      bound=limits['p_choose_best'])
    best_chosen_ticks = np.arange(0, 1.1, 0.2)

    gaze_influence_range = extract_range(subject_summary['gaze_influence'],
                                         bound=limits['gaze_influence'])
    gaze_influence_ticks = np.arange(-1, 1.1, 0.2)

    # Scatter plots
    plot_correlation(subject_summary['rt']['mean'],
                     subject_summary['best_chosen']['mean'],
                     marker='o',
                     markercolor='C0',
                     regression=regression,
                     annotate=annotate,
                     annotation_pos=(0.1, 0.01),
                     xlabel='Mean RT (s)',
                     ylabel='P(choose best)',
                     xlim=rt_range,
                     xticks=rt_ticks,
                     ylim=best_chosen_range,
                     yticks=best_chosen_ticks,
                     fontsize_title=fontsize,
                     fontsize_axeslabel=fontsize,
                     fontsize_ticklabels=fontsize,
                     fontsize_annotation=fontsize,
                     ax=ax10)

    plot_correlation(subject_summary['gaze_influence'],
                     subject_summary['rt']['mean'],
                     marker='o',
                     markercolor='C0',
                     regression=regression,
                     annotate=annotate,
                     annotation_pos=(0.1, 0.01),
                     ylabel='Mean RT (s)',
                     xlabel='Gaze influence\non P(choice | value)',
                     ylim=rt_range,
                     yticks=rt_ticks,
                     xlim=gaze_influence_range,
                     xticks=gaze_influence_ticks,
                     fontsize_title=fontsize,
                     fontsize_axeslabel=fontsize,
                     fontsize_ticklabels=fontsize,
                     fontsize_annotation=fontsize,
                     ax=ax11)

    plot_correlation(subject_summary['best_chosen']['mean'],
                     subject_summary['gaze_influence'],
                     marker='o',
                     markercolor='C0',
                     regression=regression,
                     annotate=annotate,
                     annotation_pos=(0.1, 0.01),
                     ylabel='Gaze influence\non P(choice | value)',
                     xlabel='P(choose best)',
                     ylim=gaze_influence_range,
                     yticks=gaze_influence_ticks,
                     xlim=best_chosen_range,
                     xticks=best_chosen_ticks,
                     fontsize_title=fontsize,
                     fontsize_axeslabel=fontsize,
                     fontsize_ticklabels=fontsize,
                     fontsize_annotation=fontsize,
                     ax=ax12)

    # Marginal histograms
    ax00.hist(subject_summary['rt']['mean'],
              bins=np.linspace(rt_range[0], rt_range[1], nbins + 1),
              color='C0')

    ax01.hist(subject_summary['gaze_influence'],
              bins=np.linspace(gaze_influence_range[0],
                               gaze_influence_range[1], nbins + 1),
              color='C0')

    ax02.hist(subject_summary['best_chosen']['mean'],
              bins=np.linspace(best_chosen_range[0], best_chosen_range[1],
                               nbins + 1),
              color='C0')

    hist_lim = np.max(
        [ax00.get_ylim()[1],
         ax01.get_ylim()[1],
         ax02.get_ylim()[1]]).astype(np.int) + 1

    # Labels
    for label, ax in zip(list('ABC'), [ax00, ax01, ax02]):
        ax.text(-0.45,
                1.1,
                label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                va='top')
    for label, ax in zip(list('DEF'), [ax10, ax11, ax12]):
        ax.text(-0.45,
                1.025,
                label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                va='top')

    # Fine-tune marginal histograms
    for ax in np.array([ax00, ax01, ax02]):
        ax.set_xticks([])
        ax.set_ylim([0, hist_lim])
        ax.set_yticks([0, hist_lim])
        ax.set_yticklabels([0, hist_lim], fontsize=fontsize)
        ax.set_ylabel('Frequency', fontsize=fontsize)

    ax00.set_xticks(rt_ticks)
    ax00.set_xticklabels([])
    ax01.set_xticks(gaze_influence_ticks)
    ax01.set_xticklabels([])
    ax02.set_xticks(best_chosen_ticks)
    ax02.set_xticklabels([])

    ax00.set_xlim(rt_range)
    ax01.set_xlim(gaze_influence_range)
    ax02.set_xlim(best_chosen_range)

    for ax in [ax00, ax01, ax02, ax10, ax11, ax12]:
        sns.despine(ax=ax)

    fig.tight_layout()

    return fig


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

    xs = np.linspace(*ax.get_xlim(), 100)

    ax.plot(xs, intercept + slope * xs, color=color, **kwargs)

    return ax


def plot_individual(observed,
                    predictions,
                    prediction_labels=None,
                    colors=None,
                    fontsize=7,
                    alpha=1.0,
                    figsize=None,
                    limits={
                               'p_choose_best': (0, 1),
                               'rt': (0, None),
                               'gaze_influence': (None, None)
                           }):

    # count number of predictions
    n_predictions = len(predictions)
    # define prediction labels
    if prediction_labels is None:
        prediction_labels = [
            'Prediction {}'.format(i + 1) for i in range(n_predictions)
        ]

    # define figre
    if figsize is None:
        figsize = cm2inch(18, 6 * n_predictions)
    fig, axs = plt.subplots(n_predictions, 3, figsize=figsize, dpi=330)
    if axs.ndim == 1:
        axs = axs.reshape([1, axs.size])

    # add default limits
    for key, lim in zip(['p_choose_best', 'rt', 'gaze_influence'],
                        [(0,1), (0,None), (None,None)]):
        if key not in limits.keys():
            limits[key] = lim

    # create subject summary for observed
    n_items = np.int(len([c for c in observed.columns if 'item_value_' in c]))
    for i, prediction in enumerate(predictions):
        n_items_pred = np.int(
            len([c for c in prediction.columns if 'item_value_' in c]))
        if n_items != n_items_pred:
            raise ValueError(
                'observed and prediction {} contain unequal number of items'.
                format(i))
    observed_subject_summary = aggregate_subject_level_data(observed,
                                                            n_items=n_items)

    # extract oberved value ranges
    rt_range = extract_range(observed_subject_summary['rt']['mean'],
                             bound=limits['rt'])
    best_chosen_range = extract_range(
        observed_subject_summary['best_chosen']['mean'], bound=limits['p_choose_best'])
    gaze_influence_range = extract_range(
        observed_subject_summary['gaze_influence'], bound=limits['gaze_influence'])

    # plot observed vs predicted
    for m, prediction in enumerate(predictions):

        # create subject summary for prediction
        prediction_subject_summary = aggregate_subject_level_data(
            prediction, n_items=n_items)

        # a) Mean RT
        axs[m, 0].scatter(observed_subject_summary['rt']['mean'],
                          prediction_subject_summary['rt']['mean'],
                          marker='o',
                          color='none',
                          edgecolor='C0',
                          linewidth=0.5,
                          s=30)
        axs[m, 0].scatter(observed_subject_summary['rt']['mean'],
                          prediction_subject_summary['rt']['mean'],
                          marker='o',
                          color='C0',
                          alpha=0.5,
                          linewidth=0,
                          s=30)

        # b) P(choose best)
        axs[m, 1].scatter(observed_subject_summary['best_chosen']['mean'],
                          prediction_subject_summary['best_chosen']['mean'],
                          marker='o',
                          color='none',
                          edgecolor='C0',
                          linewidth=0.5,
                          s=30)
        axs[m, 1].scatter(observed_subject_summary['best_chosen']['mean'],
                          prediction_subject_summary['best_chosen']['mean'],
                          marker='o',
                          color='C0',
                          alpha=0.5,
                          linewidth=0,
                          s=30)

        # c) Gaze Influence
        axs[m, 2].scatter(observed_subject_summary['gaze_influence'],
                          prediction_subject_summary['gaze_influence'],
                          marker='o',
                          color='none',
                          edgecolor='C0',
                          linewidth=0.5,
                          s=30)
        axs[m, 2].scatter(observed_subject_summary['gaze_influence'],
                          prediction_subject_summary['gaze_influence'],
                          marker='o',
                          color='C0',
                          alpha=0.5,
                          linewidth=0,
                          s=30)

        # update parameter ranges
        rt_range_prediction = extract_range(
            prediction_subject_summary['rt']['mean'], bound=limits['rt'])
        if rt_range[0] > rt_range_prediction[0]:
            rt_range[0] = rt_range_prediction[0]
        if rt_range[1] < rt_range_prediction[1]:
            rt_range[1] = rt_range_prediction[1]

        best_chosen_range_prediction = extract_range(
            prediction_subject_summary['best_chosen']['mean'], bound=limits['p_choose_best'])
        if best_chosen_range[0] > best_chosen_range_prediction[0]:
            best_chosen_range[0] = best_chosen_range_prediction[0]
        if best_chosen_range[1] < best_chosen_range_prediction[1]:
            best_chosen_range[1] = best_chosen_range_prediction[1]

        gaze_influence_range_prediction = extract_range(
            prediction_subject_summary['gaze_influence'], bound=limits['gaze_influence'])
        if gaze_influence_range[0] > gaze_influence_range_prediction[0]:
            gaze_influence_range[0] = gaze_influence_range_prediction[0]
        if gaze_influence_range[1] < gaze_influence_range_prediction[1]:
            gaze_influence_range[1] = gaze_influence_range_prediction[1]

        # label axes
        axs[m, 0].set_ylabel('{}\n\nPredicted Mean RT (s)'.format(
            prediction_labels[m]),
                             fontsize=fontsize)
        axs[m, 0].set_xlabel('Observed Mean RT (s)', fontsize=fontsize)
        axs[m, 1].set_ylabel('Predicted P(choose best)', fontsize=fontsize)
        axs[m, 1].set_xlabel('Observed P(choose best)', fontsize=fontsize)
        axs[m, 2].set_ylabel('Predicted Gaze Influence\non P(choice | value)',
                             fontsize=fontsize)
        axs[m, 2].set_xlabel('Observed Gaze Influence\non P(choice | value)',
                             fontsize=fontsize)

    # update axes limits and ticks
    if (rt_range[1] - rt_range[0]) > 3:
        rt_tickstep = 1.5
    else:
        rt_tickstep = 0.75
    rt_ticks = np.arange(rt_range[0], rt_range[1] + rt_tickstep,
                         rt_tickstep)
    for ax in axs[:, 0]:
        ax.set_yticks(rt_ticks)
        ax.set_xticks(rt_ticks)
        ax.set_xlim(rt_range)
        ax.set_ylim(rt_range)
        
    best_chosen_ticks = np.arange(0,1.1,0.2)
    for ax in axs[:, 1]:
        ax.set_yticks(best_chosen_ticks)
        ax.set_xticks(best_chosen_ticks)
        ax.set_xlim(best_chosen_range)
        ax.set_ylim(best_chosen_range)
        
    gaze_influence_ticks = np.arange(-1,1.1,0.2)
    for ax in axs[:, 2]:
        ax.set_yticks(gaze_influence_ticks)
        ax.set_xticks(gaze_influence_ticks)
        ax.set_xlim(gaze_influence_range)
        ax.set_ylim(gaze_influence_range)
       
    # label panels
    for label, ax in zip(list('ABCDEF'), axs.ravel()):
        ax.text(-0.4,
                1.1,
                label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                va='top')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=fontsize)
        # plot diagonal
        ax.plot(ax.get_xlim(),
                ax.get_xlim(),
                linewidth=1,
                color='black',
                alpha=1.0,
                zorder=-1)

    fig.tight_layout()

    return fig


def behaviour_parameter_correlation(estimates,
                                    subject_summary,
                                    figsize=cm2inch(18., 6),
                                    alpha=0.5,
                                    fontsize=7):

    # create figure
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # plot v vs Mean RT
    v_range = extract_range(estimates['v'])
    plot_correlation(estimates['v'],
                     subject_summary['rt']['mean'],
                     alpha=alpha,
                     markercolor='C0',
                     regression=True,
                     color='gray',
                     xlabel=r'$\hat{v}$',
                     ylabel='Mean RT (s)',
                     xlim=v_range,
                     ax=axs[0])

    # plot gamma vs gaze influence
    gamma_range = extract_range(estimates['gamma'])
    plot_correlation(estimates['gamma'],
                     subject_summary['gaze_influence'],
                     alpha=alpha,
                     markercolor='C0',
                     regression=True,
                     xlabel=r'$\hat{\gamma}$',
                     ylabel='Gaze influence on\nP(choice | value)',
                     xlim=gamma_range,
                     ax=axs[1],
                     annotation_pos=(0.1, 0.9),
                     color='gray')

    # plot gamma vs best chosen
    plot_correlation(estimates['gamma'],
                     subject_summary['best_chosen']['mean'],
                     alpha=alpha,
                     markercolor='C0',
                     regression=True,
                     color='gray',
                     xlabel=r'$\hat{\gamma}$',
                     ylabel='P(choose best)',
                     xlim=gamma_range,
                     ax=axs[2])
    # Labels
    for label, ax in zip(list('ABC'), axs.ravel()):
        ax.text(-0.25,
                1.1,
                label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                va='top')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Font sizes
        ax.tick_params(axis='both', labelsize=fontsize)

    fig.tight_layout(pad=1)

    return fig


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

    Args:
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


def plot_node_hierarchical(model,
                           parameters=['v', 'gamma', 's', 'tau'],
                           comparisons=None,
                           xlimits=dict(v=dict(dist=(0, 1),
                                               delta=(-0.5, 0.5)),
                                        gamma=dict(dist=(-1, 1),
                                                   delta=(-0.5, 1.5)),
                                        s=dict(dist=(0, 0.5),
                                               delta=(-0.1, 0.1)),
                                        tau=dict(dist=(0, 3),
                                                 delta=(-1, 1))),
                           fontsize=7):
    """Plot group nodes and comparisons from hierarchical model.

    Args:
        model (glambox.GLAM): GLAM model of type 'hierarchical'
        parameters (list): List of parameter names (e.g., ['v', 'gamma'])
        comparisons (list, optional): List of condition pairs (e.g., [('A', 'B')])

    Returns:
        fig, {axs}
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


def traceplot(trace, varnames='all', combine_chains=False,
              ref_val={}):
    """A traceplot replacement, because arviz is broken.
    This is tested for traces that come out of individual
    and hierarchical GLAM fits.

    Args:
        trace (PyMC.MultiTrace): A trace object.
        varnames (str, optional): List of variables to include
        combine_chains (bool, optional): Toggle concatenation of chains.
        ref_val (dict, optional): Reference values per parameter.

    Returns:
        figure, axes
    """
    if varnames == 'all':
        varnames = [var for var in trace.varnames
                    if not var.endswith('__')]
    nvars = len(varnames)
    if combine_chains:
        nchains = 1
    else:
        nchains = trace.nchains

    fig, axs = plt.subplots(nvars, 2, figsize=(8, nvars * 2))

    for v, var in enumerate(varnames):

        samples = trace.get_values(var, combine=combine_chains)
        if not isinstance(samples, list):
            samples = [samples]

        for chain in range(nchains):
            # group level parameters are (nsamples)
            # but subject level parameters are (nsamples x nsubjects x nconditions)
            chain_samples = samples[chain]
            if chain_samples.ndim == 1:
                chain_samples = chain_samples[:, None, None]
            nsamples, nsubjects, nconditions = chain_samples.shape

            for i in range(nsubjects):
                # Trace
                axs[v, 0].set_xlabel('')
                axs[v, 0].set_ylabel('Sample value')
                axs[v, 0].set_title(var)
                axs[v, 0].plot(chain_samples[:, i, :],
                               alpha=0.3)

                # KDE
                sns.kdeplot(chain_samples[:, i, 0],
                            ax=axs[v, 1])
                axs[v, 1].set_title(var)
                axs[v, 1].set_ylabel('Frequency')

        # Reference Values
        if ref_val.get(var, False) is not False:
            axs[v, 0].axhline(ref_val.get(var),
                              color='black', linewidth=2)
            axs[v, 1].axvline(ref_val.get(var),
                              color='black', linewidth=2)

    fig.tight_layout()
    return fig, axs
