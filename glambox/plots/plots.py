#!/usr/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ._plots_internal import *
from glambox.analysis import aggregate_subject_level_data


def plot_behaviour_aggregate(bar_data,
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
    """
    Create a group-level aggregate plot with 
    the following four metrics:
    A) RT ~ (max value - max value others)
    B) P(choose best) ~ (item value - max value others)
    C) P(choose best) ~ (item gaze - max gaze others)
    D) Corrected P(choose best) ~ (item gaze - max gaze others)
    For further details on these measures, see the manuscript

    Parameters
    ----------
    bar_data : pandas.DataFrame
        response data to plot as bars

    line_data : list of pandas.DataFrames, optional
        response data to plot as colored lines

    line_labels : array_like, strings, optional
        legend labels for line_data

    fontsize : int, optional
        fontsize for plotting,
        defaults to 7

    value_bins : int or array_like, optional
        x-axis bins for panels A - B
        if an int is given, this many bins will be created,
        defaults to 7

    gaze_bins :  int or array_like, optional
        x-axis bins for panels A - B
        if an int is given, this many bins will be created,
        defaults to 7

    limits : dict, optional
        dict containing one entry for:
        ['rt', 'p_choose_best', 'corrected_p_choose_best']
        each entry is a tuple, defining the y-limits for
        the respective metrics

    Returns
    -------
    Tuple
        matplotlib figure and axes object
    """
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
    axs[1] = plot_pchoose_by_value_minus_max_others(bar_data,
                                            line_data,
                                            xlabel_skip=4,
                                            ax=axs[1],
                                            bins=value_bins,
                                            fontsize=fontsize,
                                            ylims=limits['p_choose_best'])
    axs[2] = plot_pchoose_by_gaze_minus_max_others(bar_data,
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


def plot_behaviour_associations(data,
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
    """
    Plot individual differences on
    the following three panels:
    D) p(choose best) ~ response time
    E) gaze influence score ~ response time
    F) gaze influence score ~ p(choose best)

    In addition to each correlation plot,
    distributions of the three underlying 
    behavioural measures are given in
    panels A-C:
    A) Response time
    B) P(choose best)
    C) Gaze influence score

    For further details on these measures,
    see the manuscript

    Parameters
    ----------
    data : pandas.DataFrame
        response data

    nbins : int, optional
        defining the number of bins to
        use for the marginal histograms

    fontsize : int, optional
        defining the plotting fontsize,
        defaults to 7

    regression : bool, optional
        whether to compute and plot
        a linear regression fit,
        defaults to True

    annotate : bool, optional
        whether to add pearson's r
        correlation coefficient and p-value
        to plot,
        defaults to True

    figsize : tuple, optional
        size of of plotting figure

    limits : dict, optional
        dict containing one entry for:
        ['rt', 'p_choose_best', 'corrected_p_choose_best']
        each entry is a tuple, defining the y-limits for
        the respective metrics

    Returns
    -------
    Tuple
        matplotlib figure object, axs

    """

    if (regression == False) & (annotate == True):
        print('annotate only possible, if regression = True.')

    fig = plt.figure(figsize=figsize, dpi=330)

    ax00 = plt.subplot2grid((3, 3), (0, 0), rowspan=1)
    ax10 = plt.subplot2grid((3, 3), (1, 0), rowspan=2)

    ax01 = plt.subplot2grid((3, 3), (0, 1), rowspan=1)
    ax11 = plt.subplot2grid((3, 3), (1, 1), rowspan=2)

    ax02 = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
    ax12 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

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
    plot_correlation(subject_summary['best_chosen']['mean'],
                     subject_summary['rt']['mean'],
                     marker='o',
                     markercolor='C0',
                     regression=regression,
                     annotate=annotate,
                     annotation_pos=(0.1, 0.01),
                     ylabel='Mean RT (s)',
                     xlabel='P(choose best)',
                     ylim=rt_range,
                     yticks=rt_ticks,
                     xlim=best_chosen_range,
                     xticks=best_chosen_ticks,
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

    plot_correlation(subject_summary['gaze_influence'],
                     subject_summary['best_chosen']['mean'],
                     marker='o',
                     markercolor='C0',
                     regression=regression,
                     annotate=annotate,
                     annotation_pos=(0.1, 0.01),
                     xlabel='Gaze influence\non P(choice | value)',
                     ylabel='P(choose best)',
                     xlim=gaze_influence_range,
                     xticks=gaze_influence_ticks,
                     ylim=best_chosen_range,
                     yticks=best_chosen_ticks,
                     fontsize_title=fontsize,
                     fontsize_axeslabel=fontsize,
                     fontsize_ticklabels=fontsize,
                     fontsize_annotation=fontsize,
                     ax=ax12)

    # Marginal histograms
    ax00.hist(subject_summary['rt']['mean'],
              bins=np.linspace(rt_range[0], rt_range[1], nbins + 1),
              color='C0')

    ax01.hist(subject_summary['best_chosen']['mean'],
              bins=np.linspace(best_chosen_range[0], best_chosen_range[1],
                               nbins + 1),
              color='C0')

    ax02.hist(subject_summary['gaze_influence'],
              bins=np.linspace(gaze_influence_range[0],
                               gaze_influence_range[1], nbins + 1),
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
    for ax, ax_xticks, ax_xlabel, ax_xlim in zip(np.array([ax00, ax01, ax02]),
                                                 [rt_ticks, best_chosen_ticks, gaze_influence_ticks],
                                                 ['Mean RT (s)', 'P(choose best)', 'Gaze influence\non P(choice | value)'],
                                                 [rt_range, best_chosen_range, gaze_influence_range]) :
        ax.set_xticks(ax_xticks)
        ax.set_xticklabels(ax_xticks)
        ax.set_xlim(ax_xlim)
        ax.set_ylim([0, hist_lim])
        ax.set_yticks([0, hist_lim])
        ax.set_yticklabels([0, hist_lim], fontsize=fontsize)
        ax.set_ylabel('Frequency', fontsize=fontsize)
        ax.set_xlabel(ax_xlabel, fontsize=fontsize)

    for ax in [ax00, ax01, ax02, ax10, ax11, ax12]:
        sns.despine(ax=ax)

    #fig.tight_layout()

    return fig#, [ax00, ax01, ax02, ax10, ax11, ax12]


def plot_individual_fit(observed,
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
    """
    Plot individual observed vs predicted data
    on three metrics:
    A) response time
    B) p(choose best)
    C) gaze influence score
    For details on these measures, 
    see the manuscript

    Parameters
    ----------
    observed : pandas.DataFrame
        observed response data

    predictions : list of pandas.DataFrame
        predicted response datasets

    prediction_labels : array_like, strings, optional
        legend labels for predictions

    colors : array_like, strings, optional
        colors to use for predictions

    fontsize : int, optional
        plotting fontsize

    alpha : float, optional
        alpha level for predictions
        should be between [0,1]

    figsize : tuple, optional
        matplotlib figure size

    limits : dict, optional
        dict containing one entry for:
        ['rt', 'p_choose_best', 'corrected_p_choose_best']
        each entry is a tuple, defining the y-limits for
        the respective metrics

    Returns
    ---
    Tuple
        matplotlib figure object, axs
    """

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

    return fig, axs


def compare_parameters(model,
                       parameters=['v', 's', 'gamma', 'tau'],
                       comparisons=None, **kwargs):
    """Plot posterior distributions of parameters
    and differences between groups or conditions.
    
    Parameters
    ----------
    model : glambox.GLAM instance
        A fitted GLAM instance.
    parameters : list of str, optional
        A list of parameters to be plotted. 
        Defaults to all model parameters.
    comparisons : List of tuples, optional
        List of pairwise comparisons between groups
        or conditions to be plotted. Each comparison
        must be given as a tuple of two conditions
        (e.g., [('A', 'B'), ('A', 'C')])
    
    Returns
    -------
    Tuple
        matplotlib figure object, axs
    """
    if model.type == 'individual':
        fig, axs = compare_parameters_individual(
            model=model, parameters=parameters, comparisons=comparisons, **kwargs)
    elif model.type == 'hierarchical':
        fig, axs = compare_parameters_hierarchical(
            model=model, parameters=parameters, comparisons=comparisons, **kwargs)
    else:
        raise ValueError('Model type not understood.')
    return fig, axs


def traceplot(trace, varnames='all', combine_chains=False,
              ref_val={}):
    """
    A traceplot replacement, because arviz is broken.
    This is tested for traces that come out of individual
    and hierarchical GLAM fits.

    Parameters
    ----------
    trace : PyMC.MultiTrace)
        a trace object.
    
    varnames : str, optional
        list of variables to include
        
    combine_chains : bool, optional
        toggle concatenation of chains.
    
    ref_val : dict, optional)
        reference values per parameter.

    Returns
    -------
    Tuple
        matplotlib figure and axes objects
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
