

def behaviour_parameter_correlation(estimates,
                                    subject_summary,
                                    figsize=cm2inch(18., 6),
                                    alpha=0.5,
                                    fontsize=7):
    """
    Correlation plot between [v, gamma] and 
    behavioral metrics ['rt', 'p(choose best)', 'gaze influence score']
    For details on the metrics, see the manuscript

    Input
    ---
    estimates : dict
        dict with array_like parameter estimates
        for 'v' and 'gamma'

    subject_summary : dataframe
        subject-level summary dataframe for subjects
        whose estimates are given,
        can be computed with analysis.aggregate_subject_level_data

    figsize : tuple, optional
        matplotlib figure size

    alpha : float, optional
        alpha level for scatter plots and
        regression line

    fontsize : int, optional
        plotting fontsize, defaults to 7

    Returns
    ---
    matplotlib figure object
    """

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
