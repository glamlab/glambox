#!/usr/bin/python

import numpy as np
from scipy.stats import mode


def format_data(df):
    """
    Extracts and formats data
    from <pandas.DataFrame> to model friendly entities.
    """
    subjects = df['subject'].unique()
    n_subjects = len(subjects)
    n_items = len([col for col in df.columns
                   if col.startswith('item_value_')])
    subject_idx = np.array(df.subject.values.astype(int))
    gaze = df[['gaze_{}'.format(i) for i in range(n_items)]].values
    values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
    choice = df['choice'].values.astype(int)
    rts = df['rt'].values
    rts = rts.astype('int')
    gaze_sorted = sort_according_to_choice(gaze, choice)
    values_sorted = sort_according_to_choice(values, choice)

    # compute random choice likelihood
    rtmax = df.groupby('subject').rt.max().values
    rtmin = df.groupby('subject').rt.min().values
    error_lls = 1 / (n_items * (rtmax-rtmin))

    output = dict(subjects=subjects,
                  n_subjects=n_subjects,
                  n_items=n_items,
                  subject_idx=subject_idx,
                  gaze=gaze_sorted,
                  values=values_sorted,
                  rts=rts,
                  error_lls=error_lls)
    return output


def sort_according_to_choice(x, choices):
    """
    Sort a <numpy.array> x according to choice indices.
    """
    x_sorted = np.zeros_like(x) * np.nan
    # Get values of chosen entries, and put them into first column
    x_sorted[:, 0] = x[np.arange(x.shape[0]), choices.astype(int)]
    # and everything else into the next columns
    others = np.vstack([x[i, np.where(np.arange(x.shape[1]) != c)]
                        for i, c in enumerate(choices)])
    x_sorted[:, 1:] = others
    return x_sorted


def extract_modes(traces, parameters=None, precision=None, f_burn=0.5):

    if not isinstance(traces, list):
        traces = [traces]

    modes = []

    for trace in traces:

        if parameters is None:
            parameters = [var for var in trace.varnames
                          if not var.endswith('__')]

            print('/!\ Automatically setting parameter precision...')
            precision_defaults = dict(v=6, gamma=2, s=6, SNR=2, tau=2, t0=-1)
            precision = [precision_defaults.get(parameter.split('_')[0], 6)
                         for parameter in parameters]

        n_samples = len(trace)
        trace_modes = {}

        for parameter, prec in zip(parameters, precision):
            trace_modes[parameter] = mode(np.round(trace.get_values(parameter, burn=int(f_burn*n_samples)), prec))[0][0]
        modes.append(trace_modes)

    if len(modes) == 1:
        return modes[0]
    else:
        return modes
