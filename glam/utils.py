#!/usr/bin/python

import numpy as np
import pandas as pd
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


def get_design(data, depends_on):
    """
    Extract information about the experimental design
    from `data` and `depends_on`.
    This information is used to map parameter estimates
    back to subjects, etc.

    Parameters:
    data: df
    depends_on: dict

    Returns:
    dict
    """
    parameters = ['v', 'gamma', 's', 'tau', 't0']

    subject_idx = data['subject']
    subjects = subject_idx.unique()

    design = dict()
    
    for parameter in parameters:
        design[parameter] = dict()
        design[parameter]['subject_index'] = subject_idx[:].astype(np.int)
        dependence = depends_on.get(parameter)
        if dependence is not None:
            # extract condition levels
            conditions = data[dependence].unique()
            design[parameter]['conditions'] = conditions
            
            # Initialize empty design matrix D for this parameter
            D = np.zeros((subjects.size, conditions.size), dtype=np.int)
            # create parameter mapping containing condition names and indices (i.e., columns of design matrix)
            design[parameter]['condition_mapping'] = {condition: c
                                                      for c, condition in enumerate(conditions)}
            # create an array to index which condition each trial-entry in the data belongs to
            design[parameter]['condition_index'] = np.zeros_like(subject_idx, dtype=np.int)

            # For each condition level
            for c, condition in enumerate(conditions):
                # mark which trial-entries belong to this condition
                design[parameter]['condition_index'][data[dependence]==condition] = c

                design[parameter][condition] = dict()
                # Subset data to condition-specific data
                data_subset = data[data[dependence]==condition].copy()
                # find all subject_IDs in this condition
                subject_subset = data_subset['subject'].unique()
                # within this level, generate mapping between subject_ids and indices
                # e.g., gamma_high[5] corresponds to subject_id 10
                design[parameter][condition]['subject_mapping'] = {s: int(subject)
                                                               for s, subject in enumerate(subject_subset)}
                # Set cells with subject in this condition to 1
                for s, subject in enumerate(subject_subset):
                    D[np.int(subject), c] = np.int(s+1)
        else:
            D = (np.arange(subjects.size)[:,None] + 1).astype(np.int)
            design[parameter]['conditions'] = None
            design[parameter]['condition_index'] = np.zeros_like(subject_idx, dtype=np.int)

        # Save design matrix D
        design[parameter]['D'] = D

        # Detect within / between subject design
        if D.shape[1] == 1:
            design_type = 'fixed'
        elif np.all((D!=0).sum(axis=1) > 1):
            design_type = 'within'
        elif np.all((D!=0).sum(axis=1) == 1):
            design_type = 'between'
        else:
            design_type = 'mixed'
        design[parameter]['type'] = design_type

    return design


def map_individual_estimates(model):
    """
    Generate a DataFrame containing estimates,
    with one row per participant and one column
    per parameter. Note: If a parameter has multiple
    conditions, multiple columns are added (e.g.,
    v_slow, v_high).

    Parameters:
    model: GLAM.model

    Returns:
    DataFrame
    """
    design = get_design(model)
    subjects = model.data['subject'].unique()
    parameters = ['v', 'gamma', 's', 'tau', 't0']

    if model.type == 'hierarchical':
        estimate_df = pd.DataFrame(dict(subject=subjects))
        for parameter in parameters:
            dependence = model.depends_on.get(parameter, None)
            if dependence is not None:
                conditions = model.data[dependence].unique()
                for condition in conditions:
                    par_con = parameter + '_' + condition
                    condition_index = design[parameter + '_mapping'][condition]
                    subject_map = design['subject_parameter_mapping'][par_con]
                    estimate_df[par_con] = np.zeros(subjects.size) * np.nan
                    for i, estimate in enumerate(model.estimates[par_con]):
                        estimate_df[par_con][estimate_df['subject'] == subject_map[i]] = estimate

            else:
                estimate_df[parameter] = np.zeros(subjects.size) * np.nan
                for i, estimate in enumerate(model.estimates[parameter]):
                    estimate_df[parameter][estimate_df['subject'] == i] = estimate
    elif model.type == 'individual':
        estimate_df = pd.DataFrame()
        for s, subject_estimates in enumerate(model.estimates):
            subject_estimates_flat = {key: val.ravel() for key, val in subject_estimates.items()}
            tmp = pd.DataFrame(subject_estimates_flat, index=np.array([s]))
            tmp['subject'] = s
            estimate_df = pd.concat([estimate_df, tmp])
    else:
        raise ValueError('Model type not understood. Have you called `make_model` before?')
    return estimate_df
