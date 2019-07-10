#!/usr/bin/python

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_gaze_influence_score(data, n_items=None):
    """
    Compute gaze influence score for each
    subject in the data;
    Gaze influence score is defined
    as the average difference between
    the corrected choice probability
    of all positive and negative relative gaze values
    (see manuscript).
    Input
    ---
    data (dataframe):
            aggregate response data
    Returns
    ---
    array of single-subject gaze influence scores
    """

    import statsmodels.api as sm

    data = data.copy()

    choice = np.zeros((data.shape[0], n_items))
    choice[np.arange(data.shape[0]), data['choice'].values.astype('int32')] = 1

    # compute value measures
    gaze = data[['gaze_{}'.format(i) for i in range(n_items)]].values
    rel_gaze = np.zeros_like(gaze)
    values = data[['item_value_{}'.format(i) for i in range(n_items)]].values
    rel_values = np.zeros_like(values)
    value_range = np.zeros_like(values)
    for t in range(values.shape[0]):
        for i in range(n_items):
            index = np.where(np.arange(n_items) != i)
            rel_gaze[t, i] = gaze[t, i] - np.mean(gaze[t, index])
            rel_values[t, i] = values[t, i] - np.mean(values[t, index])
            value_range[t, i] = np.max(values[t, index]) - np.min(
                values[t, index])

    # create new dataframe (long format, one row per item)
    data_long = pd.DataFrame(
        dict(subject=np.repeat(data['subject'].values, n_items),
             is_choice=choice.ravel(),
             value=values.ravel(),
             rel_value=rel_values.ravel(),
             value_range_others=value_range.ravel(),
             rel_gaze=rel_gaze.ravel(),
             gaze_pos=np.array(rel_gaze.ravel() > 0, dtype=np.bool),
             gaze=gaze.ravel()))

    # estimate value-based choice prob.
    # for each individual and subtract
    # from empirical choice
    data_out_list = []
    for s, subject in enumerate(data['subject'].unique()):
        subject_data = data_long[data_long['subject'] == subject].copy()

        # Only add range of other items if more than 2 items (otherwise it's just another constant, resulting in a singular matrix)
        if n_items > 2:
            X = subject_data[['rel_value', 'value_range_others']]
        else:
            X = subject_data[['rel_value']]

        X = sm.add_constant(X)
        y = subject_data['is_choice']

        logit = sm.Logit(y, X)
        result = logit.fit(disp=0, maxiter=100)  # method='lbfgs',
        predicted_pchoose = result.predict(X)

        subject_data['corrected_choice'] = subject_data['is_choice'] - \
            predicted_pchoose
        data_out_list.append(subject_data)

    data_out = pd.concat(data_out_list)

    # compute corrected psychometric, given gaze
    tmp = data_out.groupby(['subject',
                            'gaze_pos']).corrected_choice.mean().unstack()
    gaze_influence_score = (tmp[True] - tmp[False]).values

    return gaze_influence_score


def compute_mean_rt(df):
    """
    Computes subject wise mean RT
    """
    return df.groupby('subject').rt.mean().values


def compute_p_choose_best(df):
    """
    Computes subject wise P(choose best)
    """
    if 'best_chosen' not in df.columns:
        df = add_best_chosen(df)
    return df.groupby('subject').best_chosen.mean().values


def add_best_chosen(df):
    """
    Adds 'best_chosen' variable to DataFrame,
    independent of number of items (works with nan columns)
    """
    df = df.copy()
    values = df[[c for c in df.columns if c.startswith('item_value_')]].values
    choices = df['choice'].values.astype(np.int)
    best_chosen = (values[np.arange(choices.size), choices] == np.nanmax(
        values, axis=1)).astype(int)
    df['best_chosen'] = best_chosen
    return df


def run_linear_model(x, y, verbose=True):

    X = sm.add_constant(x)
    lm = sm.OLS(y, X).fit()

    if verbose:
        print(lm.summary())
        print('Slope = {:.2f}'.format(lm.params[-1]))
        print('t({:d}) = {:.2f}'.format(int(lm.df_resid), lm.tvalues[-1]))
        print('P = {:.10f}'.format(lm.pvalues[-1]))
    return lm


def q1(series):
    q1 = series.quantile(0.25)
    return q1


def q3(series):
    q3 = series.quantile(0.75)
    return q3


def iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return IQR


def std(series):
    sd = series.std(ddof=0)
    return sd


def se(series):
    n = len(series)
    se = series.std() / np.sqrt(n)
    return se


def aggregate_subject_level_data(data, n_items):
    """
    Aggregates a single dataset to subject level
    """
    data = data.copy()

    # add best chosen variable
    data = add_best_chosen(data)

    # Summarize variables
    subject_summary = data.groupby('subject').agg({
        'rt': ['mean', std, 'min', 'max', se, q1, q3, iqr],
        'best_chosen':
        'mean'
    })
    # Influence of gaze on P(choose left)
    subject_summary['gaze_influence'] = compute_gaze_influence_score(
        data, n_items=n_items)

    # subject_summary['dataset'] = data.groupby('subject')['dataset'].head(1).values

    return subject_summary


def aggregate_group_level_data(subject_summary):
    """
    Aggregates a subject summary to group level
    """
    group_summary = subject_summary.agg({
        ('rt', 'mean'): ['mean', std, 'min', 'max', se, iqr],
        ('best_chosen', 'mean'): ['mean', std, 'min', 'max', se, iqr],
        'gaze_influence': ['mean', std, 'min', 'max', se, iqr]
    })
    group_summary = group_summary[[('rt', 'mean'), ('best_chosen', 'mean'),
                                   ('gaze_influence')]].copy()
    group_summary.columns = ['Mean RT', 'P(choose best)', 'Gaze Influence']
    return group_summary.T
