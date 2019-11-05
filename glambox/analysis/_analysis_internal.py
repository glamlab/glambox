#!/usr/bin/python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pymc3 import hpd, summary


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
    data : dataframe
        aggregate response data

    n_items : int, optional
        number of choice alternatives in data

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
    Computes subject-wise mean RT

    Input
    ---
    df : dataframe
        aggregate response data

    Returns
    ---
        array of subject-wise mean RTs
    """
    return df.groupby('subject').rt.mean().values


def compute_p_choose_best(df):
    """
    Computes subject-wise P(choose best)

    Input
    ---
    df : dataframe
        aggregate response data

    Returns
    ---
        array of subject-wise P(choose best)
    """
    if 'best_chosen' not in df.columns:
        df = add_best_chosen(df)
    return df.groupby('subject').best_chosen.mean().values


def add_best_chosen(df):
    """
    Adds 'best_chosen' variable to DataFrame,
    independent of number of items
    (works with nan columns)

    Input
    ---
    df : dataframe
        aggregate response data

    Returns
    ---
        copy of df with 'best chosen' indicator 
    """
    df = df.copy()
    values = df[[c for c in df.columns if c.startswith('item_value_')]].values
    choices = df['choice'].values.astype(np.int)
    best_chosen = (values[np.arange(choices.size), choices] == np.nanmax(
        values, axis=1)).astype(int)
    df['best_chosen'] = best_chosen
    return df


def run_linear_model(x, y, verbose=True):
    """
    Compute a linear regression model, 
    regressing y onto x

    Input
    ---
    x : ndarray
        independent variable data

    y : array
        dependent variable data

    verbose : bool, optional
        whether to print results of
        linear model fit

    Returns
    ---
        fitted statsmodels OLS object 
    """

    X = sm.add_constant(x)
    lm = sm.OLS(y, X).fit()

    if verbose:
        print(lm.summary())
        print('Slope = {:.2f}'.format(lm.params[-1]))
        print('t({:d}) = {:.2f}'.format(int(lm.df_resid), lm.tvalues[-1]))
        print('P = {:.10f}'.format(lm.pvalues[-1]))
    return lm


def q1(series):
    """
    Extract 25% quantile from 
    pandas series

    Input
    ---
        series : pandas series

    Returns
    ---
        25% quantile
    """
    q1 = series.quantile(0.25)
    return q1


def q3(series):
    """
    Extract 75% quantile from 
    pandas series

    Input
    ---
        series : pandas series

    Returns
    ---
        27% quantile
    """
    q3 = series.quantile(0.75)
    return q3


def iqr(series):
    """
    Extract inter-quantile range
    (25%-75%)

    Input
    ---
        series : pandas series

    Returns
    ---
        inter-quantile range
        (25-75%)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return IQR


def std(series):
    """
    Extract standar deviation (SD)

    Input
    ---
        series : pandas series

    Returns
    ---
        SD
    """
    sd = series.std(ddof=0)
    return sd


def se(series):
    """
    Extract standar error (SE)

    Input
    ---
        series : pandas series

    Returns
    ---
        SE
    """
    n = len(series)
    se = series.std() / np.sqrt(n)
    return se



def compare_parameters_hierarchical(model,
                           parameters=['v', 'gamma', 's', 'tau'],
                           comparisons=None):
    """Compute comparisons of group level parameters between groups / conditions.
    
    Args:
        model (glambox.GLAM): Fitted GLAM instance
        parameters (list, optional): List of parameters. Defaults to ['v', 'gamma', 's', 'tau'].
        comparisons (list of tuples, optional): List of comparisons to perform. Must be a list of tuples, e.g., `[('A', 'B'), ('A', 'C')]`. Defaults to None.
    
    Returns:
        pandas.DataFrame: Distribution statistics of group level parameter differences.
    """

    if comparisons is None:
        comparisons = []
    n_params = len(parameters)
    n_comps = len(comparisons)

    comparison_df = []

    for p, parameter in enumerate(parameters):

        # Comparisons
        for c, comparison in enumerate(comparisons):
            comparison_string = '{}-{}'.format(*comparison)
            df_pc = pd.DataFrame(dict(parameter=parameter, comparison=comparison_string),
                                      index=[0])
            # Check if parameter has dependence
            if model.design[parameter]['dependence'] is not None:
                # Then, if both conditions are present, compute posterior of the difference
                c0_present = (comparison[0] in model.design[parameter]['conditions'])
                c1_present = (comparison[1] in model.design[parameter]['conditions'])
                if c0_present & c1_present:
                    difference = (
                        model.trace[0].get_values(parameter + '_' +
                                                  comparison[0] + '_mu') -
                        model.trace[0].get_values(parameter + '_' +
                                                  comparison[1] + '_mu'))
                    
                    hpd_lower, hpd_upper = hpd(difference, alpha=0.05)
                    df_pc['hpd_2.5'] = hpd_lower
                    df_pc['hpd_97.5'] = hpd_upper
                    df_pc['mean'] = np.mean(difference)
                    df_pc['p>0'] = np.mean(difference > 0)
                else:
                    # Otherwise, state that at least one condition is not present.
                    df_pc['warning'] = 'At least one condition is missing.'
            else:
                # Or that the parameter has no dependencies.
                df_pc['warning'] = 'Parameter has no dependencies.'
            
            comparison_df.append(df_pc)
    
    comparison_df = pd.concat(comparison_df, sort=False).reset_index(drop=True)

    return comparison_df


def compare_parameters_individual(model,
                                  parameters,
                                  comparisons=None):

    if comparisons is None:
        comparisons = []
    n_params = len(parameters)
    n_comps = len(comparisons)

    subjects = model.data['subject'].unique().astype(int)
    summaries = [summary(trace) for trace in model.trace]

    comparison_df = []
    
    for p, parameter in enumerate(parameters):

        # Comparisons
        for c, comparison in enumerate(comparisons):
            comparison_string = '{}-{}'.format(*comparison)
            df_pc = pd.DataFrame(dict(subject=subjects, parameter=parameter, comparison=comparison_string),
                                 index=subjects)

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
                    hpdlower, hpdupper = hpd(differences.T, alpha=0.05).T
                    plarger0 = np.mean(differences > 0, axis=1)
                    df_pc['mean'] = means
                    df_pc['hpd_2.5'] = hpdlower
                    df_pc['hpd_97.5'] = hpdupper
                    df_pc['p>0'] = plarger0

                else:
                    # Otherwise, state that at least one condition is not present.
                    df_pc['warning'] = 'At least one condition is missing.'
            else:
                # Or that the parameter has no dependencies.
                df_pc['warning'] = 'Parameter has no dependencies.'

            comparison_df.append(df_pc)
    
    comparison_df = pd.concat(comparison_df, sort=False).sort_values('subject').reset_index(drop=True)

    return comparison_df
