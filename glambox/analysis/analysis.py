#!/usr/bin/python
import numpy as np
import pandas as pd
import pymc3 as pm
from ._analysis_internal import *


def aggregate_subject_level_data(data, n_items):
    """
    Compute subject-level response characteristics on:
    RT, P(choose best), gaze influence score

    The gaze influence score is defined
    as the average difference between
    the corrected choice probability
    of all positive and negative relative gaze values
    (see manuscript)

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the experimental data.  
        Each row corresponds to one trial.  
        Must include the following columns:  

        - `subject` (int, consecutive, starting with 0)
        - `trial` (int, starting with 0)
        - `choice` (int, items should be 0, 1, ..., N)
        - `rt` (float, in seconds) 
        - additional variables coding groups or conditions (str or int)
        
        For each item `i` in the choice set:  

        - `item_value_i`: The item value (float, best on a scale between 1 and 10)
        - `gaze_i`: The fraction of total trial time the item was looked at in the trial (float, between 0 and 1)
    
    n_items : int
        number of choice alternatives in the data

    Returns
    -------
    pandas.DataFrame
        DataFrame of subject-level response characteristics.
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

    return subject_summary


def aggregate_group_level_data(data, n_items):
    """
    Compute group-level response characteristics on:
    RT, P(choose best), gaze influence score

    The gaze influence score is defined
    as the average difference between
    the corrected choice probability
    of all positive and negative relative gaze values
    (see manuscript)

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the experimental data.  
        Each row corresponds to one trial.  
        Must include the following columns:  

        - `subject` (int, consecutive, starting with 0)
        - `trial` (int, starting with 0)
        - `choice` (int, items should be 0, 1, ..., N)
        - `rt` (float, in seconds) 
        - additional variables coding groups or conditions (str or int)
        
        For each item `i` in the choice set:  

        - `item_value_i`: The item value (float, best on a scale between 1 and 10)
        - `gaze_i`: The fraction of total trial time the item was looked at in the trial (float, between 0 and 1)

    n_items : int
        number of choice alternatives in the data

    Returns
    -------
    pandas.DataFrame
        DataFrame of group-level response characteristics
    """
    subject_summary = aggregate_subject_level_data(data, n_items)
    group_summary = subject_summary.agg({
        ('rt', 'mean'): ['mean', std, 'min', 'max', se, iqr],
        ('best_chosen', 'mean'): ['mean', std, 'min', 'max', se, iqr],
        'gaze_influence': ['mean', std, 'min', 'max', se, iqr]
    })
    group_summary = group_summary[[('rt', 'mean'), ('best_chosen', 'mean'),
                                   ('gaze_influence')]].copy()
    group_summary.columns = ['Mean RT', 'P(choose best)', 'Gaze Influence']
    return group_summary.T



def compare_parameters(model,
                       parameters=['v', 's', 'gamma', 'tau'],
                       comparisons=None, **kwargs):
    """
    Perform comparisons between parameters and return statistics as DataFrame
    
    Parameters
    ----------
    model : glambox.GLAM
        Fitted glambox.GLAM instance
    
    parameters : list of str, optional
        List of parameters to perform comparisons on.
        Defaults to all model parameters.
    
    comparisons : list of tuples, optional
        List of comparisons between groups or conditions.
        Each comparison must be given as a tuple
        (e.g., [('A', 'B'), ('A', 'C')])
        Defaults to None.
    
    Returns
    -------
    pandas.DataFrame
        Distribution statistics of parameter differences.
    """
    if model.type == 'individual':
        comparison_df = compare_parameters_individual(
            model=model, parameters=parameters, comparisons=comparisons)
    elif model.type == 'hierarchical':
        comparison_df = compare_parameters_hierarchical(
            model=model, parameters=parameters, comparisons=comparisons)
    else:
        raise ValueError('Model type not understood.')
    return comparison_df



def compare_models(models, **kwargs):
    """
    Compares multiple fitted models.
    
    Parameters
    ----------
    models : list of glambox.GLAM
        List of fitted GLAM model instances.
    
    **kwargs : optional
        Additional keyword arguments to be passed to pymc3.compare
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing information criteria for each model.
    """

    # Check that more than one model is entered
    assert len(models) > 1, "Must enter at least two models."

    # Check model names, create some if there are none
    for m, model in enumerate(models):
        if model.name is None:
            model.name = 'model_{}'.format(m)

    # Check that all models have the same type:
    assert all([model.type == models[0].type for model in models]
               ), "Models have different types and cannot be compared."

    # Check that all models have the same number of PyMC3 models and traces:
    assert all([len(model.trace) == len(models[0].trace) for model in models]
               ), "Model instances have different numbers of subjects and cannot be compared."

    if models[0].type == 'hierarchical':
        df = pm.compare(model_dict={model.model: model.trace[0]
                                    for model in models},
                        **kwargs)
        # read out column names
        cols = df.columns.tolist()
        # include model column
        df.index.name = 'model'
        df = df.reset_index()
        # reorder columns so that model comes first
        df = df[['model'] + cols]

    elif models[0].type == 'individual':
        df = []
        for s in range(len(models[0].trace)):
            compare_df_s = pm.compare(model_dict={model.model[s]: model.trace[s]
                                                  for model in models},
                                      **kwargs)
            # read out column names
            cols = compare_df_s.columns.tolist()
            # include subject column
            compare_df_s['subject'] = s
            # include model column
            compare_df_s.index.name = 'model'
            compare_df_s = compare_df_s.reset_index()
            # reorder columns so that subject and model come first
            compare_df_s = compare_df_s[['subject', 'model'] + cols]
            df.append(compare_df_s)
        df = pd.concat(df).reset_index(drop=True)

    return df
    
