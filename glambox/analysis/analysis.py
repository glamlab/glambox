#!/usr/bin/python
import numpy as np
import pandas as pd
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

    Input
    ---
    df : dataframe
        aggregate response data

    n_items : int
        number of choice alternatives in the data

    Returns
    ---
    df : dataframe
        df of subject-level response characteristics
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


def aggregate_group_level_data(data, n_items):
    """
    Compute group-level response characteristics on:
    RT, P(choose best), gaze influence score

    The gaze influence score is defined
    as the average difference between
    the corrected choice probability
    of all positive and negative relative gaze values
    (see manuscript)

    Input
    ---
    df : dataframe
        aggregate response data

    n_items : int
        number of choice alternatives in the data

    Returns
    ---
    df : dataframe
        df of group-level response characteristics
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



def compare_parameters(model, parameters, comparisons=None, **kwargs):
    """Perform comparisons between parameters and return statistics as DataFrame
    
    Args:
        model ([type]): [description]
        parameters ([type]): [description]
        comparisons ([type], optional): [description]. Defaults to None.
    
    Raises:
        ValueError: [description]
    
    Returns:
        pandas.DataFrame: Distribution statistics of parameter differences.
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
    """Compares multiple models.
    
    Args:
        models (list): List of fitted GLAM model instances.
        **kwargs (optionsl): Additional keyword arguments to be passed to pymc3.compare
    
    Returns:
        pandas.DataFrame containing information criteria for each model.
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
    