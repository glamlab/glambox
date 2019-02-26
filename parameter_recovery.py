#!/usr/bin/python

import glam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymc3 import summary, traceplot, trace_to_dataframe
from itertools import product
from os.path import join, isfile


def recover_glam(parameters, simulate_group_args=None, make_model_args=None, fit_args=None):
    """
    Recover a single GLAM. This function
    1) Initializes the model
    2) Generates synthetic data using `simulate_group_args` and `parameters`
    3) Builds the model using `make_model_args`
    4) Fits the model using `fit_args`.

    Args:
        parameters: Parameter argument for `simulate_group` methods.
        simulate_group_args (dict): dict of kwargs to `simulate_group` method
        make_model_args (dict): dict of kwargs to `make_model` method
        fit_args (dict): dict of kwargs to `fit` method

    Returns:
        glam.model: Fitted model instance.
    """
    simulate_group_args.update(parameters=parameters)

    G = glam.GLAM()

    G.simulate_group(**simulate_group_args)
    G.make_model(**make_model_args)
    G.fit(**fit_args)

    return G


def generate_parameter_sets(parameter_info, levels=['low', 'medium', 'high'], parameters=['v', 'gamma', 's', 'tau']):
    """
    Generates parameter sets.

    Args:
        parameter_info (dict): dict containing information about parameter levels.
        levels (list, optional): list of level labels
        parameters (list, optional): list of parameter names

    Yields:
        tuple: int index, dict of parameter levels, dict of parameter values
    """
    constellations = list(product(levels, levels, levels))

    index = 0

    for variable_parameter in parameter_info.keys():

        others = [p for p in parameters
                  if p != variable_parameter]

        for other_constellation in constellations:

            for variable_value in parameter_info[variable_parameter]['variable']:

                level_set = dict()
                level_set[variable_parameter] = 'variable'
                parameter_set = dict(t0=[0])
                parameter_set[variable_parameter] = [variable_value]

                for o, other in enumerate(others):
                    level_set[other] = other_constellation[o]
                    parameter_set[other] = [
                        parameter_info[other][other_constellation[o]]]

                index += 1
                yield index, level_set, parameter_set


def generate_hierarchical_parameter_sets(parameter_info, levels=['low', 'medium', 'high'], parameters=['v', 'gamma', 's', 'tau']):
    """
    Generates parameter sets for hierarchical simulation.

    Args:
        parameter_info (dict): dict containing information about parameter levels
        levels (list, optional): list of level labels
        parameters (list, optional): list of parameter names
    Yields:
        tuple: int index, dict of parameter levels, dict of population parameters
    """
    all_parameter_levels = list(product(levels, levels, levels, levels))

    for index, parameter_levels in enumerate(all_parameter_levels):
        parameter_set = dict()
        level_set = dict()
        for p, parameter in enumerate(parameters):
            level_set[parameter] = parameter_levels[p]
            parameter_set[parameter] = dict(mu=parameter_info[parameter]['mu'][level_set[parameter]],
                                            sd=parameter_info[parameter]['sd'],
                                            bounds=parameter_info[parameter]['bounds'])
        yield index, level_set, parameter_set


def recover_and_save(generated_input, output_folder=None, label=None,
                     simulate_group_args=None, make_model_args=None, fit_args=None):
    """
    Recovers a single GLAM using `recover_glam`
    using input from `generate_parameter_sets`
    and saves the output.
    """
    index, level_set, parameter_set = generated_input

    # check if already done:
    filename = join(
        output_folder, 'parameter-recovery_{}_part{}.csv'.format(label, index))
    if isfile(filename):
        print("Found previous result for index {}. Skipping...".format(index))
        return
    else:
        # perform actual recovery
        result = recover_glam(parameter_set,
                              simulate_group_args=simulate_group_args,
                              make_model_args=make_model_args,
                              fit_args=fit_args)

        # check for convergence:
        if fit_args['method'] == 'VI':
            all_converged = np.nan
        else:
            all_converged = summary(result.trace[0], varnames=['v', 'gamma', 's', 'tau'])[
                'Rhat'].between(0.95, 1.05).all()

        # assemble output dataframe row
        output = pd.DataFrame(dict(index=[index],
                                   v_level=[level_set['v']],
                                   gamma_level=[level_set['gamma']],
                                   s_level=[level_set['s']],
                                   tau_level=[level_set['tau']],
                                   v_gen=parameter_set['v'],
                                   gamma_gen=parameter_set['gamma'],
                                   s_gen=parameter_set['s'],
                                   tau_gen=parameter_set['tau'],
                                   v_rec=result.estimates['v'][0],
                                   gamma_rec=result.estimates['gamma'][0],
                                   s_rec=result.estimates['s'][0],
                                   tau_rec=result.estimates['tau'][0],
                                   converged=[all_converged]))

        output.to_csv(filename,
                      index=False)

        return output


def recover_and_save_hierarchical(generated_input, output_folder=None, label=None,
                                  simulate_group_args=None, make_model_args=None, fit_args=None,
                                  save_trace_to=None, save_traceplot_to=None):
    """
    Recovers a single GLAM using `recover_glam`
    using input from `generate_hierarchical_parameter_sets`
    and saves the output.
    """
    index, level_set, parameter_set = generated_input

    # check if already done:
    filename = join(
        output_folder, 'parameter-recovery_{}_part{}.csv'.format(label, index))
    if isfile(filename):
        print("Found previous result for index {}. Skipping...".format(index))
        return
    else:
        # perform actual recovery
        result = recover_glam(parameter_set,
                              simulate_group_args=simulate_group_args,
                              make_model_args=make_model_args,
                              fit_args=fit_args)
        # save trace
        if save_trace_to is not None:
            trace_to_dataframe(result.trace[0]).to_csv(join(save_trace_to, 'parameter-recovery_hierarchical_{}_trace_{}.csv'.format(label, index)))
        if save_traceplot_to is not None:
            traceplot(result.trace[0])
            plt.savefig(join(save_traceplot_to, 'parameter-recovery_hierarchical_{}_traceplot_{}.png'.format(label, index)))
            plt.close()

        # check for convergence:
        if fit_args['method'] == 'VI':
            all_converged = np.nan
        else:
            all_converged = summary(result.trace[0], varnames=['v', 'gamma', 's', 'tau'])[
                'Rhat'].between(0.95, 1.05).all()

        output = pd.DataFrame(dict(index=[index],
                                   v_level=[level_set['v']],
                                   gamma_level=[level_set['gamma']],
                                   s_level=[level_set['s']],
                                   tau_level=[level_set['tau']],
                                   v_gen=parameter_set['v']['mu'],
                                   v_gen_actual=result.parameters['condition_0']['v'].mean(),
                                   gamma_gen=parameter_set['gamma']['mu'],
                                   gamma_gen_actual=result.parameters['condition_0']['gamma'].mean(),
                                   s_gen=parameter_set['s']['mu'],
                                   s_gen_actual=result.parameters['condition_0']['s'].mean(),
                                   tau_gen=parameter_set['tau']['mu'],
                                   tau_gen_actual=result.parameters['condition_0']['tau'].mean(),
                                   v_rec=result.estimates['v_mu'][0],
                                   v_rec_hpd_lower=result.estimates['v_mu_hpd_2.5'][0],
                                   v_rec_hpd_upper=result.estimates['v_mu_hpd_97.5'][0],
                                   gamma_rec=result.estimates['gamma_mu'][0],
                                   gamma_rec_hpd_lower=result.estimates['gamma_mu_hpd_2.5'][0],
                                   gamma_rec_hpd_upper=result.estimates['gamma_mu_hpd_97.5'][0],
                                   s_rec=result.estimates['s_mu'][0],
                                   s_rec_hpd_lower=result.estimates['s_mu_hpd_2.5'][0],
                                   s_rec_hpd_upper=result.estimates['s_mu_hpd_97.5'][0],
                                   tau_rec=result.estimates['tau_mu'][0],
                                   tau_rec_hpd_lower=result.estimates['tau_mu_hpd_2.5'][0],
                                   tau_rec_hpd_upper=result.estimates['tau_mu_hpd_97.5'][0],
                                   converged=[all_converged]))

        output.to_csv(filename,
                      index=False)

        return output
