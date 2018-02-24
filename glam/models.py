#!/usr/bin/python

import glam
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import pandas as pd


class GLAM(object):
    """
    GLAM model instance that includes
    data, pymc3.model.Model instance,
    trace, parameter estimates,
    fit indices and predictions.
    """
    def __init__(self, data=None):
        self.data = data
        if self.data is not None:
            self.n_items = len([col for col in data.columns
                                if col.startswith('item_value_')])
        else:
            self.n_items = None

        self.parameters = dict()

    def simulate_group(self, kind='hierarchical',
                       n_individuals=10, n_trials=100, n_items=3,
                       individual_idx=None,
                       stimuli=None,
                       parameters=None,
                       error_weight=0.05,
                       error_range=(0, 5000),
                       value_range=(0, 10),
                       label=None,
                       seed=None):
        """
        Simulate data from known parameters.
        Data is added to any existing attached dataframes,
        which allows blockwise simulation of multiple groups.

        Parameters:
        -----------
        kind should be one of ['individual', 'hierarchical']
            defaults to 'hierarchical'
        n_individuals: integer, number of individuals to be added for this condition
        n_trials: integer, number of trials per individual
        n_items: integer, number of items per trial
        individual_idx: array of integers
            individual indices to use (for within individual design simulation)
            defaults to continuous participant numbering across conditions
        stimuli: pandas DataFrame
            Instead of simulating item_value and gaze data for the given number
            of individuals and trials, a DataFrame containing item_values, gaze_data
            and participant indices can be supplied.
            This overrides n_individuals and n_trials arguments.
        parameters: dict with keys: 'v', 'gamma', 's', 't0', 'tau'
            if kind is individual:
                values: arrays of length n_individual
            if kind is hierarchical:
                values: dicts with keys: mu, sd, bounds
                                   values: floats for mu, sd, tuple for bounds
        error_weight: float in (0, 1)
            Probability of simulating error trial with random chocie and uniform RT
        error_range: int tuple of length 2
            Range of error RTs
        value_range: tuple of length 2
            Range of item value ratings to be simulated
        label: string
            condition label. defaults "to condition_n"
        seed: integer, np.random.seed(argument)
        """

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # n_items
        self.n_items = n_items

        # Create condition label, if none given
        if label is None:
            if self.data is not None:
                if 'condition' in self.data.columns:
                    n_existing_conditions = len(self.data['condition'].unique())
                    label = 'condition_{}'.format(n_existing_conditions)
                else:
                    self.data['condition'] = 'condition_0'
                    label = 'condition_1'
            else:
                label = 'condition_0'

        # Create individual_idx if none given
        if individual_idx is None:
            if self.data is not None:
                n_existing_individuals = self.data['subject'].max() + 1
            else:
                n_existing_individuals = 0
            individual_idx = np.arange(n_existing_individuals, n_existing_individuals+n_individuals)

        # Set up parameters
        if kind is 'hierarchical':
            default_parameters = dict(v=dict(mu=0.00007, sd=0.00001, bounds=(0.00003, 0.00015)),
                                      s=dict(mu=0.008, sd=0.001, bounds=(0.005, 0.011)),
                                      gamma=dict(mu=0.3, sd=0.3, bounds=(-1.0, 1.0)),
                                      tau=dict(mu=0.8, sd=0.1, bounds=(0.2, 2)),
                                      t0=dict(mu=0, sd=0, bounds=(0.0, 0.0))  # results in no t0
                                      )

            parameters_formatted = dict()
            for p in ['v', 's', 'gamma', 'tau', 't0']:
                if parameters is not None:
                    default_parameters[p].update(parameters.get(p, default_parameters[p]))

                tmp = np.random.normal(loc=default_parameters[p]['mu'],
                                       scale=default_parameters[p]['sd'],
                                       size=n_individuals)
                tmp[tmp < default_parameters[p]['bounds'][0]] = default_parameters[p]['bounds'][0]
                tmp[tmp > default_parameters[p]['bounds'][1]] = default_parameters[p]['bounds'][1]
                parameters_formatted[p] = tmp
        elif kind is 'individual':
            parameters_formatted = parameters

        # Save generating parameters
        self.parameters[label] = parameters_formatted

        if stimuli is None:
            data = pd.DataFrame()
            for i, idx in enumerate(individual_idx):
                parameters_individual = [parameters_formatted[p][i]
                                         for p in ['v', 'gamma', 's', 'tau', 't0']]
                # No stimuli supplied, so we generate item_value and gaze data
                values = np.random.randint(value_range[0], value_range[1],
                                           size=(n_trials, n_items))
                gaze = np.random.uniform(0, 1,
                                         size=(n_trials, n_items))
                gaze = gaze / gaze.sum(axis=1, keepdims=True)

                individual_data = glam.simulation.simulate_subject(parameters_individual,
                                                                   values,
                                                                   gaze,
                                                                   n_repeats=1,
                                                                   subject=idx,
                                                                   boundary=1,
                                                                   error_weight=error_weight,
                                                                   error_range=error_range)
                data = pd.concat([data, individual_data])

            data['condition'] = label
            if self.data is None:
                self.data = data
            else:
                self.data = pd.concat([self.data, data]).reset_index(drop=True)

        else:
            print('Not implemented yet. Please run again with "stimuli=None".')
            return

    def make_model(self, kind, depends_on=dict(v=None, gamma=None, s=None, tau=None, t0=None), **kwargs):
        self.type = kind
        self.depends_on = depends_on
        self.design = glam.utils.get_design(self)
        self.model = make_models(df=self.data, kind=kind, design=self.design, **kwargs)

    def fit(self, method='NUTS', **kwargs):
        self.trace = glam.fit.fit_models(self.model, method=method, **kwargs)
        self.estimates = glam.utils.get_estimates(self)

    def compute_dic(self):
        if not isinstance(self.model, list):
            self.dic = pm.dic(trace=self.trace, model=self.model)
        else:
            self.dic = np.array([pm.dic(trace=trace, model=model)
                                 for (trace, model) in zip(self.trace, self.model)])

    def predict(self, n_repeats=1, boundary=1.0, error_weight=0.05, verbose=True):
        self.prediction = glam.simulation.predict(self,
                                                  n_repeats=n_repeats,
                                                  boundary=boundary,
                                                  error_weight=error_weight,
                                                  verbose=verbose)

    def exchange_data(self, new_data, verbose=True):
        if verbose:
            print('Replaced attached data ({} trials) with new data ({} trials)...'.format(len(self.data), len(new_data)))
        self.data = new_data


def make_models(df, kind, verbose=True, design=dict(v=None, gamma=None, s=None, tau=None, t0=None), **kwargs):

    if kind == 'individual':
        data = glam.utils.format_data(df)
        if verbose:
            print('Generating single subject models for {} subjects...'.format(data['n_subjects']))
        models = []
        for s, subject in enumerate(data['subjects']):
            design_subject = dict()
            for parameter in ['v', 'gamma', 's', 'tau', 't0']:
              design_subject[parameter] = dict()
              design_subject[parameter]['conditions'] = design[parameter]['conditions']
              design_subject[parameter]['condition_index'] = design[parameter]['condition_index'][data['subject_idx'] == subject]
            subject_model = make_subject_model(rts=data['rts'][data['subject_idx'] == subject],
                                               gaze=data['gaze'][data['subject_idx'] == subject],
                                               values=data['values'][data['subject_idx'] == subject],
                                               error_ll=data['error_lls'][s],
                                               design=design_subject,
                                               **kwargs)
            models.append(subject_model)
        return models

    elif kind == 'pooled':
        if verbose:
            print('Generating pooled model including {} trials...'.format(len(df)))
        pooled = df.copy()
        pooled['subject'] = 0
        data = glam.utils.format_data(pooled)
        pooled_model = make_subject_model(rts=data['rts'],
                                          gaze=data['gaze'],
                                          values=data['values'],
                                          error_ll=data['error_lls'][0],
                                          design=design,
                                          **kwargs)
        return pooled_model

    elif kind == 'hierarchical':
        data = glam.utils.format_data(df)
        if verbose:
            print('Generating hierarchical model for {} subjects...'.format(data['n_subjects']))
        hierarchical_model = make_hierarchical_model(rts=data['rts'],
                                                     gaze=data['gaze'],
                                                     values=data['values'],
                                                     error_lls=data['error_lls'],
                                                     subject_idx=data['subject_idx'],
                                                     design=design,
                                                     **kwargs)
        return hierarchical_model

    else:
        raise ValueError("'{}' model not Recognized. Use 'individual', 'pooled' or 'hierarchical'.".format(kind))


def generate_subject_model_parameters(parameter,
                                      design,
                                      lower, upper,
                                      val, testval):

    if design['conditions'] is not None:
        if val is None:
          parms = []
          for c, condition in enumerate(design['conditions']):
            if len(np.unique(design['condition_index'])) == 1:
              if c == np.unique(design['condition_index']):
                parms.append(pm.Uniform('{}'.format(parameter),
                                        lower,
                                        upper,
                                        testval=testval,
                                        shape=(1, 1)))
              else:
                parms.append(tt.zeros((1,1)))
            else:
              parms.append(pm.Uniform('{}_{}'.format(parameter, condition),
                                      lower,
                                      upper,
                                      testval=testval,
                                      shape=(1, 1)))
          parms = tt.concatenate(parms, axis=1)
        else:
            if len(val) != len(design['conditions']):
                raise ValueError('Number of values in {}_val does not match the number of conditions.'.format(parameter))
            else:
                parms = tt.stack([pm.Deterministic('{}_{}'.format(parameter, condition),
                                                   tt.ones((1, 1))*v)
                                  for condition, v in zip(design['{}_conditions'.format(parameter)], val)])
    else:
        if val is None:
            parms = pm.Uniform(parameter, lower, upper, testval=testval, shape=(1, 1))
        else:
            parms = pm.Deterministic(parameter, tt.ones((1, 1))*val)

    return parms


def make_subject_model(rts, gaze, values, error_ll,
                       v_val=None,
                       gamma_val=None,
                       s_val=None,
                       tau_val=None,
                       t0_val=None,
                       zerotol=1e-6, error_weight=0.05, boundary=1.,
                       gamma_bounds=(-1, 1),
                       design=dict(v=dict(), gamma=dict(), s=dict(), tau=dict(), t0=dict())):
    with pm.Model() as glam_individual:

        # Mechanics
        b = pm.Deterministic('b', tt.constant(boundary, dtype='float32'))
        p_error = pm.Deterministic('p_error', tt.constant(error_weight, dtype='float32'))

        v = generate_subject_model_parameters(parameter='v',
                                              design=design['v'],
                                              lower=zerotol, upper=0.01,
                                              val=v_val, testval=0.0002)

        gamma = generate_subject_model_parameters(parameter='gamma',
                                                  design=design['gamma'],
                                                  lower=gamma_bounds[0], upper=gamma_bounds[1],
                                                  val=gamma_val, testval=0)

        s = generate_subject_model_parameters(parameter='s',
                                              design=design['s'],
                                              lower=zerotol, upper=0.02,
                                              val=s_val, testval=0.0075)

        tau = generate_subject_model_parameters(parameter='tau',
                                                design=design['tau'],
                                                lower=0, upper=5,
                                                val=tau_val, testval=1)

        t0 = generate_subject_model_parameters(parameter='t0',
                                               design=design['t0'],
                                               lower=0, upper=500,
                                               val=t0_val, testval=1)

        # Likelihood
        def lda_logp(rt,
                     gaze, values,
                     error_ll,
                     v_index,
                     tau_index,
                     gamma_index,
                     s_index,
                     t0_index,
                     zerotol):

            # compute drifts
            drift = glam.components.expdrift(v[0, tt.cast(v_index, dtype='int32')][:, None],
                                             tau[0, tt.cast(tau_index, dtype='int32')][:, None],
                                             gamma[0, tt.cast(gamma_index, dtype='int32')][:, None],
                                             values,
                                             gaze,
                                             zerotol)
            glam_ll = glam.components.tt_wienerrace_pdf(rt[:, None],
                                                        drift,
                                                        s[0, tt.cast(s_index, dtype='int32')][:, None],
                                                        b,
                                                        t0[0, tt.cast(t0_index, dtype='int32')][:, None],
                                                        zerotol)

            # mix likelihoods
            mixed_ll = ((1-p_error) * glam_ll + p_error * error_ll)

            mixed_ll = tt.where(tt.isnan(mixed_ll), 0., mixed_ll)
            mixed_ll = tt.where(tt.isinf(mixed_ll), 0., mixed_ll)
            return tt.sum(tt.log(mixed_ll + zerotol))

        obs = pm.DensityDist('obs', logp=lda_logp,
                             observed=dict(rt=rts,
                                           gaze=gaze,
                                           values=values,
                                           error_ll=error_ll,
                                           v_index=design['v']['condition_index'].astype(np.int32),
                                           tau_index=design['tau']['condition_index'].astype(np.int32),
                                           gamma_index=design['gamma']['condition_index'].astype(np.int32),
                                           s_index=design['s']['condition_index'].astype(np.int32),
                                           t0_index=design['t0']['condition_index'].astype(np.int32),
                                           zerotol=zerotol))
    return glam_individual


def generate_hierarchical_model_parameters(parameter,
                                           n_subjects,
                                           design,
                                           mu_lower, mu_upper,
                                           sd_lower, sd_upper,
                                           bound_lower, bound_upper,
                                           val, testval):

    if (design['conditions'] is not None):
        if val is None:
            mu = tt.stack([pm.Uniform('{}_{}_mu'.format(parameter, condition),
                                      mu_lower,
                                      mu_upper,
                                      testval=testval) for condition in design['conditions']])
            sd = tt.stack([pm.Uniform('{}_{}_sd'.format(parameter, condition),
                                      sd_lower,
                                      sd_upper,
                                      testval=testval) for condition in design['conditions']])
            bounded = pm.Bound(pm.Normal, bound_lower, bound_upper)
            parms = []
            n_subjects_per_condition = []
            for c, condition in enumerate(design['conditions']):
                n_subjects_in_condition = np.unique(design['subject_index'][design['condition_index'] == c]).size
                n_subjects_per_condition.append(n_subjects_in_condition)
                parms_tmp = bounded('{}_{}'.format(parameter, condition),
                                    mu=mu[c],
                                    sd=sd[c],
                                    shape=(n_subjects_in_condition))
                parms_tmp = tt.concatenate([tt.zeros(1), parms_tmp])
                parms.append(parms_tmp[design['D'][:, c]][:, None])
            parms = tt.concatenate(parms, axis=1)
        
        else:
            parms = []
            n_subjects_per_condition = []
            for c, condition in enumerate(design['conditions']):
                n_subjects_in_condition = np.unique(design['subject_index'][design['condition_index'] == c]).size
                n_subjects_per_condition.append(n_subjects_in_condition)
                if len(val) == len(design['conditions']):
                    parms.append(pm.Deterministic('{}_{}'.format(parameter, condition),
                                                  tt.ones(n_subjects_in_condition, 1) * val[c]))
                else:
                    raise ValueError('Number of values in {}_val does not match the number of specified {}-conditions.'.format(parameter, parameter))               
            # make sure all elements in parms have same size
            for set_i, parm_set in enumerate(parms):
                if n_subjects_per_condition[set_i] < n_subjects:
                    parms[set_i] = tt.concatenate([parm_set,
                                                   tt.zeros((n_subjects-n_subjects_per_condition[set_i], 1))],
                                                  axis=0)
            parms = tt.concatenate(parms, axis=1)

    else:
        if val is None:
            mu = pm.Uniform('{}_mu'.format(parameter), mu_lower, mu_upper, testval=testval)
            sd = pm.Uniform('{}_sd'.format(parameter), sd_lower, sd_upper, testval=testval)
            bounded = pm.Bound(pm.Normal, bound_lower, bound_upper)
            parms = bounded(parameter, mu=mu, sd=sd, shape=(n_subjects, 1))
        else:
            parms = pm.Deterministic(parameter, tt.ones((n_subjects, 1)) * val)

    return parms


def make_hierarchical_model(rts, gaze, values, error_lls,
                            subject_idx,
                            v_val=None,
                            gamma_val=None,
                            s_val=None,
                            tau_val=None,
                            t0_val=None,
                            zerotol=1e-6, error_weight=0.05, boundary=1.,
                            gamma_bounds=(-1, 1),
                            design=dict(v=dict(), gamma=dict(), s=dict(), tau=dict(), t0=dict())):

    n_subjects = np.unique(subject_idx).size

    with pm.Model() as glam_hierarchical:

        # Mechanics
        b = pm.Deterministic('b', tt.constant(boundary, dtype='float32'))
        p_error = pm.Deterministic('p_error', tt.constant(error_weight, dtype='float32'))

        # Parameter priors
        v = generate_hierarchical_model_parameters(parameter='v',
                                                   n_subjects=n_subjects,
                                                   design=design['v'],
                                                   mu_lower=zerotol, mu_upper=0.0005,
                                                   sd_lower=zerotol, sd_upper=0.0005,
                                                   bound_lower=0, bound_upper=0.0005,
                                                   val=v_val, testval=0.0001)

        gamma = generate_hierarchical_model_parameters(parameter='gamma',
                                                       n_subjects=n_subjects,
                                                       design=design['gamma'],
                                                       mu_lower=gamma_bounds[0], mu_upper=gamma_bounds[1],
                                                       sd_lower=zerotol, sd_upper=gamma_bounds[1] - gamma_bounds[0],
                                                       bound_lower=gamma_bounds[0], bound_upper=gamma_bounds[1],
                                                       val=gamma_val, testval=.5)

        s = generate_hierarchical_model_parameters(parameter='s',
                                                   n_subjects=n_subjects,
                                                   design=design['s'],
                                                   mu_lower=zerotol, mu_upper=0.02,
                                                   sd_lower=zerotol, sd_upper=0.02,
                                                   bound_lower=zerotol, bound_upper=0.02,
                                                   val=s_val, testval=0.0075)

        tau = generate_hierarchical_model_parameters(parameter='tau',
                                                     n_subjects=n_subjects,
                                                     design=design['tau'],
                                                     mu_lower=0, mu_upper=5,
                                                     sd_lower=zerotol, sd_upper=5,
                                                     bound_lower=0, bound_upper=5,
                                                     val=tau_val, testval=.5)

        if t0_val is None:
            t0 = pm.Uniform('t0', 0, 500, testval=50, shape=(n_subjects, 1))
        else:
            t0 = pm.Deterministic('t0', tt.ones((n_subjects, 1)) * t0_val)

        # Likelihood
        def lda_logp(rt,
                     gaze, values,
                     error_lls,
                     s_condition_index,
                     s_subject_index,
                     v_condition_index,
                     v_subject_index,
                     tau_condition_index,
                     tau_subject_index,
                     gamma_condition_index,
                     gamma_subject_index,
                     t0_condition_index,
                     t0_subject_index,
                     zerotol):

            # compute drifts
            drift = glam.components.expdrift(v[tt.cast(v_subject_index, dtype='int32'),
                                               tt.cast(v_condition_index, dtype='int32')][:, None],
                                             tau[tt.cast(tau_subject_index, dtype='int32'),
                                                 tt.cast(tau_condition_index, dtype='int32')][:, None],
                                             gamma[tt.cast(gamma_subject_index, dtype='int32'),
                                                   tt.cast(gamma_condition_index, dtype='int32')][:, None],
                                             values, gaze, zerotol)
            glam_ll = glam.components.tt_wienerrace_pdf(rt[:, None],
                                                        drift,
                                                        s[tt.cast(s_subject_index, dtype='int32'),
                                                          tt.cast(s_condition_index, dtype='int32')][:, None],
                                                        b,
                                                        t0[tt.cast(t0_subject_index, dtype='int32'),
                                                          tt.cast(t0_condition_index, dtype='int32')][:, None],
                                                        zerotol)

            # mix likelihoods
            mixed_ll = ((1-p_error) * glam_ll + p_error * error_lls[subject_idx])

            mixed_ll = tt.where(tt.isnan(mixed_ll), 0., mixed_ll)
            mixed_ll = tt.where(tt.isinf(mixed_ll), 0., mixed_ll)
            return tt.sum(tt.log(mixed_ll + zerotol))

        obs = pm.DensityDist('obs', logp=lda_logp,
                             observed=dict(rt=rts,
                                           gaze=gaze,
                                           values=values,
                                           error_lls=error_lls,
                                           s_condition_index=design['s']['condition_index'].astype(np.int32),
                                           s_subject_index=design['s']['subject_index'].astype(np.int32),
                                           v_condition_index=design['v']['condition_index'].astype(np.int32),
                                           v_subject_index=design['v']['subject_index'].astype(np.int32),
                                           tau_condition_index=design['tau']['condition_index'].astype(np.int32),
                                           tau_subject_index=design['tau']['subject_index'].astype(np.int32),
                                           gamma_condition_index=design['gamma']['condition_index'].astype(np.int32),
                                           gamma_subject_index=design['gamma']['subject_index'].astype(np.int32),
                                           t0_condition_index=design['t0']['condition_index'].astype(np.int32),
                                           t0_subject_index=design['t0']['subject_index'].astype(np.int32),
                                           zerotol=zerotol))
    return glam_hierarchical
