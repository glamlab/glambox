#!/usr/bin/python
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import pandas as pd

from .simulation import simulate_subject, predict
from .utils import get_design, get_estimates, format_data
from .fit import fit_models
from .components import *

class GLAM(object):
    """GLAM model instance that includes data, pymc3.model.Model instance,
    trace, parameter estimates, fit indices and predictions.
    
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
    
    name : str
        A name for the model. Useful if multiple models are fitted and compared.
    """
    def __init__(self, data=None, name=None):
        self.data = data
        if self.data is not None:
            self.n_items = len(
                [col for col in data.columns if col.startswith('item_value_')])
        else:
            self.n_items = None
        self.name = name

        self.parameters = dict()

    def simulate_group(self,
                       kind='hierarchical',
                       n_individuals=10,
                       n_trials=100,
                       n_items=3,
                       individual_idx=None,
                       stimuli=None,
                       parameters=None,
                       error_weight=0.05,
                       error_range=(0, 5),
                       value_range=(0, 10),
                       label=None,
                       seed=None):
        """
        Simulate data from known parameters.
        Data is added to any existing attached dataframes,
        which allows blockwise simulation of multiple groups.

        Parameters
        ----------
        kind : str, optional
            Should be one of ['individual', 'hierarchical', 'pooled'],
            defaults to 'hierarchical'
        
        n_individuals : int, optional
            number of individuals to be added for this condition
        
        n_trials : int, optional
            number of trials per individual
        
        n_items : int, optional
            number of items per trial
        
        individual_idx : array of ints, optional
            individual indices to use (for within individual design simulation)
            defaults to continuous participant numbering across conditions
        
        stimuli : DataFrame, optional
            instead of simulating item_value and gaze data for the given number
            of individuals and trials, a DataFrame containing item_values, gaze_data
            and participant indices can be supplied.
            This overrides n_individuals and n_trials arguments.
        
        parameters : dict, optional
            dict with keys: 'v', 'gamma', 's', 't0', 'tau'
            if kind is individual:
                values: arrays of length n_individual
            if kind is hierarchical:
                values: dicts with keys: mu, sd, bounds
                                   values: floats for mu, sd, tuple for bounds
        
        error_weight : float, optional
            range: [0, 1],
            probability of simulating error trial
            with random choice and uniform RT
        
        error_range : int tuple of length 2, optional
            range of error RTs
        
        value_range : tuple of length 2, optional
            range of item value ratings to be simulated
        
        label : str, optional
            condition label. defaults "to condition_n"
        
        seed : int, optional
            np.random.seed(argument)

        Returns 
        -------
        None 
            Adds `data` to GLAM model instance
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
                    n_existing_conditions = len(
                        self.data['condition'].unique())
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
            individual_idx = np.arange(n_existing_individuals,
                                       n_existing_individuals + n_individuals)

        # Set up parameters
        if kind == 'hierarchical':
            default_parameters = dict(
                v=dict(mu=0.6, sd=0.25, bounds=(0, 2)),
                s=dict(mu=0.25, sd=0.05, bounds=(0, 1)),
                gamma=dict(mu=0.3, sd=0.3, bounds=(-1.0, 1.0)),
                tau=dict(mu=1, sd=0.2, bounds=(0.2, 2)),
                t0=dict(mu=0, sd=0, bounds=(0.0, 0.0)))  # results in no t0

            parameters_formatted = dict()
            for p in ['v', 's', 'gamma', 'tau', 't0']:
                if parameters is not None:
                    default_parameters[p].update(
                        parameters.get(p, default_parameters[p]))

                tmp = np.random.normal(loc=default_parameters[p]['mu'],
                                       scale=default_parameters[p]['sd'],
                                       size=n_individuals)
                tmp[tmp < default_parameters[p]['bounds']
                    [0]] = default_parameters[p]['bounds'][0]
                tmp[tmp > default_parameters[p]['bounds']
                    [1]] = default_parameters[p]['bounds'][1]
                parameters_formatted[p] = tmp
        elif kind == 'individual':
            parameters_formatted = parameters
        else:
            raise ValueError(
                "'kind' must be 'individual' or 'hierarchical' (is '{}')".
                format(kind))

        # Save generating parameters
        self.parameters[label] = parameters_formatted

        if stimuli is None:
            data = pd.DataFrame()
            for i, idx in enumerate(individual_idx):
                parameters_individual = [
                    parameters_formatted[p][i]
                    for p in ['v', 'gamma', 's', 'tau', 't0']
                ]
                # No stimuli supplied, so we generate item_value and gaze data
                values = np.random.randint(value_range[0],
                                           value_range[1],
                                           size=(n_trials, n_items))
                gaze = np.random.uniform(0, 1, size=(n_trials, n_items))
                gaze = gaze / gaze.sum(axis=1, keepdims=True)

                individual_data = simulate_subject(
                    parameters_individual,
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

    def make_model(self,
                   kind='individual',
                   depends_on=dict(v=None,
                                   gamma=None,
                                   s=None,
                                   tau=None,
                                   t0=None),
                   within_dependent=[],
                   **kwargs):
        """
        Build the GLAM PyMC3 model, specifying model kind and dependencies.

        Parameters
        ----------
        kind : str, optional
            should be one of ['individual', 'hierarchical', 'pooled'],
            defaults to 'individual'

        depends_on : dict, optional
            dictionary specifying for each GLAM model parameter
            whether the parameter is dependent on any levels
            of the data
            e.g. {'v': 'speed'}, here one v parameter is created 
            for each level of the 'speed' factor in the response data
            (factor must be encoded in data)

        within_dependent : list, optional
            list of parameter names ('v', 'gamma', 's', 'tau')
            each included parameter is modeled as 
            dependent within a subject (i.e., as drawn
            from the same meta-distribution)
            only, if parameter dependency-structure specified
            in depends_on

        Returns
        -------
        None
            Adds PyMC3 `model`, `depends_on`, `within_dependent` and `design` 
            to GLAM model object
        """
        self.type = kind
        self.depends_on = depends_on
        self.within_dependent = within_dependent
        self.design = get_design(self)
        self.model = make_models(df=self.data,
                                 kind=kind,
                                 design=self.design,
                                 **kwargs)
        if kind == 'hierarchical':
            self.model.name = self.name
        else:
            for model in self.model:
                model.name = self.name

    def fit(self, method='MCMC', **kwargs):
        """
        Perform parameter estimation of the model.

        Parameters
        ----------
        method : string ['MCMC', 'VI'], optional
            specifies fitting method to use,
            can be either 'MCMC' for MCMC-sampling
            or 'VI' for variational inference.
            Defaults to 'MCMC'.

        Returns
        -------
        None
            Adds `trace` and well as `estimates`
            to GLAM model object
        """
        self.trace = fit_models(self.model, method=method, **kwargs)
        self.estimates = get_estimates(self)

    def compute_waic(self):
        """
        Compute WAIC for all appended
        PyMC3 models.

        Returns
        -------
        None
            Adds WAIC to GLAM model object.
        """
        if not isinstance(self.model, list):
            self.waic = pm.waic(trace=self.trace, model=self.model)
        else:
            self.waic = np.array([
                pm.waic(trace=trace, model=model)
                for (trace, model) in zip(self.trace, self.model)
            ])

    def predict(self,
                n_repeats=1,
                boundary=1.0,
                error_weight=0.05,
                verbose=True):
        """
        Predict choices and RTs for included data

        Parameters
        ----------
        n_repeats : int, optional
            Number of repeats of each trial
            included in data during prediction
            Defaults to 1.

        boundary : float, optional
            Magnitude of decision boundary.
            Defaults to 1.0

        error_weight : float, optional
            float between [0, 1],
            defaults to 0.05
            determining probability that choice and
            RT are drawn according to a unifrom
            error distribution
            (see manuscript)

        verbose : bool, optional
            Toggle verbosity.
            
        Returns
        -------
        None
            Adds `predictions` to GLAM model object
        """
        self.prediction = predict(self,
                                  n_repeats=n_repeats,
                                  boundary=boundary,
                                  error_weight=error_weight,
                                  verbose=verbose)

    def exchange_data(self, new_data, verbose=True):
        """
        Exchange GLAM model data.
        Useful for out-of-sample predictions using fitted parameters.

        Parameters
        ----------
        new_data : pandas.DataFrame
            new data to exchange old data with

        verbose : bool, optional
            Toggle verbosity.
            
        Returns
        -------
        None
            `new_data` replaces `data` attribute of GLAM model object
        """
        if verbose:
            print(
                'Replaced attached data ({} trials) with new data ({} trials)...'
                .format(len(self.data), len(new_data)))
        self.data = new_data


def make_models(df,
                kind,
                verbose=True,
                design=dict(v=None, gamma=None, s=None, tau=None, t0=None),
                **kwargs):
    """
    Create PyMC3 GLAM model instances,
    given a response dataset

    Parameters
    ----------
    df : pandas.DataFrame
        response data

    kind : str
        specifying the assumed parameter structure:
        'pooled':
            one single parameter set for all
            subjects
        'individual': 
            parameters are assumed to be drawn 
            independently for each subject
        'hierarchical' (resulting in )
            subject parameters are assumed
            to be drawn from shared
            group-level distributions

    verbose : bool, optional
        Toggle verbosity.

    design : dict, optional
        dict with one entry per model parameter,
        speciying the parameter's dependency structure
        (see utils.get_design)

    Returns
    -------
    PyMC3 GLAM model instances
    """

    if kind == 'individual':
        data = format_data(df)
        if verbose:
            print('Generating single subject models for {} subjects...'.format(
                data['n_subjects']))
        models = []
        for s, subject in enumerate(data['subjects']):
            design_subject = dict()
            for parameter in ['v', 'gamma', 's', 'tau', 't0']:
                design_subject[parameter] = design[parameter].copy()
                design_subject[parameter]['conditions'] = design[parameter][
                    'conditions']
                design_subject[parameter]['condition_index'] = design[
                    parameter]['condition_index'][data['subject_idx'] == subject]
            subject_model = make_subject_model(
                rts=data['rts'][data['subject_idx'] == subject],
                gaze=data['gaze'][data['subject_idx'] == subject],
                values=data['values'][data['subject_idx'] == subject],
                error_ll=data['error_lls'][s],
                design=design_subject,
                **kwargs)
            models.append(subject_model)
        return models

    elif kind == 'pooled':
        if verbose:
            print('Generating pooled model including {} trials...'.format(
                len(df)))
        pooled = df.copy()
        pooled['subject'] = 0
        data = format_data(pooled)
        pooled_model = make_subject_model(rts=data['rts'],
                                          gaze=data['gaze'],
                                          values=data['values'],
                                          error_ll=data['error_lls'][0],
                                          design=design,
                                          **kwargs)
        return pooled_model

    elif kind == 'hierarchical':
        data = format_data(df)
        if verbose:
            print('Generating hierarchical model for {} subjects...'.format(
                data['n_subjects']))
        hierarchical_model = make_hierarchical_model(
            rts=data['rts'],
            gaze=data['gaze'],
            values=data['values'],
            error_lls=data['error_lls'],
            subject_idx=data['subject_idx'],
            design=design,
            **kwargs)
        return hierarchical_model

    else:
        raise ValueError(
            "'{}' model not Recognized. Use 'individual', 'pooled' or 'hierarchical'."
            .format(kind))


def generate_subject_model_parameters(parameter,
                                      design,
                                      lower,
                                      upper,
                                      val,
                                      testval,
                                      within_dependent=False):
    """
    Generate a subject-level model parameter.

    Parameters
    ----------
    parameter : str
        one of ['v', 'gamma', 's', 'tau']

    design : dict
        dependency structure for parameter

    lower : float
        lower bound for parameter distribution

    upper : float
        upper bound for parameter distribution

    val : float
        if specified, parameter is model as a deterministic
        variable and set to the specified value

    testval : float
        PyMC3 testvalue for parameter

    within_dependent : bool, optional
        whether parameters with a dependency
        should be drawn from a shared meta-distribution (if True)
        or independently of one another (if False),
        defaults to False

    Returns
    -------
    PyMC3 parameter variable
    """

    if design['conditions'] is not None:
        if val is None:
            parms = []

            # if we want a meta distribution, we need to initialize meta mean and ds priors here.
            if within_dependent:
                bounded = pm.Bound(pm.Normal, lower, upper)
                meta_mu = pm.Uniform('{}_mu'.format(parameter), lower, upper)
                meta_sd = pm.Uniform('{}_sd'.format(parameter), 1e-10,
                                     upper - lower)

            for c, condition in enumerate(design['conditions']):
                if len(np.unique(design['condition_index'])) == 1:
                    if c == np.unique(design['condition_index']):
                        parms.append(
                            pm.Uniform('{}'.format(parameter),
                                       lower,
                                       upper,
                                       testval=testval,
                                       shape=(1, 1)))
                    else:
                        parms.append(tt.zeros((1, 1)))
                else:
                    # if we want meta distribution for conditions, now drawn parameters should come from meta distribution
                    if within_dependent:
                        parms.append(
                            bounded('{}_{}'.format(parameter, condition),
                                    mu=meta_mu,
                                    sd=meta_sd,
                                    shape=(1, 1)))
                    else:
                        parms.append(
                            pm.Uniform('{}_{}'.format(parameter, condition),
                                       lower,
                                       upper,
                                       testval=testval,
                                       shape=(1, 1)))
            parms = tt.concatenate(parms, axis=1)
        else:
            if len(val) != len(design['conditions']):
                raise ValueError(
                    'Number of values in {}_val does not match the number of conditions.'
                    .format(parameter))
            else:
                parms = tt.stack([
                    pm.Deterministic('{}_{}'.format(parameter, condition),
                                     tt.ones((1, 1)) * v)
                    for condition, v in zip(
                        design['{}_conditions'.format(parameter)], val)
                ])
    else:
        if val is None:
            parms = pm.Uniform(parameter,
                               lower,
                               upper,
                               testval=testval,
                               shape=(1, 1))
        else:
            parms = pm.Deterministic(parameter, tt.ones((1, 1)) * val)

    return parms


def make_subject_model(rts,
                       gaze,
                       values,
                       error_ll,
                       v_val=None,
                       gamma_val=None,
                       s_val=None,
                       tau_val=None,
                       t0_val=0,
                       zerotol=1e-6,
                       error_weight=0.05,
                       boundary=1.,
                       gamma_bounds=(-2, 1),
                       design=dict(v=dict(),
                                   gamma=dict(),
                                   s=dict(),
                                   tau=dict(),
                                   t0=dict())):
    """
    Create subject-level PyMC3 model

    Parameters
    ----------
    rts : array_like, int or float
        response times in seconds per trial

    gaze : array_like, float 
        gaze distribution,
        specifying the observed gaze for
        each choice alternative in each trial
        shape: (trials x alternatives)
        gaze values must be between [0,1]

    values : array_like, float
        value of each choice alternative
        in each trial; 
        shape: (trials x alternatives)

    error_ll : float
        likelihood of erroneous choice model, 
        between [0,1]

    [v_val, gamma_val, s_val, tau_val] : float, optional
        if specified, specified parameter is 
        deterministically set to the specified value

    t0_val : int, optional
        if specified, t0 value is determinstically
        set to this value, defaults to 0

    zerotol : float, optional
        tolerance for mathematical stability,
        defaults to 1e-6

    error_weight : float, optional
        probability with which choices are modeled
        as resulting from errornous choice model
        (as specified by error_ll),
        defaults to 0.05 (5%) 

    boundary : float, optional
        decision boundary for linear
        stochastic race, defaults to 1

    gamma_bounds : tuple of floats, optional
        bounds for gamma distribution,
        defaults to [-10, 1]

    design : dict, optional
        dict with one entry per model parameter,
        specifying the parameter's dependency structure
        (see utils.get_design)

    Returns
    -------
    PyMC3 model instance
    """
    with pm.Model() as glam_individual:

        # Mechanics
        b = pm.Deterministic('b', tt.constant(boundary, dtype='float32'))
        p_error = pm.Deterministic('p_error',
                                   tt.constant(error_weight, dtype='float32'))

        v = generate_subject_model_parameters(
            parameter='v',
            design=design['v'],
            lower=zerotol,
            upper=4,
            val=v_val,
            testval=0.5,
            within_dependent=(design['v']['within_dependent']))

        gamma = generate_subject_model_parameters(
            parameter='gamma',
            design=design['gamma'],
            lower=gamma_bounds[0],
            upper=gamma_bounds[1],
            val=gamma_val,
            testval=0,
            within_dependent=(design['gamma']['within_dependent']))

        s = generate_subject_model_parameters(
            parameter='s',
            design=design['s'],
            lower=zerotol,
            upper=4,
            val=s_val,
            testval=0.75,
            within_dependent=(design['s']['within_dependent']))

        tau = generate_subject_model_parameters(
            parameter='tau',
            design=design['tau'],
            lower=0,
            upper=10,
            val=tau_val,
            testval=1,
            within_dependent=(design['tau']['within_dependent']))

        t0 = generate_subject_model_parameters(
            parameter='t0',
            design=design['t0'],
            lower=0,
            upper=0.5,
            val=t0_val,
            testval=0.01,
            within_dependent=(design['t0']['within_dependent']))

        # Likelihood
        def lda_logp(rt, gaze, values, error_ll, v_index, tau_index,
                     gamma_index, s_index, t0_index, zerotol):

            # compute drifts
            R = make_R(
                v[0, tt.cast(v_index, dtype='int32')][:, None],
                tau[0, tt.cast(tau_index, dtype='int32')][:, None],
                gamma[0, tt.cast(gamma_index, dtype='int32')][:, None], values,
                gaze, zerotol)
            glam_ll = tt_wienerrace_pdf(
                rt[:, None], R,
                s[0, tt.cast(s_index, dtype='int32')][:, None], b,
                t0[0, tt.cast(t0_index, dtype='int32')][:, None], zerotol)

            # mix likelihoods
            mixed_ll = ((1 - p_error) * glam_ll + p_error * error_ll)

            mixed_ll = tt.where(tt.isnan(mixed_ll), 0., mixed_ll)
            mixed_ll = tt.where(tt.isinf(mixed_ll), 0., mixed_ll)
            return tt.log(mixed_ll + zerotol)

        obs = pm.DensityDist(
            'obs',
            logp=lda_logp,
            observed=dict(
                rt=rts,
                gaze=gaze,
                values=values,
                error_ll=error_ll,
                v_index=design['v']['condition_index'].astype(np.int32),
                tau_index=design['tau']['condition_index'].astype(np.int32),
                gamma_index=design['gamma']['condition_index'].astype(
                    np.int32),
                s_index=design['s']['condition_index'].astype(np.int32),
                t0_index=design['t0']['condition_index'].astype(np.int32),
                zerotol=zerotol))
    return glam_individual


def generate_hierarchical_model_parameters(parameter,
                                           n_subjects,
                                           design,
                                           mu_mean,
                                           mu_sd,
                                           mu_lower,
                                           mu_upper,
                                           sd_mean,
                                           sd_sd,
                                           sd_lower,
                                           sd_upper,
                                           val,
                                           offset=True,
                                           within_dependent=False):
    """
    Generate a hierarchical-level model parameter distribution

    Parameters
    ----------
    parameter : str
        one of ['v', 'gamma', 's', 'tau']

    n_subjects : int
        number of subject parameters drawn
        from hierarchical distribution

    design : dict
        dict with one entry per model parameter,
        specifying the parameter's dependency structure
        (see utils.get_design)

    [mu_mean, mu_sd] : float
        mean / standard deviation of
        the distribution for the hierarchical mean

    [mu_lower, mu_upper] : float
        lower / upper bound of the  
        distribution for the hierarchical mean

    [sd_mean, sd_sd] : float
        mean / standard deviation of
        the distribution for the hierarchical standard deviation

    [sd_lower, sd_upper] : float
        lower / upper bound of the 
        distribution for the hierarchical standard deviation

    val : float
        if specified, parameter is deterministically set 
        to this value for all subjects 

    offset : bool, optional
        if True, subject-level parameters are modeled as 
        offset from the group mean;
        if False, subject-level parameters are modeled as
        distributed around the group mean,
        defaults to True
        For further details, see: 
        https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/

    Returns
    -------
    PyMC3 parameter variable
    """

    if (design['conditions'] is not None):  # Parameter has dependence
        if val is None:  # Parameter is not set deterministically to some value
            if not within_dependent:  # The parameter's conditions have distributions from which subject parameters are drawn (e.g., a subject's fast v is drawn from the distribution of all subjects' fast vs)
                bounded_mu = pm.Bound(pm.Normal,
                                      lower=mu_lower,
                                      upper=mu_upper)
                bounded_sd = pm.Bound(pm.Normal,
                                      lower=sd_lower,
                                      upper=sd_upper)
                mu = tt.stack([
                    bounded_mu('{}_{}_mu'.format(parameter, condition),
                               mu_mean,
                               mu_sd,
                               testval=mu_mean)
                    for condition in design['conditions']
                ])
                sd = tt.stack([
                    bounded_sd('{}_{}_sd'.format(parameter, condition),
                               sd_mean,
                               sd_sd,
                               testval=sd_mean)
                    for condition in design['conditions']
                ])
                parms = []
                n_subjects_per_condition = []
                for c, condition in enumerate(design['conditions']):
                    n_subjects_in_condition = np.unique(
                        design['subject_index'][design['condition_index'] == c]).size
                    n_subjects_per_condition.append(n_subjects_in_condition)
                    if offset:
                        # Escape the Funnel of Hell (cf. https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/)
                        parms_tmp_offset = pm.Normal(
                            '{}_{}_offset'.format(parameter, condition),
                            mu=0,
                            sd=1,
                            shape=(n_subjects_in_condition))
                        parms_tmp = pm.Deterministic(
                            '{}_{}'.format(parameter, condition),
                            tt.clip(mu[c] + parms_tmp_offset * sd[c], mu_lower,
                                    mu_upper))
                    else:  # no offset
                        # Disregard Funnel of Hell
                        bounded_i = pm.Bound(pm.Normal,
                                             lower=mu_lower,
                                             upper=mu_upper)
                        parms_tmp = bounded_i('{}_{}'.format(parameter, condition),
                                              mu=mu[c],
                                              sd=sd[c],
                                              shape=(n_subjects_in_condition))
                    parms_tmp = tt.concatenate([tt.zeros(1), parms_tmp])
                    parms.append(parms_tmp[design['D'][:, c]][:, None])
                parms = tt.concatenate(parms, axis=1)

            else:  # within dependent: The parameter's values per condition are drawn from individual subject distributions (e.g., a subject's fast and slow v are both drawn from a subject's base v)
                raise ValueError('within dependent not implemented yet')
        else:
            parms = []
            n_subjects_per_condition = []
            for c, condition in enumerate(design['conditions']):
                n_subjects_in_condition = np.unique(design['subject_index'][
                    design['condition_index'] == c]).size
                n_subjects_per_condition.append(n_subjects_in_condition)
                if len(val) == len(design['conditions']):
                    parms.append(
                        pm.Deterministic(
                            '{}_{}'.format(parameter, condition),
                            tt.ones(n_subjects_in_condition, 1) * val[c]))
                else:
                    raise ValueError(
                        'Number of values in {}_val does not match the number of specified {}-conditions.'
                        .format(parameter, parameter))
            # make sure all elements in parms have same size
            for set_i, parm_set in enumerate(parms):
                if n_subjects_per_condition[set_i] < n_subjects:
                    parms[set_i] = tt.concatenate([parm_set,
                                                   tt.zeros((n_subjects - n_subjects_per_condition[set_i], 1))],
                                                  axis=0)
            parms = tt.concatenate(parms, axis=1)

    else:  # Parameter has no dependence
        if val is None:
            bounded_mu = pm.Bound(pm.Normal, lower=mu_lower, upper=mu_upper)
            bounded_sd = pm.Bound(pm.Normal, lower=sd_lower, upper=sd_upper)
            mu = bounded_mu('{}_mu'.format(parameter),
                            mu_mean,
                            mu_sd,
                            testval=mu_mean)
            sd = bounded_sd('{}_sd'.format(parameter),
                            sd_mean,
                            sd_sd,
                            testval=sd_mean)
            if offset:
                # Escape the Funnel of Hell (cf. https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/)
                parms_offset = pm.Normal(parameter + '_offset',
                                         mu=0,
                                         sd=1,
                                         shape=(n_subjects, 1))
                parms = pm.Deterministic(
                    parameter,
                    tt.clip(mu + parms_offset * sd, mu_lower, mu_upper))
            else:  # no offset
                # Disregard Funnel of Hell
                bounded = pm.Bound(pm.Normal, lower=mu_lower, upper=mu_upper)
                parms = bounded(parameter, mu=mu, sd=sd, shape=(n_subjects, 1))
        else:
            parms = pm.Deterministic(parameter, tt.ones((n_subjects, 1)) * val)

    return parms


def make_hierarchical_model(rts,
                            gaze,
                            values,
                            error_lls,
                            subject_idx,
                            v_val=None,
                            gamma_val=None,
                            s_val=None,
                            tau_val=None,
                            t0_val=0,
                            zerotol=1e-6,
                            error_weight=0.05,
                            boundary=1.,
                            gamma_bounds=(-2, 1),
                            offset=False,
                            f=10,
                            design=dict(v=dict(),
                                        gamma=dict(),
                                        s=dict(),
                                        tau=dict(),
                                        t0=dict())):
    """
    Create hierarchical PyMC3 model

    Parameters
    ----------
    rts : array_like, int or float
        response times in seconds per trial

    gaze : array_like, float 
        gaze distribution,
        specifying the observed gaze for
        each choice alternative in each trial
        shape: (trials x alternatives)
        gaze values must be between [0,1]

    values : array_like, float
        value of each choice alternative
        in each trial; 
        shape: (trials x alternatives)

    error_ll : array_like, floats
        likelihood of erroneous choice model
        for each subject, between [0,1]

    subject_idx : array_like, int
        indicating for each row in 
        [rts, gaze, values] which subject 
        the data belongs to
        shape: (trials)
        subjects indexed from [0 - (n_subjects - 1)]

    [v_val, gamma_val, s_val, tau_val] : float, optional
        if specified, specified parameter is 
        deterministically set to the specified value
        for all subjects

    t0_val : int, optional
        if specified, t0 value is determinstically
        set to this value for all subjects, defaults to 0

    zerotol : float, optional
        tolerance for mathematical stability,
        defaults to 1e-6

    error_weight : float, optional
        probability with which choices are modeled
        as resulting from errornous choice model
        (as specified by error_ll),
        defaults to 0.05 (5%) 

    boundary : float, optional
        decision boundary for linear
        stochastic race, defaults to 1

    gamma_bounds : tuple of floats, optional
        bounds for gamma distribution,
        defaults to [-10, 1]

    design : dict, optional
        dict with one entry per model parameter,
        specifying the parameter's dependency structure
        (see utils.get_design)

    offset : bool, optional
        if True, subject-level parameters are modeled as 
        offset from the group mean;
        if False, subject-level parameters are modeled as
        distributed around the group mean,
        defaults to True
        For further details, see: 
        https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/

    f : int, optional
        multiplying factor for the standard deviation
        of the hiearrchical distributions,
        higher values indicate wider prior,
        defaults to 10

    design : dict, optional
        dict with one entry per model parameter,
        specifying the parameter's dependency structure
        (see utils.get_design)

    Returns
    -------
    PyMC3 model instance
    """

    n_subjects = np.unique(subject_idx).size

    with pm.Model() as glam_hierarchical:

        # Mechanics
        b = pm.Deterministic('b', tt.constant(boundary, dtype='float32'))
        p_error = pm.Deterministic('p_error',
                                   tt.constant(error_weight, dtype='float32'))

        # Parameter priors
        v = generate_hierarchical_model_parameters(
            parameter='v',
            n_subjects=n_subjects,
            design=design['v'],
            mu_mean=0.633,
            mu_sd=f * 0.264,
            mu_lower=zerotol,
            mu_upper=4,
            sd_mean=0.26,
            sd_sd=f * 0.105,
            sd_lower=zerotol,
            sd_upper=1,
            val=v_val,
            offset=offset,
            within_dependent=design['v']['within_dependent'])

        gamma = generate_hierarchical_model_parameters(
            parameter='gamma',
            n_subjects=n_subjects,
            design=design['gamma'],
            mu_mean=0.124,
            mu_sd=f * 0.108,
            mu_lower=gamma_bounds[0],
            mu_upper=gamma_bounds[1],
            sd_mean=0.345,
            sd_sd=f * 0.098,
            sd_lower=zerotol,
            sd_upper=1,
            val=gamma_val,
            offset=offset,
            within_dependent=design['gamma']['within_dependent'])

        s = generate_hierarchical_model_parameters(
            parameter='s',
            n_subjects=n_subjects,
            design=design['s'],
            mu_mean=0.273,
            mu_sd=f * 0.076,
            mu_lower=zerotol,
            mu_upper=4,
            sd_mean=0.047,
            sd_sd=f * 0.008,
            sd_lower=zerotol,
            sd_upper=0.2,
            val=s_val,
            offset=offset,
            within_dependent=design['s']['within_dependent'])

        tau = generate_hierarchical_model_parameters(
            parameter='tau',
            n_subjects=n_subjects,
            design=design['tau'],
            mu_mean=1.03,
            mu_sd=f * 0.577,
            mu_lower=0,
            mu_upper=10,
            sd_mean=0.615,
            sd_sd=f * 0.259,
            sd_lower=zerotol,
            sd_upper=3,
            val=tau_val,
            offset=offset,
            within_dependent=design['tau']['within_dependent'])

        if t0_val is None:
            t0 = pm.Uniform('t0', 0, 1.0, testval=0.005, shape=(n_subjects, 1))
        else:
            t0 = pm.Deterministic('t0', tt.ones((n_subjects, 1)) * t0_val)

        # Likelihood
        def lda_logp(rt, gaze, values, error_lls, s_condition_index,
                     s_subject_index, v_condition_index, v_subject_index,
                     tau_condition_index, tau_subject_index,
                     gamma_condition_index, gamma_subject_index,
                     t0_condition_index, t0_subject_index, zerotol):

            # compute drifts
            R = make_R(
                v[tt.cast(v_subject_index, dtype='int32'),
                  tt.cast(v_condition_index, dtype='int32')][:, None],
                tau[tt.cast(tau_subject_index, dtype='int32'),
                    tt.cast(tau_condition_index, dtype='int32')][:, None],
                gamma[tt.cast(gamma_subject_index, dtype='int32'),
                      tt.cast(gamma_condition_index, dtype='int32')][:, None],
                values, gaze, zerotol)
            glam_ll = tt_wienerrace_pdf(
                rt[:, None], R,
                s[tt.cast(s_subject_index, dtype='int32'),
                  tt.cast(s_condition_index, dtype='int32')][:, None], b,
                t0[tt.cast(t0_subject_index, dtype='int32'),
                   tt.cast(t0_condition_index, dtype='int32')][:, None],
                zerotol)

            # mix likelihoods
            mixed_ll = ((1 - p_error) * glam_ll +
                        p_error * error_lls[subject_idx])

            mixed_ll = tt.where(tt.isnan(mixed_ll), 0., mixed_ll)
            mixed_ll = tt.where(tt.isinf(mixed_ll), 0., mixed_ll)
            return tt.log(mixed_ll + zerotol)

        obs = pm.DensityDist(
            'obs',
            logp=lda_logp,
            observed=dict(
                rt=rts,
                gaze=gaze,
                values=values,
                error_lls=error_lls,
                s_condition_index=design['s']['condition_index'].astype(
                    np.int32),
                s_subject_index=design['s']['subject_index'].astype(np.int32),
                v_condition_index=design['v']['condition_index'].astype(
                    np.int32),
                v_subject_index=design['v']['subject_index'].astype(np.int32),
                tau_condition_index=design['tau']['condition_index'].astype(
                    np.int32),
                tau_subject_index=design['tau']['subject_index'].astype(
                    np.int32),
                gamma_condition_index=design['gamma']
                ['condition_index'].astype(np.int32),
                gamma_subject_index=design['gamma']['subject_index'].astype(
                    np.int32),
                t0_condition_index=design['t0']['condition_index'].astype(
                    np.int32),
                t0_subject_index=design['t0']['subject_index'].astype(
                    np.int32),
                zerotol=zerotol))
    return glam_hierarchical
