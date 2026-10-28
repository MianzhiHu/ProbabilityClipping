import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax


# Parameter bounds for each model type. Each tuple is (lower_bound, upper_bound).
model_bounds = {
    'dd_exponential': [(0.01, 20.0), (0.0, 1.0)],
    'dd_hyperbolic': [(0.01, 20.0), (0.0, 1.0)],
    'dd_hyperboloid': [(0.01, 20.0), (0.0, 1.0), (0.0, 5.0)],
}


def random_initial_guess(bounds):
    return [np.random.uniform(low, high) for (low, high) in bounds]


def fit_participant(model, participant_id, pdata, model_type, num_iterations=1000):
    print(f"Fitting participant {participant_id}...")
    start_time = time.time()

    total_n = len(pdata['choice'])

    model.iteration = 0
    best_nll = np.inf  # Initialize best negative log likelihood to be positive infinity
    best_parameters = None
    best_clipping_count = None
    best_trial_nll = None
    best_trial_prob = None

    for _ in range(num_iterations):  # Randomly initiate the starting parameter for 1000 times

        model.iteration += 1

        print('Participant {} - Iteration [{}/{}]'.format(participant_id, model.iteration,
                                                          num_iterations))
        bounds = model.parameter_bounds

        # generate initial guesses from dictionary bounds
        initial_guess = random_initial_guess(bounds)

        result = minimize(model.negative_log_likelihood, initial_guess,
                          args=tuple(pdata[column] for column in model.input_columns),
                          bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})

        if result.fun < best_nll:
            best_nll = result.fun
            best_parameters = result.x

    # Optimizers may finish on a probe other than result.x. Save diagnostics
    # recomputed at the selected best parameters.
    best_nll = model.negative_log_likelihood(best_parameters, *(pdata[column] for column in model.input_columns))
    best_clipping_count = model.clipping_count
    best_trial_nll = model.trial_nll.copy()
    best_trial_prob = model.trial_prob.copy()

    # Calculate AIC and BIC
    k = len(best_parameters)  # Number of parameters
    aic = 2 * k + 2 * best_nll
    bic = k * np.log(total_n) + 2 * best_nll

    result_dict = {
        'participant_id': participant_id,
        'best_nll': best_nll,
        'best_parameters': best_parameters,
        'AIC': aic,
        'BIC': bic,
        'clipping_count': best_clipping_count,
        'trial_nll': best_trial_nll,
        'trial_prob': best_trial_prob
    }

    print(f"Participant {participant_id} fitted in {(time.time() - start_time) / 60} minutes.")

    return result_dict


class DelayedDiscounting:
    def __init__(self, model_type, softmax_function='unlimited'):
        if model_type not in ['dd_exponential', 'dd_hyperbolic', 'dd_hyperboloid']:
            raise ValueError('Use exponential, hyperbolic, or hyperboloid')
        self.model_type = model_type
        self.input_columns = ['large_amount', 'later_delay', 'small_amount', 'choice']

        # Initialize potential parameters
        self._default_attrs = ['t', 'k', 's']
        for attr in self._default_attrs:
            setattr(self, attr, None)

        self._param_map = {
            'dd_exponential': {'t': 0, 'k': 1},
            'dd_hyperbolic': {'t': 0, 'k': 1},
            'dd_hyperboloid': {'t': 0, 'k': 1, 's': 2},
        }
        self.parameter_bounds = list(model_bounds[model_type])

        self._function_map = {
            'dd_exponential': self.exponential_function,
            'dd_hyperbolic': self.hyperbolic_function,
            'dd_hyperboloid': self.hyperboloid_function,
        }

        self._softmax_map = {
            'e64': self.softmax_e64,
            'e16': self.softmax_e16,
            'e04': self.softmax_e04,
            'e02': self.softmax_e02,
            'unlimited': self.softmax_unlimited
        }
        if softmax_function not in self._softmax_map:
            raise ValueError('softmax_function must be one of: ' + ', '.join(self._softmax_map))
        self._softmax_function = self._softmax_map[softmax_function]
        self.softmax_function = softmax_function
        self.clipping_count = 0
        self.trial_nll = []
        self.trial_prob = []

    def _clip_probabilities(self, probabilities, epsilon):
        """Count trials with any raw p < epsilon, then clip to [epsilon, 1-epsilon].

        Count once per trial, independent of the observed choice. This is a
        per-call count, not an accumulation over optimizer evaluations.
        """
        self.clipping_count = int(np.count_nonzero(np.any(probabilities <= epsilon, axis=-1)))
        return np.clip(probabilities, epsilon, 1 - epsilon)

    def softmax_e64(self, x):
        # Apply the original custom softmax separately to each trial's options.
        x = np.asarray(x, dtype=float)
        x_norm = x - np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(self.t * x_norm)
        return self._clip_probabilities(e_x / e_x.sum(axis=-1, keepdims=True), 1e-64)

    def softmax_e02(self, x):
        # Apply the original custom softmax separately to each trial's options.
        x = np.asarray(x, dtype=float)
        x_norm = x - np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(self.t * x_norm)
        return self._clip_probabilities(e_x / e_x.sum(axis=-1, keepdims=True), 1e-2)

    def softmax_e16(self, x):
        # Apply the original custom softmax separately to each trial's options.
        x = np.asarray(x, dtype=float)
        x_norm = x - np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(self.t * x_norm)
        return self._clip_probabilities(e_x / e_x.sum(axis=-1, keepdims=True), 1e-16)

    def softmax_unlimited(self, x):
        """Return probabilities for sampling/display; likelihood uses logs directly.

        Very small returned probabilities may underflow to zero in float64;
        this does not truncate the log probabilities used for fitting.
        """
        return np.exp(self.log_softmax_unlimited(x))

    def log_softmax_unlimited(self, x):
        """Compute uncapped log probabilities without a probability-space round trip."""
        self.clipping_count = 0
        x = np.asarray(x, dtype=float)
        x_norm = x - np.max(x, axis=-1, keepdims=True)
        x_norm_t = self.t * x_norm
        return log_softmax(x_norm_t, axis=-1)

    def _trial_likelihood(self, values, choice):
        """Store observed-choice NLLs and back-calculated probabilities in trial order."""
        if self.softmax_function == 'unlimited':
            log_probabilities = self.log_softmax_unlimited(values)
        else:
            # Preserve the existing probability clipping for the other rules.
            log_probabilities = np.log(self._softmax_function(values))
        choice = np.asarray(choice, dtype=int)
        trial_nll = -log_probabilities[np.arange(len(choice)), choice]
        self.trial_nll = trial_nll.tolist()
        self.trial_prob = np.exp(-trial_nll).tolist()
        return trial_nll, log_probabilities

    def softmax_e04(self, x):
        # Apply the original custom softmax separately to each trial's options.
        x = np.asarray(x, dtype=float)
        x_norm = x - np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(self.t * x_norm)
        return self._clip_probabilities(e_x / e_x.sum(axis=-1, keepdims=True), 1e-4)

    def exponential_function(self, large_amout, later_delay):
        subjective_value = large_amout * np.exp(-self.k * later_delay)
        return subjective_value

    def hyperbolic_function(self, large_amout, later_delay):
        subjective_value = large_amout / (1 + self.k * later_delay)
        return subjective_value

    def hyperboloid_function(self, large_amout, later_delay):
        subjective_value = large_amout / (1 + self.k * later_delay)**self.s
        return subjective_value

    def negative_log_likelihood(self, params, large_amount, later_delay, small_amount, choice):
        cfg = self._param_map.get(self.model_type, {})
        for attr, idx in cfg.items():
            setattr(self, attr, params[idx])

        # Make sure inputs are numeric numpy arrays
        large_amount = np.asarray(large_amount, dtype=float)
        later_delay = np.asarray(later_delay, dtype=float)
        small_amount = np.asarray(small_amount, dtype=float)
        choice = np.asarray(choice)

        # Compute subjective values for the larger-later option and stack with smaller-sooner values
        # 0 = smaller-sooner; 1 = larger-later
        subjective_values = self._function_map[self.model_type](large_amount, later_delay)
        values = np.column_stack([small_amount, subjective_values])

        trial_nll, _ = self._trial_likelihood(values, choice)
        total_nll = trial_nll.sum()

        return float(total_nll)

    def simulate(self, n_participants, n_trials, large_amount_range, later_delay_range, small_amount_range, manual_bounds=None):
        """Return simulated trials with participant generating parameters as columns.

        Parameters (t, k, and s for hyperboloid) are repeated on each trial so
        they are preserved by DataFrame.to_csv(..., index=False). The generating
        dictionary is retained for compatibility with existing callers.
        Amounts and delays are sampled as integers within inclusive ranges.
        """
        simulated_data = []
        integer_ranges = []
        for low, high in (large_amount_range, later_delay_range, small_amount_range):
            low, high = int(np.ceil(low)), int(np.floor(high))
            integer_ranges.append((low, high + 1))
        if manual_bounds is not None:
            bounds = manual_bounds
        else:
            bounds = self.parameter_bounds
        if len(bounds) != len(self._param_map[self.model_type]):
            raise ValueError('manual_bounds must match the active parameters in order: '
                             + ', '.join(self._param_map[self.model_type]))

        for participant_id in range(n_participants):
            # Randomly sample parameters within the bounds for this model
            params = [np.random.uniform(low, high) for (low, high) in bounds]
            for attr, idx in self._param_map[self.model_type].items():
                setattr(self, attr, params[idx])
            generating = {name: getattr(self, name) for name in self._param_map[self.model_type]}

            for trial in range(n_trials):
                large_amount = np.random.randint(*integer_ranges[0])
                later_delay = np.random.randint(*integer_ranges[1])
                small_amount = np.random.randint(*integer_ranges[2])

                # Compute subjective values and choice probabilities
                subjective_value = self._function_map[self.model_type](large_amount, later_delay)
                values = np.array([small_amount, subjective_value])
                probabilities = self._softmax_function(values.reshape(1, -1)).flatten()

                # Simulate choice based on probabilities
                choice = np.random.choice([0, 1], p=probabilities)

                # Record generating parameters and trial data
                simulated_data.append({
                    'participant_id': participant_id,
                    'model_type': self.model_type,
                    'trial': trial,
                    'large_amount': large_amount,
                    'later_delay': later_delay,
                    'small_amount': small_amount,
                    'choice': choice,
                    **generating
                })

        return pd.DataFrame(simulated_data)

    def fit(self, data, num_iterations=20, max_workers=None):
        # Detect how many works we have
        workers = max_workers or os.cpu_count()

        # Creating a list to hold the future results
        futures = []
        results = []

        # Starting a pool of workers with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submitting jobs to the executor for each participant
            for participant_id, participant_data in data.items():
                # fit_participant is the function to be executed in parallel
                future = executor.submit(fit_participant, self, participant_id, participant_data, self.model_type,
                                         num_iterations)
                futures.append(future)

            # Collecting results as they complete
            for future in futures:
                results.append(future.result())

        return pd.DataFrame(results)

    def evaluate(self, params, data):
        cfg = self._param_map[self.model_type]

        results = []

        for participant_id, participant_data in data.items():

            participant_params = np.asarray(params.loc[participant_id], dtype=float)

            # Set model parameters
            for attr, idx in cfg.items():
                setattr(self, attr, participant_params[idx])

            # Get THIS participant's test data
            large_amount = np.asarray(participant_data['large_amount'], dtype=float)
            later_delay = np.asarray(participant_data['later_delay'], dtype=float)
            small_amount = np.asarray(participant_data['small_amount'], dtype=float)
            choice = np.asarray(participant_data['choice'],dtype=int)

            # Subjective values
            subjective_values = self._function_map[self.model_type](large_amount, later_delay)
            values = np.column_stack([small_amount, subjective_values])

            trial_nll, log_probabilities = self._trial_likelihood(values, choice)
            total_nll = trial_nll.sum()
            mean_nll = trial_nll.mean()

            # Accuracy
            predicted_choice = log_probabilities.argmax(axis=1)
            n_correct = int((predicted_choice == choice).sum())
            accuracy = n_correct / len(choice)

            results.append({
                'participant_id': participant_id,
                'total_nll': float(total_nll),
                'mean_nll': float(mean_nll),
                'accuracy': float(accuracy),
                'n_trials': len(choice),
                'n_correct': n_correct,
                'clipping_count': self.clipping_count,
                'trial_nll': self.trial_nll.copy(),
                'trial_prob': self.trial_prob.copy()
            })

        return pd.DataFrame(results)


# Helper function to convert a dataframe into a dictionary of participant data
def dict_generator_cognitive(df, task='dd'):
    """
    Convert a dataframe into a dictionary.

    Parameters:
    - df: Dataframe to be converted.

    Returns:
    - A dictionary of the dataframe.
    """
    def find_col(candidates):
        """Return first candidate that’s in df.columns, else error."""
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(f"None of {candidates!r} found in DataFrame columns")

    # define for each task which output‐keys map to which column‐name candidates
    COLUMN_MAP = {
        'dd': {
            'large_amount':   ['large_amount'],
            'small_amount':   ['small_amount'],
            'later_delay':    ['later_delay'],
            'choice':   ['choice'],
        }
    }

    if task not in COLUMN_MAP:
        raise ValueError(f"Unsupported task {task!r}")

    # optional: allow different grouping columns too
    group_col = find_col(['worker_id'])

    d = {}
    for subject_id, group in df.groupby(group_col):
        entry = {}
        for key, candidates in COLUMN_MAP[task].items():
            actual_col = find_col(candidates)
            entry[key] = group[actual_col].tolist()
        d[subject_id] = entry

    return d
