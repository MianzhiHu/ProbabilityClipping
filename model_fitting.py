import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from DelayedDiscountingModels import *


def plot_trial_distributions(model_trials, metric='trial_nll', bins=50, save_dir=None):
    """Compare pooled trial distributions using common bins and x/y limits.

    model_trials maps model-family labels to four ordered softmax-rule DataFrames.
    All supplied trials are included; invalid values raise rather than being
    silently dropped. Counts describe pooled trials, not independent subjects.
    Returns figures by family; optionally exports editable SVG/PDF and PNG.
    """
    from pathlib import Path
    from matplotlib.ticker import MaxNLocator

    if metric not in ('trial_nll', 'trial_prob'):
        raise ValueError("metric must be 'trial_nll' or 'trial_prob'")
    if not model_trials:
        raise ValueError('Provide at least one model family.')
    values = {}
    for family, variants in model_trials.items():
        if len(variants) != 4:
            raise ValueError('Each family must contain four comparison panels.')
        for rule, frame in variants.items():
            data = np.asarray(frame[metric], dtype=float)
            if data.ndim != 1 or not data.size or not np.isfinite(data).all():
                raise ValueError(f'{family}/{rule}: provide finite, nonempty trial values.')
            if (data < 0).any() or (metric == 'trial_prob' and (data > 1).any()):
                raise ValueError(f'{family}/{rule}: values outside the metric domain.')
            values[family, rule] = data

    upper = 1.0 if metric == 'trial_prob' else max(v.max() for v in values.values())
    edges = np.linspace(0.0, upper if upper > 0 else 1.0, bins + 1)
    ymax = max(np.histogram(v, bins=edges)[0].max() for v in values.values())
    labels = {'e02': r'$\epsilon = 10^{-2}$', 'e04': r'$\epsilon = 10^{-4}$',
              'e16': r'$\epsilon = 10^{-16}$', 'e64': r'$\epsilon = 10^{-64}$',
              'unlimited': 'Unlimited'}
    style = {
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }
    figures = {}
    with plt.rc_context(style):
        for family, variants in model_trials.items():
            fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 125 / 25.4),
                                     sharex=True, sharey=True, layout='constrained')
            for letter, ax, rule in zip('abcd', axes.flat, variants):
                data = values[family, rule]
                color = '#5B7894' if rule != 'unlimited' else '#B47447'
                ax.hist(data, bins=edges, color=color, edgecolor='white', linewidth=0.25)
                ax.set(xlim=(edges[0], edges[-1]), ylim=(0, ymax * 1.08),
                       title=labels.get(rule, rule))
                ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
                        fontsize=9, fontweight='bold', va='bottom')
                ax.text(0.97, 0.95, f'n = {data.size:,} trials', transform=ax.transAxes,
                        ha='right', va='top', fontsize=7)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                ax.tick_params(length=3, width=0.6)
            fig.suptitle(f'{family} model', fontsize=10, fontweight='bold')
            fig.supxlabel('Trial negative log-likelihood' if metric == 'trial_nll'
                          else 'Observed-choice probability')
            fig.supylabel('Number of trials')
            if save_dir is not None:
                output = Path(save_dir)
                output.mkdir(parents=True, exist_ok=True)
                stem = output / f'{family.lower()}_{metric}_distributions'
                fig.savefig(stem.with_suffix('.svg'))
                fig.savefig(stem.with_suffix('.pdf'))
                fig.savefig(stem.with_suffix('.png'), dpi=300)
            figures[family] = fig
    return figures


# ======================================================================================================================
# Delayed Discounting Model Fitting
# ======================================================================================================================
# Exponential Models
expo_e02 = DelayedDiscounting('dd_exponential', softmax_function='e02')
expo_e04 = DelayedDiscounting('dd_exponential', softmax_function='e04')
expo_e16 = DelayedDiscounting('dd_exponential', softmax_function='e16')
expo_e64 = DelayedDiscounting('dd_exponential', softmax_function='e64')
expo_unlimited = DelayedDiscounting('dd_exponential', softmax_function='unlimited')

# Hyperbolic Models
hyperbolic_e02 = DelayedDiscounting('dd_hyperbolic', softmax_function='e02')
hyperbolic_e04 = DelayedDiscounting('dd_hyperbolic', softmax_function='e04')
hyperbolic_e16 = DelayedDiscounting('dd_hyperbolic', softmax_function='e16')
hyperbolic_e64 = DelayedDiscounting('dd_hyperbolic', softmax_function='e64')
hyperbolic_unlimited = DelayedDiscounting('dd_hyperbolic', softmax_function='unlimited')

# Hyperboloid Models
hyperboloid_e02 = DelayedDiscounting('dd_hyperboloid', softmax_function='e02')
hyperboloid_e04 = DelayedDiscounting('dd_hyperboloid', softmax_function='e04')
hyperboloid_e16 = DelayedDiscounting('dd_hyperboloid', softmax_function='e16')
hyperboloid_e64 = DelayedDiscounting('dd_hyperboloid', softmax_function='e64')
hyperboloid_unlimited = DelayedDiscounting('dd_hyperboloid', softmax_function='unlimited')

dd_data = pd.read_csv('./delayed_discounting/empirical_data/dd_empirical_data.csv')
dd_data['choice'] = dd_data['choice'].map({'smaller_sooner': 0, 'larger_later': 1})
print(dd_data['choice'].value_counts())
dd_dict = dict_generator_cognitive(dd_data, task='dd')

if __name__ == '__main__':
    n_iter = 200
    saving_path = './delayed_discounting/fit_results/'
    os.makedirs(saving_path, exist_ok=True)

    # Fit exponential models
    results_exponential_e02 = expo_e02.fit(dd_dict, n_iter)
    results_exponential_e04 = expo_e04.fit(dd_dict, n_iter)
    results_exponential_e16 = expo_e16.fit(dd_dict, n_iter)
    results_exponential_e64 = expo_e64.fit(dd_dict, n_iter)
    results_exponential_unlimited = expo_unlimited.fit(dd_dict, n_iter)

    results_exponential_e02.to_csv(saving_path + 'exponential_e02.csv')
    results_exponential_e04.to_csv(saving_path + 'exponential_e04.csv')
    results_exponential_e16.to_csv(saving_path + 'exponential_e16.csv')
    results_exponential_e64.to_csv(saving_path + 'exponential_e64.csv')
    results_exponential_unlimited.to_csv(saving_path + 'exponential_unlimited.csv')

    # Fit hyperbolic models
    results_hyperbolic_e02 = hyperbolic_e02.fit(dd_dict, n_iter)
    results_hyperbolic_e04 = hyperbolic_e04.fit(dd_dict, n_iter)
    results_hyperbolic_e16 = hyperbolic_e16.fit(dd_dict, n_iter)
    results_hyperbolic_e64 = hyperbolic_e64.fit(dd_dict, n_iter)
    results_hyperbolic_unlimited = hyperbolic_unlimited.fit(dd_dict, n_iter)

    results_hyperbolic_e02.to_csv(saving_path + 'hyperbolic_e02.csv')
    results_hyperbolic_e04.to_csv(saving_path + 'hyperbolic_e04.csv')
    results_hyperbolic_e16.to_csv(saving_path + 'hyperbolic_e16.csv')
    results_hyperbolic_e64.to_csv(saving_path + 'hyperbolic_e64.csv')
    results_hyperbolic_unlimited.to_csv(saving_path + 'hyperbolic_unlimited.csv')

    # Fit hyperboloid models
    results_hyperboloid_e02 = hyperboloid_e02.fit(dd_dict, n_iter)
    results_hyperboloid_e04 = hyperboloid_e04.fit(dd_dict, n_iter)
    results_hyperboloid_e16 = hyperboloid_e16.fit(dd_dict, n_iter)
    results_hyperboloid_e64 = hyperboloid_e64.fit(dd_dict, n_iter)
    results_hyperboloid_unlimited = hyperboloid_unlimited.fit(dd_dict, n_iter)

    results_hyperboloid_e02.to_csv(saving_path + 'hyperboloid_e02.csv')
    results_hyperboloid_e04.to_csv(saving_path + 'hyperboloid_e04.csv')
    results_hyperboloid_e16.to_csv(saving_path + 'hyperboloid_e16.csv')
    results_hyperboloid_e64.to_csv(saving_path + 'hyperboloid_e64.csv')
    results_hyperboloid_unlimited.to_csv(saving_path + 'hyperboloid_unlimited.csv')

    print(f'Exponential e02 average bic: {results_exponential_e02["BIC"].mean()}')
    print(f'Hyperbolic e02 average bic: {results_hyperbolic_e02["BIC"].mean()}')
    print(f'Hyperboloid e02 average bic: {results_hyperboloid_e02["BIC"].mean()}')
    print(f'Exponential e04 average bic: {results_exponential_e04["BIC"].mean()}')
    print(f'Hyperbolic e04 average bic: {results_hyperbolic_e04["BIC"].mean()}')
    print(f'Hyperboloid e04 average bic: {results_hyperboloid_e04["BIC"].mean()}')
    print(f'Exponential e16 average bic: {results_exponential_e16["BIC"].mean()}')
    print(f'Hyperbolic e16 average bic: {results_hyperbolic_e16["BIC"].mean()}')
    print(f'Hyperboloid e16 average bic: {results_hyperboloid_e16["BIC"].mean()}')
    print(f'Exponential e64 average bic: {results_exponential_e64["BIC"].mean()}')
    print(f'Hyperbolic e64 average bic: {results_hyperbolic_e64["BIC"].mean()}')
    print(f'Hyperboloid e64 average bic: {results_hyperboloid_e64["BIC"].mean()}')
    print(f'Exponential unlimited average bic: {results_exponential_unlimited["BIC"].mean()}')
    print(f'Hyperbolic unlimited average bic: {results_hyperbolic_unlimited["BIC"].mean()}')
    print(f'Hyperboloid unlimited average bic: {results_hyperboloid_unlimited["BIC"].mean()}')

    print(f'Exponential e02 clipping count: {results_exponential_e02["clipping_count"].mean()}')
    print(f'Hyperbolic e02 average clipping count: {results_hyperbolic_e02["clipping_count"].mean()}')
    print(f'Hyperboloid e02 average clipping count: {results_hyperboloid_e02["clipping_count"].mean()}')
    print(f'Exponential e04 average clipping count: {results_exponential_e04["clipping_count"].mean()}')
    print(f'Hyperbolic e04 average clipping count: {results_hyperbolic_e04["clipping_count"].mean()}')
    print(f'Hyperboloid e04 average clipping count: {results_hyperboloid_e04["clipping_count"].mean()}')
    print(f'Exponential e16 average clipping count: {results_exponential_e16["clipping_count"].mean()}')
    print(f'Hyperbolic e16 average clipping count: {results_hyperbolic_e16["clipping_count"].mean()}')
    print(f'Hyperboloid e16 average clipping count: {results_hyperboloid_e16["clipping_count"].mean()}')
    print(f'Exponential e64 average clipping count: {results_exponential_e64["clipping_count"].mean()}')
    print(f'Hyperbolic e64 average clipping count: {results_hyperbolic_e64["clipping_count"].mean()}')
    print(f'Hyperboloid e64 average clipping count: {results_hyperboloid_e64["clipping_count"].mean()}')
    print(f'Exponential unlimited average clipping count: {results_exponential_unlimited["clipping_count"].mean()}')
    print(f'Hyperbolic unlimited average clipping count: {results_hyperbolic_unlimited["clipping_count"].mean()}')
    print(f'Hyperboloid unlimited average clipping count: {results_hyperboloid_unlimited["clipping_count"].mean()}')

    expo_e02_trial = results_exponential_e02.explode(['trial_nll', 'trial_prob']).reset_index()
    expo_e04_trial = results_exponential_e04.explode(['trial_nll', 'trial_prob']).reset_index()
    expo_e16_trial = results_exponential_e16.explode(['trial_nll', 'trial_prob']).reset_index()
    expo_e64_trial = results_exponential_e64.explode(['trial_nll', 'trial_prob']).reset_index()
    expo_unlimited_trial = results_exponential_unlimited.explode(['trial_nll', 'trial_prob']).reset_index()

    hyperbolic_e02_trial = results_hyperbolic_e02.explode(['trial_nll', 'trial_prob']).reset_index()
    hyperbolic_e04_trial = results_hyperbolic_e04.explode(['trial_nll', 'trial_prob']).reset_index()
    hyperbolic_e16_trial = results_hyperbolic_e16.explode(['trial_nll', 'trial_prob']).reset_index()
    hyperbolic_e64_trial = results_hyperbolic_e64.explode(['trial_nll', 'trial_prob']).reset_index()
    hyperbolic_unlimited_trial = results_hyperbolic_unlimited.explode(['trial_nll', 'trial_prob']).reset_index()

    hyperboloid_e02_trial = results_hyperboloid_e02.explode(['trial_nll', 'trial_prob']).reset_index()
    hyperboloid_e04_trial = results_hyperboloid_e04.explode(['trial_nll', 'trial_prob']).reset_index()
    hyperboloid_e16_trial = results_hyperboloid_e16.explode(['trial_nll', 'trial_prob']).reset_index()
    hyperboloid_e64_trial = results_hyperboloid_e64.explode(['trial_nll', 'trial_prob']).reset_index()
    hyperboloid_unlimited_trial = results_hyperboloid_unlimited.explode(['trial_nll', 'trial_prob']).reset_index()


    plot_trial_distributions({
        'Hyperbolic': {
            'e02': hyperbolic_e02_trial, 'e04': hyperbolic_e04_trial,
            'e16': hyperbolic_e16_trial, 'unlimited': hyperbolic_unlimited_trial,
        },
        'Exponential': {
            'e02': expo_e02_trial, 'e04': expo_e04_trial,
            'e16': expo_e16_trial, 'unlimited': expo_unlimited_trial,
        },
        'Hyperboloid': {
            'e02': hyperboloid_e02_trial, 'e04': hyperboloid_e04_trial,
            'e16': hyperboloid_e16_trial, 'unlimited': hyperboloid_unlimited_trial,
        },
    }, metric='trial_prob')
    plt.show()
