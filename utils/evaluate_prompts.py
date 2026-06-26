"""Objective functions for scoring a Critic instruction on the demonstrations.

Each objective maps a confusion-matrix summary (TPR/TNR/FPR/FNR) to a scalar
that the optimisation loop maximises. The paper's default,
``extreme_weighted_sum_rates``, heavily prioritises minimising false positives
(maximising TNR) while only weakly rewarding true positives (lambda = 0.01).
"""


def compute_score(stats, objective='extreme_weighted_sum_rates', threshold=2):
    if objective == 'sum_rates':
        return stats['TPR'] + stats['TNR']
    elif objective == 'weighted_sum_rates':
        return .1 * stats['TPR'] + stats['TNR']
    elif objective == 'heavy_weighted_sum_rates':
        return .05 * stats['TPR'] + stats['TNR']
    elif objective == 'extreme_weighted_sum_rates':
        return .01 * stats['TPR'] + stats['TNR']
    elif objective == 'rates_prio_negatives':
        return stats['TPR'] + stats['TNR'] if stats['FPR'] < threshold else stats['TNR']
    elif objective == 'rates_prio_positives':
        return stats['TPR'] + stats['TNR'] if stats['FNR'] < threshold else stats['TPR']
    else:
        raise ValueError(f"Objective {objective} not recognized")
