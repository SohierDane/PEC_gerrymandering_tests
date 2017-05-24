"""
Provides utilities for Sam Wang's 'hack the gerrymandering standard' contest.

All functions expect vote shares entered as numpy arrays.
"""

from __future__ import division

import scipy.stats as scs
import numpy as np


# provided for validation purposes
ORIGINAL_DEMOCRAT_VOTES = np.array([0.849, 0.905, 0.428, 0.366, 0.371, 0.429,
                                    0.406, 0.434, 0.383, 0.344, 0.415, 0.483,
                                    0.691, 0.769, 0.432, 0.416, 0.603, 0.36])


def is_valid_statewide_margin(dem_votes, rep_votes):
    return abs(dem_votes.sum() - rep_votes.sum()) < 0.01


def has_enough_republican_wins(dem_votes):
    return (dem_votes>0.5).sum() <= 5


def mean_median_difference(dem_votes):
    return (np.mean(dem_votes) - np.median(dem_votes)) / np.std(dem_votes, ddof=1)


def evaded_mean_med_difference(dem_votes):
    return mean_median_difference(dem_votes) < 0.3


def ttest_p_value(dem_votes, rep_votes):
    p_value_index = 1
    dem_win_margins = dem_votes[dem_votes > rep_votes]
    rep_win_margins = rep_votes[rep_votes > dem_votes]
    return scs.ttest_ind(dem_win_margins, rep_win_margins)[p_value_index] / 2


def evaded_t_test(dem_votes, rep_votes):
     return ttest_p_value(dem_votes, rep_votes) > 0.05


def is_valid_entry(dem_votes, rep_votes=None):
    if rep_votes is None:
        rep_votes = 1 - dem_votes
    
    is_valid = (is_valid_statewide_margin(dem_votes, rep_votes) and
                has_enough_republican_wins(dem_votes) and
                (evaded_mean_med_difference(dem_votes) or
                 evaded_t_test(dem_votes, rep_votes)))
    return is_valid
