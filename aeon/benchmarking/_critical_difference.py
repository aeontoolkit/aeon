"""Function to compute and plot critical difference diagrams."""

__author__ = ["SveaMeyer13", "dguijo"]

import math
import operator

import numpy as np
from scipy.stats import distributions, find_repeats, rankdata, wilcoxon

from aeon.benchmarking.utils import get_qalpha
from aeon.utils.validation._dependencies import _check_soft_dependencies


def _check_friedman(n_estimators, n_datasets, ranked_data, alpha):
    """
    Check whether Friedman test is significant.

    Larger parts of code copied from scipy.

    Parameters
    ----------
    n_estimators : int
      number of strategies to evaluate
    n_datasets : int
      number of datasets classified per strategy
    ranked_data : np.array (shape: n_estimators * n_datasets)
      rank of strategy on dataset

    Returns
    -------
    is_significant : bool
      Indicates whether strategies differ significantly in terms of performance
      (according to Friedman test).
    """
    if n_estimators < 3:
        raise ValueError(
            "At least 3 sets of measurements must be given for Friedmann test, "
            f"got {n_estimators}."
        )

    # calculate c to correct chisq for ties:
    ties = 0
    for i in range(n_datasets):
        replist, repnum = find_repeats(ranked_data[i])
        for t in repnum:
            ties += t * (t * t - 1)
    c = 1 - ties / (n_estimators * (n_estimators * n_estimators - 1) * n_datasets)

    ssbn = np.sum(ranked_data.sum(axis=0) ** 2)
    chisq = (
        12.0 / (n_estimators * n_datasets * (n_estimators + 1)) * ssbn
        - 3 * n_datasets * (n_estimators + 1)
    ) / c
    p = distributions.chi2.sf(chisq, n_estimators - 1)
    if p < alpha:
        is_significant = True
    else:
        is_significant = False
    return is_significant


def nemenyi_cliques(n_estimators, n_datasets, avranks, alpha):
    """Find cliques using post hoc Nemenyi test."""
    # Get critical value, there is an exact way now
    qalpha = get_qalpha(alpha)
    # calculate critical difference with Nemenyi
    cd = qalpha[n_estimators] * np.sqrt(
        n_estimators * (n_estimators + 1) / (6 * n_datasets)
    )
    # compute statistically similar cliques
    cliques = np.tile(avranks, (n_estimators, 1)) - np.tile(
        np.vstack(avranks.T), (1, n_estimators)
    )
    cliques[cliques < 0] = np.inf
    cliques = cliques < cd

    cliques = build_cliques(cliques)

    return cliques


def wilcoxon_holm_cliques(results, labels, avranks, alpha):
    """Find cliques using Wilcoxon and post hoc Holm test."""
    # get number of strategies:
    n_estimators = results.shape[1]

    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(n_estimators - 1):
        # get the name of classifier one
        classifier_1 = labels[i]
        # get the performance of classifier one
        perf_1 = np.array(results[:, i])

        for j in range(i + 1, n_estimators):
            # get the name of the second classifier
            classifier_2 = labels[j]
            # get the performance of classifier two
            perf_2 = np.array(results[:, j])
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method="wilcox")[1]
            # append to the list
            p_values.append((classifier_1, classifier_2, p_value, False))

    # get the number of hypothesis
    n_hypothesis = len(p_values)

    # sort the list in ascending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # correct alpha with holm
    new_alpha = float(alpha / (n_estimators - 1))

    ordered_labels = [i for _, i in sorted(zip(avranks, labels))]

    same = np.eye(len(ordered_labels), dtype=bool)

    # loop through the hypothesis
    for i in range(n_hypothesis):
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            idx_0 = np.where(np.array(ordered_labels) == p_values[i][0])[0][0]
            idx_1 = np.where(np.array(ordered_labels) == p_values[i][1])[0][0]
            same[idx_0][idx_1] = True
            same[idx_1][idx_0] = True

    cliques = build_cliques(same)

    return cliques


def build_cliques(same):
    """Build cliques."""
    n_estimators = same.shape[1]

    for i in range(n_estimators):
        if np.sum(same[i, :]) > 1:
            true_values_i = np.where(same[i, :] == 1)[0]
            first_true_i = true_values_i[0]
            last_true_i = true_values_i[-1]
            for j in range(i + 1, n_estimators):
                if np.sum(same[j, :]) >= 1:
                    true_values_j = np.where(same[j, :] == 1)[0]
                    first_true_j = true_values_j[0]
                    last_true_j = true_values_j[-1]
                    # if j is contained in i
                    if first_true_i <= first_true_j and last_true_i >= last_true_j:
                        if len(true_values_i) >= len(true_values_j):
                            same[j, :] = 0
                        else:
                            same[i, :] = 0
                    # if i is contained in j
                    elif first_true_i >= first_true_j and last_true_i <= last_true_j:
                        if len(true_values_i) >= len(true_values_j):
                            same[j, :] = 0
                        else:
                            same[i, :] = 0

    n = np.sum(same, 1)
    cliques = same[n > 1, :]

    return cliques


def plot_critical_difference(
    scores,
    labels,
    highlight=None,
    errors=False,
    cliques=None,
    clique_method="holm",
    alpha=0.05,
    width=6,
    textspace=1.5,
    reverse=True,
):
    """
    Draw critical difference diagram.

    Step 1 & 2: Calculate average ranks from data
    Step 3: Use Friedman test to check whether
    the strategy significantly affects the classification performance
    Step 4: Compute critical differences using Nemenyi post-hoc test.
    (How much should the average rank of two strategies differ to be
     statistically significant)
    Step 5: Compute statistically similar cliques of strategies
    Step 6: Draw the diagram

    See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Parts of the code are copied and adapted from here:
    https://github.com/hfawaz/cd-diagram

    Parameters
    ----------
        scores : np.array
            scores (either accuracies or errors) of dataset x strategy
        labels : list of estimators
            list with names of the estimators. Order should be the same as scores
        highlight: dict with labels and HTML colours to be used, default = None
            dict with labels and HTML colours to be used for highlighting. Order should
            be the same as scores
        errors : bool, default = False
            indicates whether scores are passed as errors (default) or accuracies
        cliques : lists of bit vectors, default = None
            e.g. [[0,1,1,1,0,0], [0,0,0,0,1,1]]
            statistically similiar cliques of estimators
            If none, cliques will be computed depending on clique_method
        clique_method : string, default = "holm"
            clique forming method, to include "nemenyi" and "holm"
        alpha : float default = 0.05
             Alpha level for statistical tests currently supported: 0.1, 0.05 or 0.01)
        width : int, default = 6
           width in inches
        textspace : int
           space on figure sides (in inches) for the method names (default: 1.5)
        reverse : bool, default = True
           if set to 'True', the lowest rank is on the right

    Returns
    -------
    fig: matplotlib.figure
        Figure created.

    Example
    -------
    >>> from aeon.benchmarking import plot_critical_difference
    >>> from aeon.benchmarking.results_loaders import get_estimator_results_as_array
    >>> methods = ["IT", "WEASEL-Dilation", "HIVECOTE2", "FreshPRINCE"]
    >>> results = get_estimator_results_as_array(estimators=methods)
    >>> plot = plot_critical_difference(results[0], methods, alpha=0.1)\
        # doctest: +SKIP
    >>> plot.show()  # doctest: +SKIP
    >>> plot.savefig("scatterplot.pdf", bbox_inches="tight")  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    # Helper Functions
    # get number of datasets and strategies:
    n_datasets, n_estimators = scores.shape[0], scores.shape[1]

    # Step 1: rank data: best algorithm gets rank of 1 second best rank of 2...
    # in case of ties average ranks are assigned
    if errors:
        # low is good -> rank 1
        ranked_data = rankdata(scores, axis=1)
    else:
        # assign opposite ranks
        ranked_data = rankdata(-1 * scores, axis=1)

    # Step 2: calculate average rank per strategy
    avranks = ranked_data.mean(axis=0)
    # Sort labels
    combined = zip(avranks, labels)
    temp_labels = []

    x = sorted(combined)
    i = 0
    for s, n in x:
        avranks[i] = s
        temp_labels.append(n)
        i = i + 1

    # sort out colours for labels
    if highlight is not None:
        colours = [
            highlight[label] if label in highlight else "#000000"
            for label in temp_labels
        ]
    else:
        colours = ["#000000"] * len(temp_labels)

    # Step 3 : check whether Friedman test is significant
    is_significant = _check_friedman(n_estimators, n_datasets, ranked_data, alpha)
    # Step 4: If Friedman test is significant find cliques
    if is_significant:
        if cliques is None:
            if clique_method == "nemenyi":
                cliques = nemenyi_cliques(n_estimators, n_datasets, avranks, alpha)
            elif clique_method == "holm":
                cliques = wilcoxon_holm_cliques(
                    scores, labels, ranked_data.mean(axis=0), alpha
                )
            else:
                raise ValueError(
                    "clique methods available are only nemenyi, bonferroni and holm."
                )
    # If Friedman test is not significant everything has to be one clique
    else:
        if cliques is None:
            cliques = [
                [
                    1,
                ]
                * n_estimators
            ]
    # Step 6 create the diagram:
    # check from where to where the axis has to go
    lowv = min(1, int(math.floor(min(avranks))))
    highv = max(len(avranks), int(math.ceil(max(avranks))))

    # set up the figure
    width = float(width)
    textspace = float(textspace)

    cline = 0.6  # space needed above scale
    linesblank = 1  # lines between scale and text
    scalewidth = width - 2 * textspace

    # calculate needed height
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((n_estimators + 1) / 2) * 0.2 + minnotsignificant + 0.2

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1.0 / height  # height factor
    wf = 1.0 / width

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(1, 0)

    def _lloc(lst, n):
        """
        List location in list of list structure.

        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(lst[0]) + n
        else:
            return n

    def _nth(lst, n):
        n = _lloc(lst, n)
        return [a[n] for a in lst]

    def _hfl(lst):
        return [a * hf for a in lst]

    def _wfl(lst):
        return [a * wf for a in lst]

    def _line(lst, color="k", **kwargs):
        ax.plot(_wfl(_nth(lst, 0)), _hfl(_nth(lst, 1)), color=color, **kwargs)

    # draw scale
    _line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    def _rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    # add ticks to scale
    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        _line([(_rankpos(a), cline - tick / 2), (_rankpos(a), cline)], linewidth=2)

    def _text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    for a in range(lowv, highv + 1):
        _text(
            _rankpos(a),
            cline - tick / 2 - 0.05,
            str(a),
            ha="center",
            va="bottom",
            size=16,
        )

    # sort out lines and text based on whether order is reversed or not
    space_between_names = 0.24
    for i in range(math.ceil(len(avranks) / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        if reverse:
            _line(
                [
                    (_rankpos(avranks[i]), cline),
                    (_rankpos(avranks[i]), chei),
                    (textspace + scalewidth + 0.2, chei),
                ],
                linewidth=linewidth,
                color=colours[i],
            )
            _text(  # labels left side.
                textspace + scalewidth + 0.3,
                chei,
                temp_labels[i],
                ha="left",
                va="center",
                size=16,
                color=colours[i],
            )
            _text(  # ranks left side.
                textspace + scalewidth - 0.3,
                chei - 0.075,
                format(avranks[i], ".4f"),
                ha="left",
                va="center",
                size=10,
                color=colours[i],
            )
        else:
            _line(
                [
                    (_rankpos(avranks[i]), cline),
                    (_rankpos(avranks[i]), chei),
                    (textspace - 0.1, chei),
                ],
                linewidth=linewidth,
                color=colours[i],
            )
            _text(  # labels left side.
                textspace - 0.2,
                chei,
                temp_labels[i],
                ha="right",
                va="center",
                size=16,
                color=colours[i],
            )
            _text(  # ranks left side.
                textspace + 0.4,
                chei - 0.075,
                format(avranks[i], ".4f"),
                ha="right",
                va="center",
                size=10,
                color=colours[i],
            )

    for i in range(math.ceil(len(avranks) / 2), len(avranks)):
        chei = cline + minnotsignificant + (len(avranks) - i - 1) * space_between_names
        if reverse:
            _line(
                [
                    (_rankpos(avranks[i]), cline),
                    (_rankpos(avranks[i]), chei),
                    (textspace - 0.1, chei),
                ],
                linewidth=linewidth,
                color=colours[i],
            )
            _text(  # labels right side.
                textspace - 0.2,
                chei,
                temp_labels[i],
                ha="right",
                va="center",
                size=16,
                color=colours[i],
            )
            _text(  # ranks right side.
                textspace + 0.4,
                chei - 0.075,
                format(avranks[i], ".4f"),
                ha="right",
                va="center",
                size=10,
                color=colours[i],
            )
        else:
            _line(
                [
                    (_rankpos(avranks[i]), cline),
                    (_rankpos(avranks[i]), chei),
                    (textspace + scalewidth + 0.1, chei),
                ],
                linewidth=linewidth,
                color=colours[i],
            )
            _text(  # labels right side.
                textspace + scalewidth + 0.2,
                chei,
                temp_labels[i],
                ha="left",
                va="center",
                size=16,
                color=colours[i],
            )
            _text(  # ranks right side.
                textspace + scalewidth - 0.4,
                chei - 0.075,
                format(avranks[i], ".4f"),
                ha="left",
                va="center",
                size=10,
                color=colours[i],
            )

    # draw lines for cliques
    start = cline + 0.2
    side = -0.02 if reverse else 0.02
    height = 0.1
    i = 1
    for clq in cliques:
        positions = np.where(np.array(clq) == 1)[0]
        min_idx = np.array(positions).min()
        max_idx = np.array(positions).max()
        _line(
            [
                (_rankpos(avranks[min_idx]) - side, start),
                (_rankpos(avranks[max_idx]) + side, start),
            ],
            linewidth=linewidth_sign,
        )
        start += height

    return fig
