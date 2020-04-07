import math
import random

import pandas as pd
from scipy.stats import stats

from randomized.RandomScore import calculate_random_score
from utils import PATH_GROUND_TRUTH_LIST, PATH_ARGUMENT_LIST
from utils.Aggregation import min_aggregation
from utils.Ranking import calculate_aggregation_with, generate_ranking_from_aggregation
from utils.Score import calculate_score


def collect_baseline_ranking(path_ground_truth_list: str = PATH_GROUND_TRUTH_LIST):
    """
    This function collects the baseline ranks with corresponding argument id
    :param path_ground_truth_list:  filepath
    :return:
    """
    ground_truth_dataset = pd.read_csv(path_ground_truth_list).to_dict()
    argument_ids = list(ground_truth_dataset['Argument ID'].values())
    ranks = list(ground_truth_dataset['Aggregate rank'].values())
    return list(zip(argument_ids, ranks))


def calculate_kendall_correlation(score_function, aggregation_function,
                                  path_ground_truth_list: str = PATH_GROUND_TRUTH_LIST,
                                  path_argument_list: str = PATH_ARGUMENT_LIST):
    """
    This function calls the score function and computes the with the results the ranking and then calculates the
    kendall tau value with the baseline ranking
    :param score_function: Functions which computes the score value like jacards similarity
    :param aggregation_function: This function collects the max, min, average oder sum value for an argument
    :param most_premises_function: If set to true score will be calculated with number of premises
    :param random_score_function: If set to true score will be drawn of uniform distribution
    :param path_ground_truth_list:
    :param path_argument_list:
    :return: kendall tau value and dictionary with tau values for all conclusions
    """
    kendall_tau_results = []
    # Calculate score values with different score function
    score_results = calculate_score(score_function, path_ground_truth_list=path_ground_truth_list,
                                    path_argument_list=path_argument_list)
    # Aggregate results with min, max, sum and average method
    aggregated_score_results = calculate_aggregation_with(aggregation_function, score_results)
    # Calculate score values with uniform correlation
    if score_function == calculate_random_score:
        random.seed(114)  # 12 15 17s
        for conclusion_id in aggregated_score_results.keys():
            for argument_id in aggregated_score_results[conclusion_id].keys():
                aggregated_score_results[conclusion_id][argument_id] = random.uniform(0, 1)
    score_ranking = generate_ranking_from_aggregation(aggregated_score_results)
    # Collect baseline ranking from ground-truth-list.csv
    baseline_ranking = collect_baseline_ranking(path_ground_truth_list)
    baseline_ranking_dict = {}
    for rank in baseline_ranking:
        baseline_ranking_dict[rank[0]] = rank[1]
    # Calculate tau values and collect values in dict
    tau_conclusion_dict = {}
    for conclusion_id in score_ranking:
        ranking = score_ranking[conclusion_id]
        baseline_list = []
        scores_list = []
        for argument_id in ranking.keys():
            baseline_list.append(baseline_ranking_dict[argument_id])
            scores_list.append(ranking[argument_id])
        tau, p_value = stats.kendalltau(baseline_list, scores_list)
        if math.isnan(tau):
            tau = 0.0
        kendall_tau_results.append(tau)
        tau_conclusion_dict[conclusion_id] = tau
    return round(sum(kendall_tau_results) / len(kendall_tau_results), 2), tau_conclusion_dict


def compute_best_and_worst(page_rank_approachs, baselines):
    """
    This functions computes the best and worst counter for the comparision of the page rank results and all other
    approaches
    :param page_rank_approachs: The kendall tau results for page rank
    :param baselines: The kendall tau results for the different approaches like frequency etc.
    :return: input data with best and worst counter
    """
    for conclusion_id in baselines[0][0].keys():
        max_value = -1.1
        min_value = 1.1
        # Find max and min value
        for page_rank_approache in page_rank_approachs:
            value = page_rank_approache[0][conclusion_id]
            if value > max_value:
                max_value = value
            if value < min_value:
                min_value = value

        for baseline_approach in baselines:
            value = baseline_approach[0][conclusion_id]
            if value > max_value:
                max_value = value
            if value < min_value:
                min_value = value

        # Increase best and worst counter
        for j in range(len(page_rank_approachs)):
            if math.isclose(page_rank_approachs[j][0][conclusion_id], max_value):
                page_rank_approachs[j][1] += 1
            if math.isclose(page_rank_approachs[j][0][conclusion_id], min_value):
                page_rank_approachs[j][2] += 1

        for i in range(len(baselines)):
            if math.isclose(baselines[i][0][conclusion_id], max_value):
                baselines[i][1] += 1
            if math.isclose(baselines[i][0][conclusion_id], min_value):
                baselines[i][2] += 1

    return page_rank_approachs, baselines
