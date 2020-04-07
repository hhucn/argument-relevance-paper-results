def calculate_aggregation_with(aggregation_method, score_results):
    """
    Aggregates the score results with aggregation method
    :param aggregation_method: Min, max ,average or sum method
    :param score_results: Results of the score function
    :return: Aggregated scores
    """
    aggregated_scores = {}

    for conclusion_id in score_results.keys():
        arguments_for_conclusion = score_results[conclusion_id]
        argument_score_collection = {}
        for argument in arguments_for_conclusion:
            score_list = []
            argument_id = None
            for argument_unit in argument:
                argument_id = argument_unit[0]
                score = argument_unit[1]
                score_list.append(score)
            argument_score_collection[argument_id] = aggregation_method(score_list)

        aggregated_scores[conclusion_id] = argument_score_collection

    return aggregated_scores


def generate_ranking_from_aggregation(aggregation_dict):
    """
    Replace the score values with the rank of the argument
    aggregation
    :param aggregation_dict: Dictionary of the aggregation results
    :return: Results with ranks
    """
    return_dict = {}
    for conclusion in aggregation_dict.keys():
        score_values = [(key, value) for key, value in aggregation_dict[conclusion].items()]
        unpacked_score_values = [value for key, value in score_values]
        score_values_sorted = sorted(unpacked_score_values, key=lambda x: x, reverse=True)

        ranking = {}
        for value in score_values:
            ranking[value[0]] = score_values_sorted.index(value[1]) + 1

        return_dict[conclusion] = ranking
    return return_dict
