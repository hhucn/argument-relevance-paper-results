from most_premises.MostPremises import calculcate_most_premises
from utils import PATH_ARGUMENT_LIST, PATH_GROUND_TRUTH_LIST
from utils.Arguments import Arguments


def calculate_score(score_function,
                    path_ground_truth_list: str = PATH_GROUND_TRUTH_LIST,
                    path_argument_list: str = PATH_ARGUMENT_LIST):
    """
    Computes the scores between conclusion and premises
    :param score_function: Method to compute the scores between premise and conclusion e.g. similarity
    :param most_premise_function: If true calculate score based on number of premises
    :param path_ground_truth_list:
    :param path_argument_list:
    :return: Score results
    """
    arguments = Arguments(path_ground_truth_list=path_ground_truth_list, path_argument_list=path_argument_list)

    score_results = {}
    for argument in arguments.ground_truth_arguments:
        premises = argument['premises']
        conclusion = argument['conclusion']
        premise_results = []
        for premise in premises:
            if not score_function == calculcate_most_premises:
                premise_results.append(tuple([premise[0], score_function(conclusion, premise)]))
            else:
                if len(premises) > 0:
                    premise_results.append([premise[0], 1.0 - 1.0 / len(premises)])
                else:
                    premise_results.append([premise[0], 0.0])

        if score_results.get(conclusion['conclusion_id']) is None:
            score_results[conclusion['conclusion_id']] = [premise_results]
        else:
            score_results[conclusion['conclusion_id']].append(premise_results)

    return score_results
