import csv
from typing import List

import pandas as pd

from utils import PATH_GROUND_TRUTH_LIST, PATH_ARGUMENT_LIST


class Arguments:
    """
    This class collects all the arguments from the various data files
    """
    def __init__(self, path_ground_truth_list: str = PATH_GROUND_TRUTH_LIST,
                 path_argument_list: str = PATH_ARGUMENT_LIST):
        """
        Initializes the arguments object and loads the data
        :param path_ground_truth_list:
        :param path_argument_list:
        """
        argument_list_dataset = self.read_argument_list(path_argument_list=path_argument_list)
        ground_truth_dataset = pd.read_csv(path_ground_truth_list).to_dict()
        conclusion_ids = list(ground_truth_dataset['Conclusion argument unit ID'].values())
        argument_ids = list(ground_truth_dataset['Argument ID'].values())
        argument_and_conclusion_pairs = list(zip(argument_ids, conclusion_ids))
        arguments = self.collect_arguments(argument_list_dataset)
        self.ground_truth_arguments = self.filter_ground_truth_arguments(arguments, argument_and_conclusion_pairs)

    @staticmethod
    def collect_arguments(argument_list_dataset: List) -> List:
        """
        Collect arguments from argument list dataset in list
        :param argument_list_dataset:
        :return: dictionary with arguments
        """
        arguments = []
        for row in argument_list_dataset:
            argument_id = int(row[0])
            conclusion_id = int(row[1])
            conclusion_text = row[2]
            number_of_premises = int(row[3])

            #  Collect premises for argument
            premises_list = []
            for i in range(0, number_of_premises * 2, 2):
                premises_list.append((argument_id, row[5 + i],
                                      int(row[4 + i])))  # 1. argument_id, 2. premise text, 3. premise argument unit id

            # Ignore arguments with this is
            if argument_id not in [441, 3757, 4087]:
                arguments.append({
                    "conclusion": {
                        "conclusion_id": conclusion_id,
                        "conclusion_text": conclusion_text
                    },
                    "premises": premises_list
                })
        return arguments

    @staticmethod
    def read_argument_list(path_argument_list: str = PATH_ARGUMENT_LIST) -> List:
        """
        Read the argument list from argument-list.csv
        :param path_argument_list:
        :return: list of argument data
        """
        data = []
        path = path_argument_list
        with open(path) as csvfile:
            argument_list_file = csv.reader(csvfile, delimiter=',')
            for row in argument_list_file:
                data.append(row)
        data.pop(0)
        return data

    def filter_ground_truth_arguments(self, arguments: List, argument_and_conclusion_pairs: List) -> List:
        """
        Filter arguments which are not in ground truth arguments
        :param arguments:
        :param argument_and_conclusion_pairs:
        :return: ground truth data which are used for ranking
        """
        ground_truth_arguments = []
        for argument in arguments:
            argument_conclusion_id = argument['conclusion']['conclusion_id']
            argument_argument_id = argument['premises'][0][0]
            if (argument_argument_id, argument_conclusion_id) in argument_and_conclusion_pairs:
                ground_truth_arguments.append(argument)
        return ground_truth_arguments
