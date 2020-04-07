import pandas as pd

from utils import PATH_ARGUMENT_UNIT_LIST


class PaperPageRank:
    """
    This class is there to use the PageRank Values calculated by Wachsmuth et. al. 2017.
    """

    def __init__(self, path_argument_unit_list: str = PATH_ARGUMENT_UNIT_LIST):
        """
        Initialize the PaperPageRank class by adding the path to the "argument-unit-list.csv".

        :param path_argument_unit_list: Path to "argument-unit-list.csv".
        """
        self.page_rank: list = pd.read_csv(path_argument_unit_list)['PageRank '].to_dict().values()
        self.frequency: list = pd.read_csv(path_argument_unit_list)['Frequency'].to_dict().values()
        self.argument_unit_id: list = pd.read_csv(path_argument_unit_list)['Argument Unit ID'].to_dict().values()
        self.page_rank: dict = dict(list(zip(self.argument_unit_id, self.page_rank)))
        self.frequency: dict = dict(list(zip(self.argument_unit_id, self.frequency)))

    def calculate_paper_page_rank(self, conclusion: dict, premise: list) -> float:
        """
        This calculates the score between the conclusion and one premise by the PageRank of the premise.
        The PageRank is divided by the frequency of the premise since the premise may occur in several arguments.

        :param conclusion: Conclusion which should be taken into account.
        :param premise: Premise which should be taken into account.
        :return: Score between the conclusion and one premise.
        """
        premise_id: int = premise[2]
        return self.page_rank[premise_id] / self.frequency[premise_id]
