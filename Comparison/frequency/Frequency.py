from pagerank.PaperPageRank import PaperPageRank
from utils import PATH_ARGUMENT_UNIT_LIST


class Frequency:
    """
    This class is for the Frequency Ranking.
    """

    def __init__(self, path_argument_unit_list: str = PATH_ARGUMENT_UNIT_LIST):
        """
        Initialize the Frequency class by adding the path to the "argument-unit-list.csv".

        :param path_argument_unit_list: Path to "argument-unit-list.csv".
        """
        self.frequency: dict = PaperPageRank(path_argument_unit_list=path_argument_unit_list).frequency

    def get_frequency_score(self, conclusion: dict, premise: tuple) -> int:
        """
        This calculates the score between the conclusion and one premise by the frequency of the premise.

        :param conclusion: Conclusion which should be taken into account.
        :param premise: Premise which should be taken into account.
        :return: Score between the conclusion and one premise.
        """
        return int(self.frequency[premise[2]])
