import json

import networkx as nx
import pandas as pd

from similarity.WordNetSentenceSimilarity import wordnet_similarity
from utils import PATH_ARGUMENT_UNIT_LIST, PATH_GROUND_TRUTH_JSON


class NetworkXPageRank:
    """
    This class is there to use comparative implementations of PageRank from NetworkX. 
    """

    def __init__(self,
                 page_rank_func,
                 alpha: float,
                 path_argument_unit_list: str = PATH_ARGUMENT_UNIT_LIST,
                 path_ground_truth: str = PATH_GROUND_TRUTH_JSON):
        """
        Initialize the NetworkXPageRank class by adding several parameters.
        
        :param page_rank_func: Specific PageRank function of NetworkX.
        :param alpha: Alpha value for the PageRank function.
        :param path_argument_unit_list: Path to "argument-unit-list.csv".
        :param path_ground_truth: Path to "groundtruth.json" in which the graph-structure of the argument-graph is stored.
        """
        self.path_ground_truth: str = path_ground_truth
        self.path_argument_unit_list: str = path_argument_unit_list

        with open(self.path_ground_truth, "r") as file:
            self.ground_truth: dict = json.load(file)
        self.G: nx.DiGraph = nx.DiGraph()
        for conclusion_id, premise_id in self.ground_truth["edges"]:
            self.G.add_edge(premise_id, conclusion_id)
        self.G_reversed = self.G.reverse(copy=True)
        self.page_rank_dict: dict = dict(page_rank_func(self.G, alpha=alpha))
        self.frequency: list = pd.read_csv(self.path_argument_unit_list)["Frequency"].to_dict().values()
        argument_unit_ids: list = pd.read_csv(self.path_argument_unit_list)["Argument Unit ID"].to_dict().values()
        argument_unit_text = pd.read_csv(self.path_argument_unit_list)["Argument Unit Text"].to_dict().values()
        self.argument_unit_text = dict(list(zip(argument_unit_ids, argument_unit_text)))
        self.frequency: dict = dict(list(zip(argument_unit_ids, self.frequency)))

    def calculate_original_page_rank(self, conclusion: dict, premise: tuple) -> float:
        """
        This calculates the score between the conclusion and one premise by the PageRank of the premise.
        This takes also the longest shortest path from the premise to any leaf of the argumentation into account.
        This catches up:
            "Even though, in cases of doubt, short ar-guments are preferable, we expect that the mostrelevant arguments
            need some space to lay out theirreasoning. However, to investigate such hypothe-ses, ranking functions are
            required that go beyondthe words in an argument and its context."
        by:
            Wachsmuth, H., Potthast, M., Khatib, K.A., Ajjour, Y., Puschmann, J., Qu, J., Dorsch, J., Morari, V.,
            Bevendorff, J., & Stein, B. (2017). Building an Argument Search Engine for the Web. ArgMining@EMNLP.

        The PageRank is divided by the frequency of the premise since the premise may occur in several arguments.

        :param conclusion: Conclusion which should be taken into account.
        :param premise: Premise which should be taken into account.
        :return: Score between the conclusion and one premise.
        """
        premise_id: int = premise[2]
        return self.page_rank_dict[premise_id] / self.frequency[premise_id]

    def take_length_of_premise_into_account(self, conclusion: dict, premise: tuple):
        premise_id: int = premise[2]
        paths = nx.single_source_shortest_path(self.G_reversed, source=premise_id)
        paths = [(key, paths[key]) for key in paths]
        max_path = max(paths, key=lambda x: len(x[1]))
        page_rank = 0
        for i in range(0, len(max_path[1]) - 1):
            node_id_c = max_path[1][i]
            node_id_p = max_path[1][i + 1]
            page_rank += wordnet_similarity(self.argument_unit_text[node_id_c],
                                            self.argument_unit_text[node_id_p]) * (
                                 self.page_rank_dict[node_id_c] / self.frequency[node_id_c])

        return page_rank / len(max_path)

    def take_similarity_of_the_beginning(self, conclusion: dict, premise: tuple):
        """
        This calculates the score between the conclusion and one premise by the PageRank of the premise.
        This takes the similarity between the conclusion and the fist premise of the argumentation into account.
        This catches up:
            "Even though, in cases of doubt, short ar-guments are preferable, we expect that the mostrelevant arguments
            need some space to lay out theirreasoning. However, to investigate such hypothe-ses, ranking functions are
            required that go beyondthe words in an argument and its context."
        by:
            Wachsmuth, H., Potthast, M., Khatib, K.A., Ajjour, Y., Puschmann, J., Qu, J., Dorsch, J., Morari, V.,
            Bevendorff, J., & Stein, B. (2017). Building an Argument Search Engine for the Web. ArgMining@EMNLP.

        The PageRank is divided by the frequency of the premise since the premise may occur in several arguments.

        :param conclusion: Conclusion which should be taken into account.
        :param premise: Premise which should be taken into account.
        :return: Score between the conclusion and one premise.
        """
        premise_id: int = premise[2]
        paths = nx.single_source_shortest_path(self.G_reversed, source=premise_id)
        paths = [(key, paths[key]) for key in paths]
        max_path = max(paths, key=lambda x: len(x[1]))
        first_premise = max_path[1][len(max_path[1]) - 1]

        return wordnet_similarity(conclusion['conclusion_text'].lower(),
                                  self.argument_unit_text[first_premise].lower()) * \
               self.page_rank_dict[premise_id] / self.frequency[premise_id]
