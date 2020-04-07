import json

import numpy as np
import pandas as pd

from pagerank import PATH_GENERATED_PAGE_RANK_DATA
from utils import PATH_ARGUMENT_UNIT_LIST, PATH_GROUND_TRUTH_JSON, PATH_NODE_MAPPING_JSON


class OriginalPageRank:
    """
    This class is there to use our own implementation of PageRank.
    """

    def __init__(self, path: str = PATH_GENERATED_PAGE_RANK_DATA,
                 alpha: float = 1.0,
                 epochs: int = 1,
                 suffix: str = "",
                 path_argument_unit_list: str = PATH_ARGUMENT_UNIT_LIST,
                 path_ground_truth: str = PATH_GROUND_TRUTH_JSON,
                 path_node_mapping: str = PATH_NODE_MAPPING_JSON):
        """
        
        :param path: Path where the pre-calculated data of this PageRank class is stored.
        :param alpha: Alpha value for the PageRank function.
        :param epochs: Number of iterations while calculating PageRank. 
        :param suffix: The suffix of the PageRank data. If no suffix is used this class can be used for pre-calculations.
        :param path_argument_unit_list: Path to "argument-unit-list.csv".
        :param path_ground_truth: Path to "groundtruth.json" in which the graph-structure of the argument-graph is stored.
        :param path_node_mapping: Path to "node_mapping.json" in which the node ids are mapped to [0, 28800).
        """
        self.path_argument_unit_list: str = path_argument_unit_list
        self.path_ground_truth: str = path_ground_truth
        self.path_node_mapping: str = path_node_mapping

        try:
            # If no suffix is used this class can be used for pre-calculations of the PageRank values.
            # This can be seen in the __main__ below.
            with open("{}/page_rank_alpha_{}_epochs_{}{}.json".format(path, alpha, epochs, suffix), "r") as file:
                self.page_rank_dict: dict = json.load(file)
        except:
            print("Page Rank will be calculated")

        with open(self.path_ground_truth, "r") as file:
            self.ground_truth: dict = json.load(file)
        with open(self.path_node_mapping, "r") as file:
            self.node_mapping: dict = json.load(file)

        self.alpha: float = alpha
        self.epochs: int = epochs

        self.frequency: list = pd.read_csv(self.path_argument_unit_list)["Frequency"].to_dict().values()
        argument_unit_ids: list = pd.read_csv(self.path_argument_unit_list)["Argument Unit ID"].to_dict().values()
        self.frequency: dict = dict(list(zip(argument_unit_ids, self.frequency)))
        self.amount_nodes: int = len(self.ground_truth["nodes"])

    def calculate_original_page_rank(self, conclusion: dict, premise: tuple) -> float:
        """
        This calculates the score between the conclusion and one premise by the PageRank of the premise.
        The PageRank is divided by the frequency of the premise since the premise may occur in several arguments.

        :param conclusion: Conclusion which should be taken into account.
        :param premise: Premise which should be taken into account.
        :return: Score between the conclusion and one premise.
        """
        premise_id: int = premise[2]
        return self.page_rank_dict[str(premise_id)] / self.frequency[premise_id]

    def __create_adjacency_dict(self) -> dict:
        """
        This method create an dict which stores premises as keys and values as the conclusions drawn from the premise.
        With this dict it is easier and faster to calculate the transitions in the graph.
        :return: Adjacent premises which concludes conclusions.
        """
        adjacency_dict: dict = {}
        for i in range(0, self.amount_nodes):
            adjacency_dict[i] = {
                "concludes": set()
            }
        for conclusion_id, premise_id in self.ground_truth["edges"]:
            # The node mapping must be used since the ids in the groundtruth are not uniform.
            mapped_premise_id: int = self.node_mapping[str(premise_id)]
            mapped_conclusion_id: int = self.node_mapping[str(conclusion_id)]
            adjacency_dict[mapped_premise_id]["concludes"].add(mapped_conclusion_id)
        return adjacency_dict

    def __create_transition_matrix_A(self) -> [np.ndarray, bool]:
        """
        This calculates the matrix A as specified in the formula.
        This matrix represents the transition probability in the graph.
        This transition probability is equivalent to the probability of drawing a conclusion by one premise in the graph.
        Notice: Rows are conclusions, Columns are premises

        :return: Transition matrix A which the transition probabilities in the graph, Boolean if A is column stochastic.
        """
        A: np.ndarray = np.zeros((self.amount_nodes, self.amount_nodes))
        adjacency_dict: dict = self.__create_adjacency_dict()
        dangling_nodes: list = []
        single_nodes: list = []
        for node in adjacency_dict:
            premise_id: int = node
            conclusions: list = adjacency_dict[premise_id]["concludes"]
            # preprocess dangling node
            if len(conclusions) == 0:
                A[:, premise_id] = np.longdouble(1 / self.amount_nodes)
                dangling_nodes += [premise_id]
            # preprocess nodes with exactly one successor
            if len(conclusions) == 1:
                conclusion_id: int = list(conclusions)[0]
                if conclusion_id != premise_id:
                    A[conclusion_id, premise_id] = np.longdouble(1)
                    single_nodes += [premise_id]
        # preprocess remaining nodes
        already_preprocessed_nodes: list = dangling_nodes + single_nodes
        to_be_preprocessed_nodes: list = list(set(range(0, self.amount_nodes)) - set(already_preprocessed_nodes))
        for node in to_be_preprocessed_nodes:
            premise_id: int = node
            conclusions: list = adjacency_dict[premise_id]["concludes"]
            for conclusion_id in conclusions:
                A[conclusion_id, premise_id] = np.longdouble(1 / len(conclusions))

        # normalize A (nodes are mapped ids)
        A = A / np.sum(A, axis=0)
        if not round(sum(np.sum(A, axis=0))) == self.amount_nodes:
            raise Exception("PageRank does not have a valid column sum")
        return A, round(sum(np.sum(A, axis=0))) == self.amount_nodes

    def __calculate_page_rank_for_graph(self) -> [np.ndarray, bool]:
        """
        This method calculates the PageRank vector of the given graph.
        This is the implementation of the mentioned EQS.

        :return: PageRank vector after n epochs, Boolean if this vector sums up to 1.
        """
        v_0: np.ndarray = np.ones(self.amount_nodes) / self.amount_nodes
        one: np.ndarray = np.ones((self.amount_nodes, self.amount_nodes))
        alphaA: np.ndarray = np.multiply(self.alpha, self.__create_transition_matrix_A()[0])
        alphaOne: np.ndarray = np.multiply((1 - self.alpha) / self.amount_nodes, one)
        M: np.ndarray = np.add(alphaA, alphaOne)
        v_n: np.ndarray = v_0
        for n in range(0, self.epochs):
            v_n = np.dot(M, v_n)
        if not round(sum(v_n)) == 1:
            raise Exception("PageRank does not sum to one")
        return v_n, round(sum(v_n)) == 1

    def __write_page_rank_to_nodes(self) -> dict:
        """
        This method writes the PageRank value to the regular argument-unit-id which was used before the mapping to
        uniformed ids.
        :return: Dict of regular argument-unit-ids and their PageRanks.
        """
        page_rank: dict = {}
        node_mapping_inverted: dict = {value: int(key) for key, value in self.node_mapping.items()}
        v_n: np.ndarray = self.__calculate_page_rank_for_graph()[0]
        for i in range(0, self.amount_nodes):
            original_id = node_mapping_inverted[i]
            page_rank[original_id] = v_n[i]
        return page_rank

    def write_page_rank_for_graph_to_file(self, suffix: str = "") -> None:
        """
        This writes the pre-calculated PageRank values to a file.

        :param suffix: Suffix of the file which should be written.
        :return: None
        """
        page_rank: dict = self.__write_page_rank_to_nodes()
        with open("./data/page_rank_alpha_{}_epochs_{}{}.json".format(self.alpha, self.epochs, suffix), "w") as file:
            json.dump(page_rank, file)


if __name__ == "__main__":
    for alpha in np.arange(0, 1.1, 0.1):
        original_page_rank = OriginalPageRank(alpha=alpha, epochs=1)
        print("Alpha {}, Epochs {}".format(round(alpha, 1), 1))
        original_page_rank.write_page_rank_for_graph_to_file(suffix="_fixed_epochs")
        print("Done")
