from flair.embeddings import WordEmbeddings, Sentence, DocumentPoolEmbeddings
from scipy.spatial.distance import cosine

from sentiment.Sentiment import remove_punctuations
from similarity import PATH_GENERATED_EMBEDDINGS_DATA
from similarity.Embeddings import load_embedding


class SemanticSimilarity:
    """
    This class computes the semantic similarity between a conclusion and a premise for ELMo or BERT embedding
    """

    def __init__(self, path_to_embedding: str = PATH_GENERATED_EMBEDDINGS_DATA, file_name: str = ""):
        """
        Initialize the embedding by loading precomputed embedding
        :param embedding_name:  filename of the embedding
        """
        self.embedding = load_embedding(path_to_embedding=path_to_embedding, file_name=file_name)

    def calculate_similarity(self, conclusion, premise):
        """
        Calculates the similarity of the premise to the conclusion
        :param conclusion:
        :param premise:
        :return: similarity score
        """
        argument = self.embedding[str(premise[0])]
        embedded_conclusion = argument[0]
        embedded_premise = argument[1][str(premise[2])]
        return 1.0 - cosine(embedded_conclusion, embedded_premise)


class GloveSemanticSimilarity:
    """
    This class computes the semantic similarity between a conclusion and a premise for a glove embedding
    """

    def __init__(self, remove_punctuation: bool):
        """
        Initialize the model by instantiate a glove embedding
        :param remove_punctuation:
        """
        glove_embedding = WordEmbeddings('glove')
        self.embedding = DocumentPoolEmbeddings([glove_embedding])
        self.remove_punctuation = remove_punctuation

    def calculate_similarity(self, conclusion, premise):
        """
        Calculates the similarity of the premise to the conclusion
        :param conclusion:
        :param premise:
        :return: similarity score
        """

        # Embedding of the conclusion and premise with and without punctuation
        if self.remove_punctuation:
            conclusion_sentence = Sentence(remove_punctuations(conclusion['conclusion_text']))
            premise_sentence = Sentence(remove_punctuations(premise[1]))
        else:
            conclusion_sentence = Sentence(conclusion['conclusion_text'])
            premise_sentence = Sentence(premise[1])
        self.embedding.embed(conclusion_sentence)
        self.embedding.embed(premise_sentence)

        conclusion_embedding = conclusion_sentence.get_embedding().detach().numpy()
        premise_embedding = premise_sentence.get_embedding().detach().numpy()
        return 1.0 - cosine(conclusion_embedding, premise_embedding)
