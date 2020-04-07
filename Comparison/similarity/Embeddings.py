import json

from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, ELMoEmbeddings, BertEmbeddings

from similarity import PATH_GENERATED_EMBEDDINGS_DATA
from utils.Arguments import Arguments
from utils.PreProcessing import remove_punctuations


def save_embedding(embedding, file_name: str):
    """
    Saves the given embedding to json file
    :param embedding:
    :param file_name:
    """
    embedding_path = PATH_GENERATED_EMBEDDINGS_DATA + file_name
    with open(embedding_path, 'w') as fp:
        json.dump(embedding, fp)


def load_embedding(path_to_embedding: str = PATH_GENERATED_EMBEDDINGS_DATA, file_name: str = ""):
    """
    Loads the embedding from file
    :param file_name:
    :param path_to_embedding:
    """
    with open(path_to_embedding + file_name, 'r') as f:
        json_data = json.load(f)
    return json_data


def compute_embedding(embedding, remove_punctuation: bool, file_name: str):
    """
    Computes the embedding with given model for all arguments
    :param embedding: Model
    :param remove_punctuation: Bool to indicate if punctuation should be removed
    :param file_name:
    """
    arguments = Arguments()
    document_embedding = DocumentPoolEmbeddings([embedding])

    embedded_arguments = {}

    for argument in arguments.ground_truth_arguments:
        premises = argument['premises']
        conclusion = argument['conclusion']

        conclusion_text = conclusion['conclusion_text']
        if remove_punctuation:
            conclusion_text = remove_punctuations(conclusion_text)
        conclusion_sentence = Sentence(conclusion_text)
        document_embedding.embed(conclusion_sentence)
        embedded_conclusion = conclusion_sentence.get_embedding().detach().numpy().tolist()

        embedded_premises = {}
        argument_uid = None

        for premise in premises:
            premise_text = premise[1]
            if remove_punctuation:
                premise_text = remove_punctuations(premise_text)
            premise_sentence = Sentence(premise_text)
            document_embedding.embed(premise_sentence)
            embedded_premise = premise_sentence.get_embedding().detach().numpy().tolist()
            embedded_premises[premise[2]] = embedded_premise
            argument_uid = premise[0]
        embedded_arguments[argument_uid] = [embedded_conclusion, embedded_premises]

        save_embedding(embedded_arguments, file_name)


if __name__ == '__main__':
    elmo_embedding = ELMoEmbeddings()
    compute_embedding(elmo_embedding, remove_punctuation=True, file_name="elmo_embeddings_without_punctuation.json")
    compute_embedding(elmo_embedding, remove_punctuation=False, file_name="elmo_embeddings_with_punctuation.json")

    bert_embedding = BertEmbeddings()
    compute_embedding(bert_embedding, remove_punctuation=True, file_name="bert_embeddings_without_punctuation.json")
    compute_embedding(bert_embedding, remove_punctuation=False, file_name="bert_embeddings_with_punctuation.json")
