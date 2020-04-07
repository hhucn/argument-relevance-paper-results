import re

import nltk
import spacy
from pycorenlp import StanfordCoreNLP


def remove_punctuations(sentence: str) -> str:
    """
    This function removes all punctuations from the sentence and returns it
    :param sentence: Premise or conclusion text
    :return:
    """
    for character in sentence:
        if character in ',.!?;:':
            sentence = sentence.replace(character, "")
    return sentence


def paper_tokenizer(sentence: str) -> set:
    """
    Tokenize the input sentence like the program of the "PageRank for Argument Relevance" paper
    :param sentence: Premise or conclusion text
    :return: set of tokens
    """
    sentence: str = remove_punctuations(sentence)
    return set(re.split(' |\n|\t', sentence.lower()))


def nltk_tokenizer(sentence: str) -> set:
    """
    Tokenize the text with the nltk tokenizer
    :param sentence: Premise or conclusion text
    :return: set of tokens
    """
    return set(nltk.word_tokenize(sentence))


def spacy_tokenizer(sentence: str) -> set:
    """
    Tokenize the text with the spacy tokenizer
    :param sentence: Premise or conclusion text
    :return: set of tokens
    """
    tokenizer = spacy.load("en_core_web_sm")
    return set(tokenizer(sentence))


def stanford_tokenizer(sentence: str) -> set:
    """
    Tokenize the sentence with the stanford coreNLP tokenizer
    :param sentence: Premise or conclusion text
    :return: set of tokens
    """
    nlp = StanfordCoreNLP('http://localhost:9000')
    properties = {
        'annotators': 'tokenize',
        'outputFormat': 'json'
    }
    annotated_text: dict = nlp.annotate(sentence, properties=properties)

    results: list = []
    for word in annotated_text['tokens']:
        results.append(word.get('word'))

    return set(results)
