import collections
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from sentiment import PATH_SENTIWORDNET
from utils.PreProcessing import remove_punctuations


class SentiWordNet:
    """
    This class calculates the sentiment with SentiWordNet
    """
    def __init__(self, path_sentiwordnet: str = PATH_SENTIWORDNET):
        """
        Initializes the SentiWordnet Class by loading the data
        :param path_sentiwordnet:
        """
        self.sentiWordNet_positives, self.sentiWordNet_negatives = parse_SentiWordNet_data(
            path_sentiwordnet=path_sentiwordnet)

    def calculate_sentiment(self, conclusion, premise):
        """
        Calculates the sentiment of the premise
        :param conclusion:
        :param premise:
        :return:
        """
        words = collections.Counter(re.split(' |\n|\t', remove_punctuations(premise[1].lower())))
        sentiment = 0.0
        for word in words:
            if self.sentiWordNet_positives.get(word) is not None:
                sentiment += words[word] * self.sentiWordNet_positives[word]
            if self.sentiWordNet_negatives.get(word) is not None:
                sentiment -= words[word] * self.sentiWordNet_negatives[word]
        return sentiment


def read_SentiWordNet(path_sentiwordnet: str = PATH_SENTIWORDNET):
    """
    Reads the SentiWordNet data
    :param path_sentiwordnet:
    :return:
    """
    path = path_sentiwordnet
    data = []
    f = open(path)
    for line in f:
        data.append(line)
    return data


def parse_SentiWordNet_data(path_sentiwordnet: str = PATH_SENTIWORDNET):
    """
    Parses the SentiWordNet data to dictionary's
    :param path_sentiwordnet:
    :return:
    """
    senti_WordNet_positives = {}
    senti_WordNet_negatives = {}

    senti_word_net_data = read_SentiWordNet(path_sentiwordnet=path_sentiwordnet)
    for row in senti_word_net_data:
        if row[0] == '#' or row[0] == '\t':
            continue
        data = row.split('\t')
        pos_score = float(data[2])
        neg_score = float(data[3])

        # Split synonym words
        words_with_numbers = data[4].split(" ")
        for word in words_with_numbers:
            parts = word.split("#")
            if parts[1] != "1":
                continue
            part_word = parts[0].lower()
            senti_WordNet_positives[part_word] = pos_score
            senti_WordNet_negatives[part_word] = neg_score
    return senti_WordNet_positives, senti_WordNet_negatives


def calculate_tokenwise_sentiment(conclusion, premise):
    """
    Calculates the sentiment tokenwise
    :param conclusion:
    :param premise:
    :return:
    """
    words = collections.Counter(re.split(' |\n|\t', remove_punctuations(premise[1].lower())))
    sentiment = 0.0
    for word in words:
        sentiment += words[word] * TextBlob(
            text=word).sentiment.polarity  # SentimentIntensityAnalyzer().polarity_scores(text=word)['compound']
    return sentiment


def calculate_nltk_sentiment(conclusion, premise):
    """
    Calculates the sentiment with the NLTK SentimentIntesityAnalyzer
    :param conclusion:
    :param premise:
    :return:
    """
    cleaned_premise_text = remove_punctuations(premise[1].lower())
    return SentimentIntensityAnalyzer().polarity_scores(text=cleaned_premise_text)['compound']


def calculate_textblob_sentiment(conclusion, premise):
    """
    Calculates the sentiment with TextBlob
    :param conclusion:
    :param premise:
    :return:
    """
    cleaned_premise_text = remove_punctuations(premise[1].lower())
    return TextBlob(text=cleaned_premise_text).sentiment.polarity
