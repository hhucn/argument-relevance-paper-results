def min_sentiment_aggregation(sentiments: list) -> float:
    """
    This method takes the minimum value of sentiments for aggregation as mentioned by Wachsmuth et. al. 2017.

    :param sentiments: List of sentiments of which the minimum value should be determined.
    :return: Minimum score of sentiments.
    """
    min_sentiment: float = 1.1
    for sentiment in sentiments:
        if sentiment < min_sentiment:
            min_sentiment = sentiment

    if min_sentiment == 1.1:
        return 0.0
    return min_sentiment


def max_sentiment_aggregation(sentiments: list) -> float:
    """
    This method takes the maximum value of sentiments for aggregation as mentioned by Wachsmuth et. al. 2017.

    :param sentiments: List of sentiments of which the maximum value should be determined.
    :return: Maximum score of sentiments.
    """
    max_sentiment: float = 0.0
    for sentiment in sentiments:
        if sentiment > max_sentiment:
            max_sentiment = sentiment
    return max_sentiment


def min_aggregation(scores: list) -> float:
    """
    This method takes the minimum value for aggregation.

    :param scores: List of scores of which the minimum value should be determined.
    :return: Minimum score.
    """
    return min(scores)


def max_aggregation(scores: list) -> float:
    """
    This method takes the maximum value for aggregation.

    :param scores: List of scores of which the maximum value should be determined.
    :return: Maximum score.
    """
    return max(scores)


def sum_aggregation(scores: list) -> float:
    """
    This method takes the sum value for aggregation.

    :param scores: List of scores of which the sum value should be determined.
    :return: Sum score.
    """
    return sum(scores)


def avg_aggregation(scores: list) -> float:
    """
    This method takes the average value for aggregation.

    :param scores: List of scores of which the average value should be determined.
    :return: Average score.
    """
    return sum(scores) / len(scores)
