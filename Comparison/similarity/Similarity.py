from utils.PreProcessing import remove_punctuations, paper_tokenizer


def caclulate_jaccard_similarity(conclusion, premise):
    """
    Calculate jacard similarity between conclusion and premise
    :param conclusion:
    :param premise:
    :return: Jacard similarity score
    """
    conclusion_tokens = paper_tokenizer(remove_punctuations(conclusion['conclusion_text']).lower())
    premise_tokens = paper_tokenizer(remove_punctuations(premise[1]).lower())

    intersection = conclusion_tokens.intersection(premise_tokens)
    return len(intersection) / (len(premise_tokens) + len(conclusion_tokens) - len(intersection))
