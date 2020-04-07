import re
from typing import Union

import nltk
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic

from sentiment.Sentiment import remove_punctuations

brown_ic = wordnet_ic.ic('ic-brown.dat')


def idf(documents: list) -> dict:
    """
    This method calculate a idf dict where the keys are words and values are idf-values.
    The words are extracted from the sentences in the conclusion in connection with the premise.

    IDF(word) = #sentences/#(sentences which contains the word)

    :param documents: List of the texts of conclusions and premises.
    :return: Dict with idf per word.
    """
    sentences: list = list()
    for document in documents:
        sentences += nltk.sent_tokenize(document, language='english')
    words: set = set()
    for i in range(0, len(sentences)):
        sentences[i] = remove_punctuations(sentences[i])
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            words.add(word)
    idf_list: dict = dict()
    for word in words:
        word_in_lists = [(word in sentence) for sentence in sentences]
        len_sentences = len(sentences)
        idf_list[word] = len_sentences / sum(word_in_lists)
    return idf_list


def get_wordnet_tag(tag: str) -> Union[str]:
    """
    Transform the tag to a Penn-Treebank Tag

    :param tag: Wordnet Tag
    :return: Penn-Treebank Tag
    """
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag[0], None)


def reduce_document_to_T(document: str) -> list:
    """
    This method calculates a document which represents the text of a argument-unit into a structure for max_sum.
    For this the document is splited into sentences.
    Then each word of each sentence is turned into a list of synsets by its penn-treebank-tag (This tag is created
    by POS-Tagging).
    Then only words with a valid tag and with at least one synset are taken into account.

    :param document: Text of the argument-unit which is taken into account.
    :return: Representation of the Text of the argument-unit which is taken into account.
    """
    sentences: list = nltk.sent_tokenize(document, language='english')

    pos_tuples: set = set()
    for sentence in sentences:
        sentence: str = remove_punctuations(sentence)
        for pos_tuple in nltk.pos_tag(nltk.word_tokenize(sentence, language='english')):
            pos_tuples.add(pos_tuple)

    T: list = list()
    for word, tag in pos_tuples:
        tag: str = get_wordnet_tag(tag)
        if tag:
            synsets: list = wordnet.synsets(word, tag)
            if synsets:
                # take the alphabetically first synset
                T.append((word, tag, [min(synsets)]))  # this can also be all synsets or min or max alone
    return T


def max_sum(w: list, T: list) -> float:
    """
    As mentioned by:
        Mihalcea R. , Corley C. & Strapparava C. (2006).
        Corpus-based and Knowledge-based Measures of Text Semantic Similarity.
        In Proceedings, The Twenty-First National Conference on Artificial Intelligence
            and the Eighteenth Innovative Applications of Artificial Intelligence Conference,
        July 16-20, 2006, Boston, Massachusetts, USA.

    :param w: A Word represented as a triple of the word, penn-treebank-tag, list of its synsets.
    :param T: List representation of a text in triples like w.
    :return: max_sum(w, T)
    """
    w_synsets: list = w[2]
    similarities: set = set()
    similarities.add(0)
    for t in T:
        t_synsets = t[2]
        for w_synset in w_synsets:
            for t_synset in t_synsets:
                try:
                    similarity: float = wordnet.wup_similarity(w_synset, t_synset, brown_ic)
                    if similarity is None:
                        similarities.add(0)
                    else:
                        similarities.add(similarity)
                except:
                    similarities.add(0)
    return max(similarities)


def wordnet_knowledge_similarity(conclusion: dict, premise: list) -> float:
    """
    This calculates the score between the conclusion and one premise by the weighted knowledge-similarity.

    :param conclusion: Conclusion which should be taken into account.
    :param premise: Premise which should be taken into account.
    :return: Score between the conclusion and one premise.
    """
    return wordnet_similarity(conclusion['conclusion_text'].lower(), premise[1].lower())


def wordnet_knowledge_similarity_averaged(conclusion: dict, premise: list) -> float:
    """
    This calculates the score between the conclusion and one premise by the averaged knowledge-similarity.

    :param conclusion: Conclusion which should be taken into account.
    :param premise: Premise which should be taken into account.
    :return: Score between the conclusion and one premise.
    """
    return wordnet_similarity_average(conclusion['conclusion_text'].lower(), premise[1].lower())


def wordnet_similarity(sentence_a: str, sentence_b: str) -> float:
    """
    This method calculates the knowledge-based similarity between two sentences as mentioned by:
        Mihalcea R. , Corley C. & Strapparava C. (2006).
        Corpus-based and Knowledge-based Measures of Text Semantic Similarity.
        In Proceedings, The Twenty-First National Conference on Artificial Intelligence
            and the Eighteenth Innovative Applications of Artificial Intelligence Conference,
        July 16-20, 2006, Boston, Massachusetts, USA.

    :param sentence_a: A text or sentence which can contain more then one sentence.
    :param sentence_b: Like sentence_b.
    :return: sim(T1, T2)
    """
    sentence_a: str = remove_unwanted_chars(sentence_a)
    sentence_b: str = remove_unwanted_chars(sentence_b)
    T_a: list = reduce_document_to_T(sentence_a)
    T_b: list = reduce_document_to_T(sentence_b)

    idf_values: dict = idf([sentence_a, sentence_b])

    counter_a: float = 0.0
    denominator_a: float = 0.0
    for t_a in T_a:
        word_a: str = t_a[0]
        counter_a += max_sum(t_a, T_b) * idf_values[word_a]
        denominator_a += idf_values[word_a]

    counter_b: float = 0.0
    denominator_b: float = 0.0
    for t_b in T_b:
        word_b: str = t_b[0]
        counter_b += max_sum(t_b, T_a) * idf_values[word_b]
        denominator_b += idf_values[word_b]

    if denominator_a == 0:
        denominator_a = 1
    if denominator_b == 0:
        denominator_b = 1
    sum_a: float = (counter_a / denominator_a)
    sum_b: float = (counter_b / denominator_b)
    return 0.5 * (sum_a + sum_b)


def wordnet_similarity_average(sentence_a: str, sentence_b: str) -> float:
    """
    This method works similar to the method  mentioned by:
        Mihalcea R. , Corley C. & Strapparava C. (2006).
        Corpus-based and Knowledge-based Measures of Text Semantic Similarity.
        In Proceedings, The Twenty-First National Conference on Artificial Intelligence
            and the Eighteenth Innovative Applications of Artificial Intelligence Conference,
        July 16-20, 2006, Boston, Massachusetts, USA.
    However, it only considers the similarity of the conclusion in the direction of the premise.
    Therefore all max_sim are averaged from T1 to T2. This reflects a measure of the quality of the reason relation
    and indicates how well a premise justifies a conclusion.

    :param sentence_a: A text or sentence which can contain more then one sentence.
    :param sentence_b: Like sentence_b.
    :return: Averaged similarity between two sentences with the focus of the first sentence.
    """
    sentence_a: str = remove_unwanted_chars(sentence_a)
    sentence_b: str = remove_unwanted_chars(sentence_b)
    T_a: list = reduce_document_to_T(sentence_a)
    T_b: list = reduce_document_to_T(sentence_b)

    counter_a: float = 0.0
    denominator_a: float = 0.0
    for t_a in T_a:
        m_sum: float = max_sum(t_a, T_b)
        if m_sum != 0:
            counter_a += m_sum
            denominator_a += 1

    if denominator_a == 0:
        denominator_a = 1

    return counter_a / denominator_a


def remove_unwanted_chars(string: str) -> str:
    """
    This removes unwanted chars and keeps all alphanumerical/numerical values as well as new-lines and phrase-dots.

    :param string: String to be cleaned.
    :return: Cleaned string which only contains alphanumerical/numerical values as well as new-lines and phrase-dots.
    """
    return re.sub('[^a-zA-Z0-9 \n\.]', '', string)
