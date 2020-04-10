# Argument Relevance Reproduction
This repo contains the results of the work "Structure or Content? Towards assessing Argument Relevance".
The data from `/Webis-ArgRank-17-Dataset` is [here](http://argumentation.bplaced.net/arguana/data) available.

# Installation
`Python >= 3.6` is expected.

First, all packages must be installed using:

    $ pip install -r requirements.txt

Afterwards the `spacy` language pack `en` must be installed. For this purpose it should be used:

    $ python -m spacy download en

Make sure this will install `en_core_web_sm==2.1.0`.

# Reproduce

It is very important to follow the steps in the given order, since otherwise the results may differ due to different hardware setups.

1. Run `Groundtruth-Graph.ipynb` to create the argumentation-graph
2. Run `Remapping-Graph.ipynb` to give each node a unique ID in [0, N-1]
3. Run `OriginalPageRank.py` to create all PageRank values for the graph given the alpha-values in [0, 1]
4. Run `Embeddings.py` to generate the BERT and EMLo embeddings
5. Run `NeuronalNetworkSentimentClassification.py` to train the neuronal network for predicting sentiment
6. Run `PageRankAnalysis.ipynb` to generate figure 1 of our paper
7. Run `VisualizationOfResults.ipynb` to generate figure 2 of our paper
8. Run `RankingOfArguments.ipynb` to generate table 1 of our paper
