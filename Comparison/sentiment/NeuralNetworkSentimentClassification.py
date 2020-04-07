import random
import time

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data, datasets

from sentiment import PATH_MODEL

# Source: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb
def generate_bigrams(x):
    """
    Generates bigrams for the data preprocessing
    :param x: token
    """
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


class FastText(nn.Module):
    """
    Architecture of the neural network
    """

    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        """
        Initialize the model
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        """
        Calculates the forward step in the NN
        :param text:
        :return:
        """
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        embedded = embedded.permute(1, 0, 2)

        # embedded = [batch size, sent len, emb dim]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        # pooled = [batch size, embedding_dim]

        return self.fc(pooled)


def binary_accuracy(preds, y):
    """
    Calculates the binary accuracy
    :param preds: Prediction
    :param y: label to compare with prediction
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def binary_precision(preds, y):
    """
    Calculates the binary precision
    :param preds: Prediction
    :param y: Label to compare with prediction
    """
    true_positiv = np.zeros(preds.shape)
    false_positiv = np.zeros(preds.shape)

    rounded_preds = torch.round(torch.sigmoid(preds)).float()
    for i in range(rounded_preds.shape[0]):
        if int(rounded_preds[i]) == 1 and int(y[i]) == 1:
            true_positiv[i] = 1.0
        if int(rounded_preds[i]) == 1 and int(y[i]) == 0:
            false_positiv[i] = 1.0
    true_positiv = torch.Tensor(true_positiv)
    false_positiv = torch.Tensor(false_positiv)
    precision = true_positiv.sum() / (true_positiv.sum() + false_positiv.sum())
    return precision


def calculate_metrics(preds, y):
    """
    Calculates the precision, accuracy and F1-Score
    :param preds: Prediction
    :param y: Label to compare prediction with
    :return:
    """
    true_positiv = np.zeros(preds.shape)
    true_negativ = np.zeros(preds.shape)
    false_positiv = np.zeros(preds.shape)
    false_negativ = np.zeros(preds.shape)

    rounded_preds = torch.round(torch.sigmoid(preds)).float()
    for i in range(rounded_preds.shape[0]):
        if int(rounded_preds[i]) == 1 and int(y[i]) == 1:
            true_positiv[i] = 1.0
        if int(rounded_preds[i]) == 1 and int(y[i]) == 0:
            false_positiv[i] = 1.0
        if int(rounded_preds[i]) == 0 and int(y[i]) == 1:
            false_negativ[i] = 1.0
        if int(rounded_preds[i]) == 0 and int(y[i]) == 0:
            true_negativ[i] = 1.0

    true_positiv = torch.Tensor(true_positiv).sum()
    false_positiv = torch.Tensor(false_positiv).sum()
    false_negativ = torch.Tensor(false_negativ).sum()
    true_negativ = torch.Tensor(true_negativ).sum()

    precision = true_positiv / (true_positiv + false_positiv)
    recall = true_positiv / (true_positiv + false_negativ)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def train(model, iterator, optimizer, criterion):
    """
    This function computes one iteration of the training of  the neural network
    :param model:
    :param iterator:
    :param optimizer:
    :param criterion:
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1_score = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        precision, recall, f1_score = calculate_metrics(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_precision += precision.item()
        epoch_recall += recall.item()
        epoch_f1_score += f1_score.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_precision / len(iterator), epoch_recall / len(
        iterator), epoch_f1_score / len(iterator)


def evaluate(model, iterator, criterion):
    """
    Evaluates the last iteration of the training process
    :param model:
    :param iterator:
    :param criterion:
    :return: the results of the metrics
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1_score = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            precision, recall, f1_score = calculate_metrics(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_precision += precision.item()
            epoch_recall += recall.item()
            epoch_f1_score += f1_score.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_precision / len(iterator), epoch_recall / len(
        iterator), epoch_f1_score / len(iterator)


def epoch_time(start_time, end_time):
    """
    Calculates the elapsed time of one iteration of the training
    :param start_time:
    :param end_time:
    :return: elapsed minutes and seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def start_training(model, train_iterator, valid_iterator, optimizer, criterion):
    """
    Computes the neural network model
    :param model:
    :param train_iterator:
    :param valid_iterator:
    :param optimizer:
    :param criterion:
    """
    N_EPOCHS = 5

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc, train_precision, train_recall, train_f1_score = train(model, train_iterator, optimizer,
                                                                                     criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1_score = evaluate(model, valid_iterator,
                                                                                        criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Precision: {train_precision * 100:.2f}% | Train Recall: {train_recall * 100:.2f}% | Train F1-Score: {train_f1_score * 100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% | Valid Precision: {valid_precision * 100:.2f}% | Valid Recall: {valid_recall * 100:.2f}% | Valid F1-Score: {valid_f1_score * 100:.2f}%')

    torch.save(model.state_dict(), "./data/nn_sentiment_model.pt")


def train_neural_network():
    """
    Initializes all necessary variables for the training process and calls training function
    """
    SEED = 1234

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field(tokenize='spacy', preprocessing=generate_bigrams)
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 1
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    start_training(model, train_iterator, valid_iterator, optimizer, criterion)


class NeuralNetworkSentiment:
    """
    This class is for the sentiment ranking with a neural network.
    """

    def __init__(self, model_path: str = PATH_MODEL):
        """
        Initialize the neural network model class and loads the pretrained model
        """
        TEXT = data.Field(tokenize='spacy', preprocessing=generate_bigrams)
        LABEL = data.LabelField(dtype=torch.float)
        MAX_VOCAB_SIZE = 25_000

        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        train_data, valid_data = train_data.split(random_state=random.seed(1234))

        TEXT.build_vocab(train_data,
                         max_size=MAX_VOCAB_SIZE,
                         vectors="glove.6B.100d",
                         unk_init=torch.Tensor.normal_)
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        OUTPUT_DIM = 1
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        device = torch.device('cpu')

        self.model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.TEXT = TEXT

    def predict_sentiment(self, conclusion, premise):
        """
        Predicts the sentiment of the given premise
        :param conclusion:
        :param premise:
        :return: Returns the sentiment score of the premise
        """
        nlp = spacy.load('en')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(premise[1])])
        indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        return torch.sigmoid(self.model(tensor)).item()


if __name__ == '__main__':
    # en_core_web_sm==2.1.0
    train_neural_network()
