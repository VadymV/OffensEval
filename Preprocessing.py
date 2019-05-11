"""
Preprocessing of the data.

@author vadym.gryshchuk vadym.gryshchuk@protonmail.com
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import re
import io
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText

nltk.download('stopwords')
nltk.download('wordnet')

pd.options.mode.chained_assignment = None  # default='warn'

REDUNDANT_INFO = ['@USER', 'URL']

EMOJI_PATTERN = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"
    u"(\ud83c[\udf00-\uffff])|"
    u"(\ud83d[\u0000-\uddff])|"
    u"(\ud83d[\ude80-\udeff])|"
    u"(\ud83c[\udde0-\uddff])"
    "+", flags=re.UNICODE)


def lemmatize(x):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(x, pos='v')


class Preprocessing:

    @staticmethod
    def remove_unnecessary_information(x):
        """
        Remove unnecessary information from a string.
        :param x: A string.
        :return: A cleaned string.
        """
        for i in REDUNDANT_INFO:
            x = x.replace(i, '')

        # Remove emojis.
        x = EMOJI_PATTERN.sub(r'', x)

        # Word tokenizer.
        tokenizer = RegexpTokenizer(r'\w+')
        x = tokenizer.tokenize(x)

        # Remove stop words.
        stop_words = set(stopwords.words('english'))
        x = [w for w in x if w not in stop_words]
        x = [w for w in x if not w.isdigit()]
        x = [w for w in x if w not in string.punctuation]
        x = [lemmatize(w) for w in x]

        return x

    @staticmethod
    def load_embeddings(file_name):
        """
        Load the embeddings.
        :param file_name: Path to embeddings.
        :return: Embeddings.
        """
        fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        print("Number of words in FastText embeddings: ", n, " Dimension:", d)
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            representation = np.asarray(tokens[1:], dtype='float32')
            data[word] = representation
        fin.close()
        return data

    @staticmethod
    def load_data(file_name):
        """
        Load data from disk.
        :param file_name: Path to the file.
        :return:
        """
        return pd.read_csv(file_name, sep="\t")

    @staticmethod
    def filter_data(data, subtask='subtask_a'):
        """
        Filter data by removing redundant information from tweets.
        :param data: Data as a pandas data frame.
        :return: Filtered data.
        """

        data.dropna(subset=[subtask], inplace=True)

        print("Labels: ", data[subtask].unique())

        if subtask == 'subtask_a':
            y = data[["subtask_a"]]
            y['subtask_a'].replace(to_replace=['OFF', 'NOT'], value=[0, 1], inplace=True)
        elif subtask == 'subtask_b':
            zero_category = data[subtask] == 'UNT'
            zero_category_data = data[zero_category]
            data = data.append([zero_category_data] * 7, ignore_index=True)
            y = data[["subtask_b"]]
            y['subtask_b'].replace(to_replace=['UNT', 'TIN'], value=[0, 1], inplace=True)
        elif subtask == 'subtask_c':
            one_category = data[subtask] == 'GRP'
            one_category_data = data[one_category]
            data = data.append([one_category_data] * 1, ignore_index=True)

            two_category = data[subtask] == 'OTH'
            two_category_data = data[two_category]
            data = data.append([two_category_data] * 5, ignore_index=True)

            y = data[["subtask_c"]]
            y['subtask_c'].replace(to_replace=['IND', 'GRP', 'OTH'], value=[0, 1, 2], inplace=True)

        X = data[["tweet"]]
        X['tweet'] = X['tweet'].map(lambda x: Preprocessing.remove_unnecessary_information(x))

        print(y[subtask].value_counts())

        return pd.Series(X['tweet']), y.values

    @staticmethod
    def filter_test_data(data):
        """
        Filter data by removing redundant information from tweets.
        :param data: Data as a pandas data frame.
        :return: Filtered data.
        """

        X = data[["tweet"]]

        X['tweet'] = X['tweet'].map(lambda x: Preprocessing.remove_unnecessary_information(x))

        return pd.Series(data['id']), pd.Series(X['tweet'])

    @staticmethod
    def prepare_data(X_train, X_test, max_tweet_length):
        """
        Prepare data by tokenizing it.
        :param X_train: Train data as an ndarray.
        :param X_test: TestA data as an ndarray.
        :param max_tweet_length: A maximum length of a tweet.
        :return: Padded train/test data, mapping of words to the number of texts they appeared, mapping of words to indices
        """

        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
        tokenizer.fit_on_texts(X_train.ravel())
        train_words_to_indices = tokenizer.texts_to_sequences(X_train.ravel())
        test_words_to_indices = tokenizer.texts_to_sequences(X_test.ravel())

        # Add zeroes to to the tweet, if its length less than max_tweet_length.
        train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_words_to_indices, maxlen=max_tweet_length,
                                                                     padding='post', truncating='post')
        test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_words_to_indices, maxlen=max_tweet_length,
                                                                    padding='post', truncating='post')

        print("Shape of the train data: ", train_padded.shape)
        print("Shape of the test data: ", test_padded.shape)

        # len(tokenizer.word_docs) + 2, because of UNKNOWN and PAD.
        return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 3

    @staticmethod
    def create_embedding_matrix(word2idx, dimension, embeddings_file_name):
        """
        Create an embedding matrix.
        :param word2idx: A mapping from a word to an index.
        :param dimension: A dimension of embeddings.
        :param embeddings_file_name: Path to the file of GloVe embeddings.
        :return: An embedding matrix.
        """
        max_words = len(word2idx) + 1
        embedding_matrix = np.zeros((max_words, dimension))

        # Load GloVe embeddings.
        embeddings_data = Preprocessing.load_embeddings(embeddings_file_name)

        zeros = 1
        for word, index in word2idx.items():
            embedding_vector = embeddings_data.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                zeros += 1

        print("Shape of the embedding matrix: ", embedding_matrix.shape)
        print("{} words are not found".format(zeros))

        return embedding_matrix

    @staticmethod
    def get_tweet_embedding(data):

        model = FastText(size=300, window=5, min_count=1, seed=42, workers=2, alpha=0.025, min_alpha=0.00025)

        docs = []

        for index, tweet in data.iteritems():
            words = word_tokenize(tweet.lower())
            docs.append(words)

        model.train(docs, total_examples=model.corpus_count, epochs=10)
        return np.asarray([model.wv[tweet] for i, tweet in data.iteritems()])

    @staticmethod
    def map_indices_to_embeddings(x_data, y_data, embedding_matrix):

        mapped_data = np.zeros((x_data.shape[0] * x_data.shape[1], embedding_matrix.shape[1]))
        mapped_labels = np.zeros((mapped_data.shape[0], 1))

        index = 0
        for r in range(0, x_data.shape[0]):

            # Iterate over rows
            row = x_data[r, :]

            for c in range(0, row.size):
                # Iterate over columns
                embedding = embedding_matrix[x_data[r, c]]
                for e in range(0, embedding_matrix.shape[1]):
                    mapped_data[index, e] = embedding[e]
                mapped_labels[index, 0] = y_data[r, 0]
                index += 1

        return mapped_data, mapped_labels




