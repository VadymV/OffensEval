# -*- coding: utf-8 -*-
"""
A model, which classifies tweets into two categories.

@author vadym.gryshchuk vadym.gryshchuk@protonmail.com
"""

import tensorflow as tf
import numpy as np

from datetime import datetime
from Utils import shuffle_batch
from Preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as sklearn_f1_score
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

# Hyperparameters:
EMBEDDINGS_FILENAME = "crawl-300d-2M.vec"
EMBEDDING_DIMENSION = 300
BATCH_SIZE = 128
MAX_TWEET_LENGTH = 20
EPOCHS = 100
LSTM_NEURONS = 256
NEURONS_HIDDEN_LAYER_1 = 128
RNN_LAYERS = 3
DROPOUT_KEEP_PROBABILITY = 0.5
NEURONS_SOFTMAX = 2
LOG_DIR = "./model_output"
LEARNING_RATE = 0.001
TRAINABLE_EMBEDDINGS = False
LAMBDA_L2_REG = 0.00001
SUBTASK = 'subtask_a'


if __name__ == "__main__":

    # Load and filter data.
    X, y = Preprocessing.filter_data(Preprocessing.load_data("./training-v1/offenseval-training-v1.tsv"), SUBTASK)

    # Split data.
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Tokenize data.
    train_data, test_data, vocab_freq, word2idx, vocab_size = Preprocessing.prepare_data(x_train, x_test,
                                                                                         MAX_TWEET_LENGTH)

    # Create an embedding matrix.
    embedding_matrix = Preprocessing.create_embedding_matrix(word2idx, EMBEDDING_DIMENSION, EMBEDDINGS_FILENAME)

    # Placeholders.
    X = tf.placeholder(tf.int32, [None, MAX_TWEET_LENGTH], name="X_input")
    y = tf.placeholder(tf.int64, [None], name="y_label")
    keep_prob = tf.placeholder_with_default(1.0, shape=())

    # Define the variable that will hold the embedding.
    embeddings = tf.get_variable(name="embeddings", shape=[vocab_size, EMBEDDING_DIMENSION],
                                 initializer=tf.constant_initializer(embedding_matrix), trainable=TRAINABLE_EMBEDDINGS)

    # Find the embeddings.
    x_embedded = tf.nn.embedding_lookup(embeddings, X)
    print("Input shape: ", x_embedded.shape)

    # A dynamic RNN.
    lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=LSTM_NEURONS, name='lstm_cell')
                  for layer in range(RNN_LAYERS)]
    cells_drop = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
                  for cell in lstm_cells]
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells_drop)

    # A bidirectional RNN is used.
    outputs, states = bi_rnn(multi_cell, multi_cell, inputs=x_embedded, dtype=tf.float32)
    print("RNN forward output shape: ", outputs[0].shape)
    print("RNN backward output shape: ", outputs[1].shape)

    outputs = tf.add(outputs[0][:, -1, :], outputs[1][:, -1, :])
    print("RNN squeezed output shape: ", outputs.shape)

    # A hidden layer.
    hidden1 = tf.layers.dense(outputs,
                              NEURONS_HIDDEN_LAYER_1, name="hidden_1", activation='relu')
    print("Hidden layer shape: ", hidden1.shape)

    # A classification layer.
    logits = tf.layers.dense(hidden1, NEURONS_SOFTMAX, name="softmax", activation='softmax')
    print("Logits shape: ", logits.shape)

    # Loss and optimizer.
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

    l2 = LAMBDA_L2_REG * sum([
        tf.nn.l2_loss(tf_var)
        for tf_var in tf.trainable_variables()
        if ("bias" not in tf_var.name or "carry_b" not in tf_var.name)]
    )

    loss += l2
    print("L2 regularized loss: ", loss)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
    training_op = optimizer.minimize(loss)

    # Predictions and accuracy.
    predictions = tf.argmax(logits, 1, name="predictions")
    correct_predictions = tf.equal(predictions, y)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    # Initializer.
    init = tf.global_variables_initializer()

    # Saver.
    saver = tf.train.Saver()

    # Summary information for saving.
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    summary_op = tf.summary.merge_all()
    logdir = "{}/run-{}/".format(LOG_DIR, now)

    with tf.Session() as sess:
        summary_writer_train = tf.summary.FileWriter(logdir + '/train', tf.get_default_graph())
        summary_writer_test = tf.summary.FileWriter(logdir + '/test')

        init.run()
        for epoch in range(1, EPOCHS + 1):
            X_batch = None
            y_batch = None
            for X_batch, y_batch in shuffle_batch(train_data, y_train, BATCH_SIZE):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, keep_prob: DROPOUT_KEEP_PROBABILITY})

            # Accuracies after one epoch.
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: test_data, y: np.reshape(y_test, (y_test.shape[0],))})

            # Get predictions for the test set.
            _, y_pred = sess.run([accuracy, predictions], feed_dict={X: test_data, y: np.reshape(y_test, (y_test.shape[0],))})

            # Write summaries.
            summary_train_acc = acc_summary.eval(feed_dict={X: train_data, y: np.reshape(y_train, (y_train.shape[0],))})
            summary_test_acc = acc_summary.eval(feed_dict={X: test_data, y: np.reshape(y_test, (y_test.shape[0],))})
            summary_writer_train.add_summary(summary_train_acc, epoch)
            summary_writer_test.add_summary(summary_test_acc, epoch)

            print("Epoch: {} Last batch accuracy: {} Test accuracy: {} F1-Score: {}".format(epoch, acc_train, acc_test,
                                                                    sklearn_f1_score(y_test, y_pred, average='macro')))

        saver.save(sess, LOG_DIR + "/tf_model")


