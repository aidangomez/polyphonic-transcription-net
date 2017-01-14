import tensorflow as tf
import math
import numpy as np

from data_set import DataSet
import runner

polyphony_label_size = 1
onset_label_size = 1

def run_net(features, feature_lengths, lstm_units, layer_count):
    note_W = tf.Variable(
        tf.truncated_normal([lstm_units, runner.train_data.note_label_size],
                            stddev=1.0 / math.sqrt(float(lstm_units))),
        name="note_ip_weights")
    note_b = tf.Variable(tf.zeros([runner.train_data.note_label_size]),
                         name="note_ip_biases")

    polyphony_W = tf.Variable(
        tf.truncated_normal([lstm_units, polyphony_label_size],
                            stddev=1.0 / math.sqrt(float(lstm_units))),
        name="polyphony_ip_weights")
    polyphony_b = tf.Variable(tf.zeros([polyphony_label_size]),
                              name="polyphony_ip_biases")

    onset_W = tf.Variable(
        tf.truncated_normal([lstm_units, onset_label_size],
                            stddev=1.0 / math.sqrt(float(lstm_units))),
        name="onset_ip_weights")
    onset_b = tf.Variable(tf.zeros([onset_label_size]),
                          name="onset_ip_biases")

    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * layer_count)

    rnn_out, _ = tf.nn.rnn(stacked_lstm, features, dtype=tf.float32, sequence_length=feature_lengths)

    note_logits = tf.concat(1, [tf.reshape(tf.matmul(t, note_W) + note_b, [runner.batch_size, 1, runner.train_data.note_label_size]) for t in rnn_out])
    polyphony_logits = tf.concat(1, [tf.reshape(tf.matmul(t, polyphony_W) + polyphony_b, [runner.batch_size, 1]) for t in rnn_out])
    onset_logits = tf.concat(1, [tf.reshape(tf.matmul(t, onset_W) + onset_b, [runner.batch_size, 1]) for t in rnn_out])

    return note_logits, polyphony_logits, onset_logits

def note_score(note_labels, note_logits, polyphony_labels, feature_lengths):
    score = 0.0
    count = 0.0
    for i in range(runner.batch_size):
        length = feature_lengths[i]
        for j in range(length):
            polyphony = math.ceil(polyphony_labels[i, j])
            if polyphony > 0:
                top_label_notes = np.argpartition(note_labels[i, j], -polyphony)[-polyphony:]
                top_logit_notes = np.argpartition(note_logits[i, j], -polyphony)[-polyphony:]
                for k in top_label_notes:
                    score += 1 if k in top_logit_notes else 0

                count += polyphony

    return 100 * score / count

def onset_score(onset_labels, onset_logits, feature_lengths):
    score = 0.0
    count = 0.0
    for i in range(runner.batch_size):
        length = feature_lengths[i]
        for j in range(length):
            score += abs(onset_labels[i, j] - onset_logits[i, j])
            count += 1

    return score / count

def polyphony_score(polyphony_labels, polyphony_logits, feature_lengths):
    score = 0.0
    count = 0.0
    for i in range(runner.batch_size):
        length = feature_lengths[i]
        for j in range(length):
            score += abs(polyphony_labels[i, j] - polyphony_logits[i, j])
            count += 1

    return score / count
