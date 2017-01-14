import argparse
import io
import json

import h5py
import numpy
import tensorflow as tf

from data_set import DataSet
import net


parser = argparse.ArgumentParser(description="")
parser.add_argument("-r", dest="root", nargs="?", help="Directory containing Training and Testing subdirectories.")
parser.add_argument("-c", dest="config", nargs="?", help="Directory containing configuration.json")
parser.add_argument("-n", dest="network", nargs="?", help="Directory containing network.json")
args = parser.parse_args()

# Data configuraion
configuration_parameter_file = io.open(args.config, "r")
configuration_parameter_json = json.load(configuration_parameter_file)

train_data = DataSet(args.root + "/Training", configuration_parameter_json)
test_data = DataSet(args.root + "/Testing", configuration_parameter_json)

network_parameter_file = io.open(args.network, "r")
network_parameter_json = json.load(network_parameter_file)

# Training configuration
learning_rate = network_parameter_json["learning_rate"]
max_epoch = network_parameter_json["max_epoch"]
batch_size = network_parameter_json["batch_size"]
test_interval = network_parameter_json["test_interval"]
snapshot_interval = network_parameter_json["snapshot_interval"]

# Network configuration
lstm_units = network_parameter_json["lstm_units"]
layer_count = network_parameter_json["layer_count"]
train_data.max_sequence_length = network_parameter_json["max_sequence_length"]
train_data.min_sequence_length = network_parameter_json["min_sequence_length"]
test_data.max_sequence_length = network_parameter_json["max_sequence_length"]
test_data.min_sequence_length = network_parameter_json["min_sequence_length"]

def fill_batch_vars(dataset, feature_var, note_label_var, polyphony_label_var, onset_label_var, feature_length_var):
    (features, feature_lengths), (note_labels, polyphony_labels, onset_labels) = dataset.next_batch(batch_size)
    feed_dict = {
        feature_var: features,
        note_label_var: note_labels,
        polyphony_label_var: polyphony_labels,
        onset_label_var: onset_labels,
        feature_length_var: feature_lengths,
    }
    return feed_dict

def exportToHDF5(i, variables, session):
    file = h5py.File("net%d.h5" % i, "w")

    for variable in variables:
        name = variable.name.replace("/", "")[:-2]
        dataset = file.create_dataset(name, variable.get_shape(), dtype=variable.dtype.as_numpy_dtype)
        dataset[...] = variable.eval(session)

if __name__ == '__main__':
    with tf.Graph().as_default(), tf.Session() as sess:
        features_placeholder = tf.placeholder(tf.float32, shape=(batch_size, train_data.max_sequence_length, train_data.feature_size))
        feature_lengths_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
        note_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, train_data.max_sequence_length, train_data.note_label_size))
        polyphony_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, train_data.max_sequence_length))
        onset_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, train_data.max_sequence_length))

        features = [tf.squeeze(t) for t in tf.split(1, train_data.max_sequence_length, features_placeholder)]

        note_logits, polyphony_logits, onset_logits = net.run_net(features, feature_lengths_placeholder, lstm_units, layer_count)

        sequence_count = tf.reduce_sum(feature_lengths_placeholder)
        loss = (tf.nn.l2_loss(note_labels_placeholder-note_logits) +
                tf.nn.l2_loss(polyphony_labels_placeholder-polyphony_logits) +
                tf.nn.l2_loss(onset_labels_placeholder-onset_logits)) / sequence_count

        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        init = tf.global_variables_initializer()
        sess.run(init)

        # Initialize the test accuracy file
        test_accuracy_file = open('accuracy.csv', 'w')
        test_accuracy_file.write("Iteration, Loss, note, polyphony, onset\n")

        for i in range(max_epoch):
            feed_dict = fill_batch_vars(train_data, features_placeholder, note_labels_placeholder, polyphony_labels_placeholder, onset_labels_placeholder, feature_lengths_placeholder)
            percent_list = []

            def report(file, feed_dict, loss):
                np_note_logits, np_polyphony_logits, np_onset_logits = sess.run([note_logits, polyphony_logits, onset_logits], feed_dict=feed_dict)
                np_note_labels, np_polyphony_labels, np_onset_labels = (feed_dict[note_labels_placeholder], feed_dict[polyphony_labels_placeholder], feed_dict[onset_labels_placeholder])
                feature_lengths = feed_dict[feature_lengths_placeholder]
                note_score = net.note_score(np_note_labels, np_note_logits, np_polyphony_labels, feature_lengths)
                polyphony_score = net.polyphony_score(np_polyphony_labels, np_polyphony_logits, feature_lengths)
                onset_score = net.onset_score(np_onset_labels, np_onset_logits, feature_lengths)

                low_note_score = numpy.mean(numpy.power(np_note_labels[:,:,0:29] - np_note_logits[:,:,0:29], 2))
                mid_note_score = numpy.mean(numpy.power(np_note_labels[:,:,29:59] - np_note_logits[:,:,29:59], 2))
                hig_note_score = numpy.mean(numpy.power(np_note_labels[:,:,59:88] - np_note_logits[:,:,59:88], 2))

                file.write("%d, %f, %f, %f, %f, %f, %f, %f\n" %
                    (i, loss, note_score, polyphony_score, onset_score, low_note_score, mid_note_score, hig_note_score))
                file.flush()


            _ = sess.run([train_op], feed_dict=feed_dict)

            if i % test_interval == 0:
                feed_dict = fill_batch_vars(test_data, features_placeholder, note_labels_placeholder, polyphony_labels_placeholder, onset_labels_placeholder, feature_lengths_placeholder)
                test_percent = sess.run(loss, feed_dict=feed_dict)
                report(test_accuracy_file, feed_dict, test_percent)

            if i % snapshot_interval == 0:
                exportToHDF5(i, tf.trainable_variables(), sess)

        test_accuracy_file.close()
