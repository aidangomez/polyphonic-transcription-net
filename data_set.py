import h5py as h5
import numpy as np

import os
import re


def num_notes(json_file, range_key):
    elems = re.findall(r"\d+", json_file[range_key])
    return int(elems[1]) - int(elems[0]) + 1


class DataSet(object):
    def __init__(self, audio_folder, json_file):
        self.max_sequence_length = 60
        self.min_sequence_length = 15

        self.audio_folder = audio_folder
        self.features = json_file["features"]
        self.individual_feature_size = int(num_notes(json_file, "spectrumNoteRange") * json_file["spectrumResolution"])
        self.feature_size = len(self.features) * self.individual_feature_size
        self.note_label_size = num_notes(json_file, "representableNoteRange")
        self.file_list = self.generate_file_list()

    def generate_file_list(self):
        file_list = []
        for _, _, files in os.walk(self.audio_folder):
            for filename in files:
                if not filename.endswith(".h5"):
                    continue
                h5file = h5.File(os.path.join(self.audio_folder, filename), "r")
                feature_count = h5file[self.features[0]].shape[0]
                if feature_count < self.min_sequence_length:
                    continue
                file_list.append((filename, feature_count))
        return file_list

    def generate_batch_list(self, batch_size):
        indices = np.random.choice(range(len(self.file_list)), batch_size)
        offsets = [np.random.choice(range(self.file_list[i][1] - self.min_sequence_length + 1)) for i in indices]
        lengths = [np.random.choice(range(self.min_sequence_length, min(self.file_list[indices[i]][1] - offsets[i], self.max_sequence_length) + 1)) for i in range(batch_size)]
        return indices, offsets, lengths

    def generate_batch_data(self, batch_size):
        indices, offsets, lengths = self.generate_batch_list(batch_size)

        labels_onset = np.zeros((batch_size, self.max_sequence_length))
        labels_polyphony = np.zeros((batch_size, self.max_sequence_length))
        labels_notes = np.zeros((batch_size, self.max_sequence_length, self.note_label_size))
        feature_data = [np.zeros((batch_size, self.max_sequence_length, self.individual_feature_size)) for feature in self.features]
        feature_lengths = np.zeros((batch_size), dtype=int)

        for i in range(batch_size):
            index = indices[i]
            offset = offsets[i]
            length = lengths[i]

            file_name = self.file_list[index][0]
            h5file = h5.File(os.path.join(self.audio_folder, file_name), "r")
            labels_onset[i, 0:length, ...] = h5file["labels/onset"][offset:offset+length, ...]
            labels_polyphony[i, 0:length, ...] = h5file["labels/polyphony"][offset:offset+length, ...]
            labels_notes[i, 0:length, ...] = h5file["labels/notes"][offset:offset+length, ...]
            feature_lengths[i, ...] = length

            for j, feature in enumerate(self.features):
                feature_data[j][i, 0:length, ...] = h5file[feature][offset:offset+length, ...]

        return labels_onset, labels_polyphony, labels_notes, feature_data, feature_lengths

    def next_batch(self, batch_size):
        labels_onset, labels_polyphony, labels_notes, feature_data, feature_lengths = self.generate_batch_data(batch_size)

        features = np.concatenate(feature_data, axis=2)

        return (features, feature_lengths), (labels_notes, labels_polyphony, labels_onset)
