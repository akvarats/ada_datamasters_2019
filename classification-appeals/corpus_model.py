import os
import json
import numpy as np


class CorpusModel(object):

    def __init__(self):
        self._corpus = None
        self._distances = None
        self._meta = None

    @property
    def corpus(self):
        return self._corpus

    @property
    def distances(self):
        return self._distances

    @distances.setter
    def distances(self, value):
        self._distances = value

    @property
    def meta(self):
        return self._meta

    def load(self, model_folder):

        meta_file_path = os.path.join(model_folder, "meta.json")
        corpus_file_path = os.path.join(model_folder, "corpus.json")
        distance_file_path = os.path.join(model_folder, "distances.npy")

        with open(meta_file_path, "rt") as f:
            self._meta = json.loads(f.read())

        with open(corpus_file_path, "rt") as f:
            self._corpus = json.loads(f.read())

        self._distances = np.load(distance_file_path)

        return self
