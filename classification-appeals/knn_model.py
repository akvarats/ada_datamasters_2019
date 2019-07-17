from collections import OrderedDict

from sklearn.neighbors import KNeighborsClassifier

from tools import get_clean_rusvectores_words


class KNNPrediction(object):

    def __init__(self):
        self._category = None
        self._probable_categories = []

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, value):
        self._category = value

    @property
    def probable_categories(self):
        return self._probable_categories

    @probable_categories.setter
    def probable_categories(self, value):
        self._probable_categories = value


class KNNModel(object):

    def __init__(self, predictor, word2vec):

        self._predictor = predictor
        self._word2vec = word2vec

        self._knn = None
        self._corpus_model = None

        self._categories = None
        self._targets_category = None

    @property
    def corpus_model(self):
        return self._corpus_model

    @property
    def corpus(self):
        return self.corpus_model.corpus if self.corpus_model else None

    @property
    def distances(self):
        return self.corpus_model.distances if self.corpus_model else None

    def fit(self, corpus_model):
        """

        :param corpus_model:
        :return:
        """
        self._corpus_model = corpus_model
        self._prepare_categories()

        self._knn = KNeighborsClassifier(n_neighbors=10, metric="precomputed")
        self._knn.fit(self.distances, self._targets_category)

        print(len(self._categories))

    def predict(self, text):

        text_vectors = get_clean_rusvectores_words(text, self._predictor, self._word2vec)

        text_distances = [self._word2vec.wmdistance(text_vectors, c["cleaned_rusvectores_words"]) for c in self.corpus]

        prediction = KNNPrediction()

        predicted_category_index = self._knn.predict([text_distances])
        prediction.category = self._category_name_by_index(predicted_category_index)

        predicted_proba = self._knn.predict_proba([text_distances])

        for i in range(0, len(predicted_proba[0])):
            if predicted_proba[0][i] > 0.0001:
                prediction.probable_categories.append((self._category_name_by_index(i), predicted_proba[0][i]))

        prediction.probable_categories = sorted(prediction.probable_categories, key=lambda x: x[1], reverse=True)

        return prediction

    # ----------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------
    def _prepare_categories(self):
        """"""
        self._categories = OrderedDict()
        self._targets_category = []

        for row in self.corpus:
            if row["category"] not in self._categories:
                category_index = len(self._categories)
                self._categories[row["category"]] = category_index
            self._targets_category.append(self._categories[row["category"]])

    def _category_name_by_index(self, category_index):
        result = None

        for cname, cindex in self._categories.items():
            if cindex == category_index:
                result = cname
                break
        return result