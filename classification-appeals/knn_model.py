from collections import OrderedDict

from sklearn.neighbors import KNeighborsClassifier

from tools import get_clean_rusvectores_words


class KNNPrediction(object):

    def __init__(self):
        self._category = None
        self._probable_categories = []

        self._all_predicted_themes = []
        self._predicted_theme = None

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

    # --------------------------------------------------------------------------------------------
    # Свойства для предстказаний по темам
    # --------------------------------------------------------------------------------------------
    @property
    def all_predicted_themes(self):
        return self._all_predicted_themes

    @all_predicted_themes.setter
    def all_predicted_themes(self, value):
        self._all_predicted_themes = value

    @property
    def predicted_theme(self):
        return self._predicted_theme

    @predicted_theme.setter
    def predicted_theme(self, value):
        self._predicted_theme = value

    @property
    def top3_predicted_themes(self):
        return self.all_predicted_themes[0:3] if self.all_predicted_themes else []


class KNNModel(object):
    """ Основная модель классификации """

    def __init__(self, predictor, word2vec):

        self._predictor = predictor
        self._word2vec = word2vec

        self._knn = None
        self._corpus_model = None

        self._categories = None
        self._targets_category = None

        self._themes = None
        self._targets_theme = None

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
        self._prepare_themes()

        self._knn_categories = KNeighborsClassifier(n_neighbors=10, metric="precomputed")
        self._knn_categories.fit(self.distances, self._targets_category)

        self._knn_themes = KNeighborsClassifier(n_neighbors=10, metric="precomputed")
        self._knn_themes.fit(self.distances, self._targets_theme)

    def predict(self, text):
        """
        Предсказывает категорию, тему и проч.
        """
        text_vectors = get_clean_rusvectores_words(text, self._predictor, self._word2vec)

        text_distances = [self._word2vec.wmdistance(text_vectors, c["cleaned_rusvectores_words"]) for c in self.corpus]

        prediction = KNNPrediction()

        # Выполняем предсказание категории
        # ---------------------------------------------------------------------------
        predicted_category_index = self._knn_categories.predict([text_distances])
        prediction.category = self._category_name_by_index(predicted_category_index)

        predicted_proba = self._knn_categories.predict_proba([text_distances])

        for i in range(0, len(predicted_proba[0])):
            if predicted_proba[0][i] > 0.0001:
                prediction.probable_categories.append((self._category_name_by_index(i), predicted_proba[0][i]))

        prediction.probable_categories = sorted(prediction.probable_categories, key=lambda x: x[1], reverse=True)

        # Выполняем предсказание темы
        # ---------------------------------------------------------------------------
        prediction.predicted_theme = self._theme_name_by_index(self._knn_themes.predict([text_distances]))
        prediction.all_predicted_themes = []

        predicted_themes_proba = self._knn.predict_proba([text_distances])

        for i in range(0, len(predicted_themes_proba[0])):
            if predicted_themes_proba[0][i] > 0.0001:
                prediction.all_predicted_themes.append((
                    self._theme_name_by_index(i), 
                    predicted_themes_proba[0][i]
                ))

        return prediction

    # ----------------------------------------------------------------------------------------------
    # всякие функции
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

    def _prepare_themes(self):
        """ """
        self._themes = OrderedDict()
        self._targets_theme = []

        for row in self.corpus:
            if row["theme"] not in self._themes:
                theme_index = len(self._themes)
                self._themes[row["theme"]] = theme_index
            self._targets_theme.append(self._themes[row["theme"]])

    def _category_name_by_index(self, category_index):
        result = None

        for cname, cindex in self._categories.items():
            if cindex == category_index:
                result = cname
                break
        return result

    def _theme_name_by_index(self, theme_index):
        result = None

        for tname, tindex in self._themes.items():
            if tindex == theme_index:
                result = tname
                break
        return result