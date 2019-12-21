from collections import OrderedDict

from sklearn.neighbors import KNeighborsClassifier

from tools import get_clean_rusvectores_words, normalized_executor


class KNNPrediction(object):

    def __init__(self):
        self._all_predicted_categories = []
        self._predicted_category = None

        self._all_predicted_themes = []
        self._predicted_theme = None

        self._all_predicted_executors = []
        self._predicted_executor = []

        self._all_predicted_ids = []
        self._predicted_id = []

    # --------------------------------------------------------------------------------------------
    # Предсказания по категориям
    # --------------------------------------------------------------------------------------------
    @property
    def all_predicted_categories(self):
        return self._all_predicted_categories

    @all_predicted_categories.setter
    def all_predicted_categories(self, value):
        self._all_predicted_categories = value

    @property
    def predicted_category(self):
        return self._predicted_category

    @predicted_category.setter
    def predicted_category(self, value):
        self._predicted_category = value

    @property
    def top3_predicted_categories(self):
        return self.all_predicted_categories[0:3] if self.all_predicted_categories else []

    # --------------------------------------------------------------------------------------------
    # Предсказания по исполнителям
    # --------------------------------------------------------------------------------------------
    @property
    def all_predicted_executors(self):
        return self._all_predicted_executors

    @all_predicted_executors.setter
    def all_predicted_executors(self, value):
        self._all_predicted_executors = value

    @property
    def predicted_executor(self):
        return self._predicted_executor

    @predicted_executor.setter
    def predicted_executor(self, value):
        self._predicted_executor = value

    @property
    def top3_predicted_executors(self):
        return self.all_predicted_executors[0:3] if self.all_predicted_executors else []

    # --------------------------------------------------------------------------------------------
    # Предсказания по темам
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

    # --------------------------------------------------------------------------------------------
    # Предсказания по темам
    # --------------------------------------------------------------------------------------------
    @property
    def all_predicted_ids(self):
        return self._all_predicted_ids

    @all_predicted_ids.setter
    def all_predicted_ids(self, value):
        self._all_predicted_ids = value

    @property
    def predicted_id(self):
        return self._predicted_id

    @predicted_id.setter
    def predicted_id(self, value):
        self._predicted_id = value

    @property
    def top3_predicted_ids(self):
        return self.all_predicted_ids[0:3] if self.all_predicted_ids else []


class KNNModel(object):
    """ Основная модель классификации """

    def __init__(self, predictor, word2vec):

        self._predictor = predictor
        self._word2vec = word2vec

        self._knn_categories = None
        self._knn_themes = None
        self._knn_executors = None
        self._knn_ids = None

        self._corpus_model = None

        self._categories = None
        self._targets_category = None

        self._themes = None
        self._targets_theme = None

        self._executors = None
        self._targets_executors = None

        self._ids = None
        self._target_ids = None

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

        :param corpus_model:C
        :return:
        """
        self._corpus_model = corpus_model
        # self._prepare_categories()
        # self._prepare_themes()
        # self._prepare_executors()
        self._prepare_ids()


        # self._knn_categories = KNeighborsClassifier(n_neighbors=10, metric="precomputed")
        # self._knn_categories.fit(self.distances, self._targets_category)

        # self._knn_themes = KNeighborsClassifier(n_neighbors=10, metric="precomputed")
        # self._knn_themes.fit(self.distances, self._targets_theme)

        # self._knn_executors = KNeighborsClassifier(n_neighbors=10, metric="precomputed")
        # self._knn_executors.fit(self.distances, self._targets_executors)

        self._knn_ids = KNeighborsClassifier(n_neighbors=10, metric="precomputed")
        self._knn_ids.fit(self.distances, self._target_ids)

    def predict(self, text):
        """
        Предсказывает категорию, тему и проч.
        """
        text_vectors = get_clean_rusvectores_words(text, self._predictor, self._word2vec)

        text_distances = [self._word2vec.wmdistance(text_vectors, c["cleaned_rusvectores_words"]) for c in self.corpus]

        prediction = KNNPrediction()

        # # Выполняем предсказание категории
        # # ---------------------------------------------------------------------------
        # prediction.predicted_category = self._category_name_by_index(self._knn_categories.predict([text_distances]))

        # predicted_categories_proba = self._knn_categories.predict_proba([text_distances])

        # for i in range(0, len(predicted_categories_proba[0])):
        #     if predicted_categories_proba[0][i] > 0.0001:
        #         prediction.all_predicted_categories.append((
        #             self._category_name_by_index(i), 
        #             predicted_categories_proba[0][i]
        #         ))

        # prediction.all_predicted_categories = sorted(prediction.all_predicted_categories, key=lambda x: x[1], reverse=True)

        # # Выполняем предсказание темы
        # # ---------------------------------------------------------------------------
        # prediction.predicted_theme = self._theme_name_by_index(self._knn_themes.predict([text_distances]))
        # prediction.all_predicted_themes = []

        # predicted_themes_proba = self._knn_themes.predict_proba([text_distances])

        # for i in range(0, len(predicted_themes_proba[0])):
        #     if predicted_themes_proba[0][i] > 0.0001:
        #         prediction.all_predicted_themes.append((
        #             self._theme_name_by_index(i), 
        #             predicted_themes_proba[0][i]
        #         ))

        # prediction.all_predicted_themes = sorted(prediction.all_predicted_themes, key=lambda x: x[1], reverse=True)

        # # Выполняем предсказание исполнителя
        # # ---------------------------------------------------------------------------
        # prediction.predicted_executor = self._executor_name_by_index(self._knn_executors.predict([text_distances]))
        # prediction.all_predicted_executors = []

        # predicted_executor_proba = self._knn_executors.predict_proba([text_distances])

        # for i in range(0, len(predicted_executor_proba[0])):
        #     if predicted_executor_proba[0][i] > 0.0001:
        #         prediction.all_predicted_executors.append((
        #             self._executor_name_by_index(i), 
        #             predicted_executor_proba[0][i]
        #         ))

        # prediction.all_predicted_executors = sorted(prediction.all_predicted_executors, key=lambda x: x[1], reverse=True)

        # Выполняем предсказание исполнителя
        # ---------------------------------------------------------------------------
        prediction.predicted_id = self._id_name_by_index(self._knn_ids.predict([text_distances]))
        prediction.all_predicted_ids = []

        predicted_id_proba = self._knn_ids.predict_proba([text_distances])

        for i in range(0, len(predicted_id_proba[0])):
            if predicted_id_proba[0][i] > 0.0001:
                prediction.all_predicted_ids.append((
                    self._id_name_by_index(i), 
                    predicted_id_proba[0][i]
                ))

        prediction.all_predicted_ids = sorted(prediction.all_predicted_ids, key=lambda x: x[1], reverse=True)

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

    def _prepare_executors(self):
        """ """
        self._executors = OrderedDict()
        self._targets_executors = []

        for row in self.corpus:
            executor = normalized_executor(row["executor"])
            if executor not in self._executors:
                executor_index = len(self._executors)
                self._executors[executor] = executor_index
            self._targets_executors.append(self._executors[executor])

    def _prepare_ids(self):
        """ """
        self._ids = OrderedDict()
        self._target_ids = []

        for row in self.corpus:
            id = normalized_executor(row["id"])
            if id not in self._ids:
                id_index = len(self._ids)
                self._ids[id] = id_index
            self._target_ids.append(self._ids[id])

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

    def _executor_name_by_index(self, executor_index):
        result = None

        for ename, eindex in self._executors.items():
            if eindex == executor_index:
                result = ename
                break
        return result

    def _id_name_by_index(self, id_index):
        result = None

        for iname, iindex in self._ids.items():
            if iindex == id_index:
                result = iname
                break
        return result
        