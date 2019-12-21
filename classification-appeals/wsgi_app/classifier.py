from corpus_model import CorpusModel
from knn_model import KNNModel
from tools import get_morph_predictor, get_word2vec_model


class PredictorManager:

    def __init__(self):
        self._predictor = None
        self._predictor = self.predictor

    @property
    def predictor(self):
        if self._predictor is None:
            corpus_model = CorpusModel().load('models/ya_1221_180')
            predictor = get_morph_predictor()

            word2vec_model = get_word2vec_model(corpus_model.meta.get("word2vec"))

            self._predictor = KNNModel(predictor=predictor, word2vec=word2vec_model)
            self._predictor.fit(corpus_model)

        return self._predictor


predictorManager = PredictorManager()


def classify(text):
    prediction = predictorManager.predictor.predict(text)
    return {
        'top3_ids': prediction.top3_predicted_ids,
        'top3_themes': prediction.top3_predicted_themes,
        'top3_categories': prediction.top3_predicted_categories,
        'top3_executors': prediction.top3_predicted_executors,
    }


if __name__ == '__main__':
    import json
    txt = """апрель"""
    print(json.dumps(classify(txt), ensure_ascii=False))
