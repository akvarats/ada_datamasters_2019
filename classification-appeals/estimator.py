import sys
import os
import argparse
import copy
import datetime
import numpy as np
import json

from multiprocessing import Pool

from corpus_model import CorpusModel
from knn_model import KNNModel
from tools import get_morph_predictor, get_word2vec_model, get_chunks, normalized_executor


def estimate_model_on_range(model_path, log_path, index_from, index_to):
    """

    :param model:
    :return:
    """

    corpus_model = CorpusModel().load(model_path)
    predictor = get_morph_predictor()
    word2vec_model = get_word2vec_model(corpus_model.meta.get("word2vec"))

    correct = 0
    subcorrect = 0
    incorrect = 0

    log_data = []

    for doc_index in range(index_from, index_to):
        corpus_model_copy = copy.deepcopy(corpus_model)

        corpus_model_copy.distances = np.delete(corpus_model_copy.distances, doc_index, axis=0)
        corpus_model_copy.distances = np.delete(corpus_model_copy.distances, doc_index, axis=1)

        del corpus_model_copy.corpus[doc_index]

        knn = KNNModel(predictor=predictor, word2vec=word2vec_model)
        knn.fit(corpus_model_copy)

        prediction = knn.predict(corpus_model.corpus[doc_index]["orig_text"])

        # оценка предсказанной категории
        category_prediction_status = None
        if prediction.predicted_category == corpus_model.corpus[doc_index]["category"]:
            category_prediction_status = "success"
        else:
            if corpus_model.corpus[doc_index]["category"] in [p[0] for p in prediction.top3_predicted_categories]:
                category_prediction_status = "top3"
            else:
                category_prediction_status = "fail"

        # оценка предсказанной темы
        theme_prediction_status = None
        if prediction.predicted_theme == corpus_model.corpus[doc_index]["theme"]:
            theme_prediction_status = "success"
        else:
            if corpus_model.corpus[doc_index]["theme"] in [p[0] for p in prediction.top3_predicted_themes]:
                theme_prediction_status = "top3"
            else:
                theme_prediction_status = "fail"

        # оценка предсказанного исполнителя
        executor_prediction_status = None
        if prediction.predicted_executor == normalized_executor(corpus_model.corpus[doc_index]["executor"]):
            executor_prediction_status = "success"
        else:
            if normalized_executor(corpus_model.corpus[doc_index]["executor"]) in [p[0] for p in prediction.top3_predicted_executors]:
                executor_prediction_status = "top3"
            else:
                executor_prediction_status = "fail"

        # запись в лог

        log_data.append(dict(
            source=dict(
                id=corpus_model.corpus[doc_index]["id"],
                text=corpus_model.corpus[doc_index]["orig_text"],
                category=corpus_model.corpus[doc_index]["category"],
                theme=corpus_model.corpus[doc_index]["theme"],
                executor=corpus_model.corpus[doc_index]["executor"],
            ),
            prediction=dict(
                category=dict(
                    predicted=prediction.predicted_category,
                    top3=prediction.top3_predicted_categories,
                    status=category_prediction_status,
                ),
                executor=dict(
                    predicted=prediction.predicted_executor,
                    top3=prediction.top3_predicted_executors,
                    status=executor_prediction_status,
                ),
                theme=dict(
                    predicted=prediction.predicted_theme,
                    top3=prediction.top3_predicted_themes,
                    status=theme_prediction_status,
                )
            )
        ))

        with open(os.path.join(log_path, "{0}_{1}.json".format(index_from, index_to)), "wt") as f:
            f.write(json.dumps(log_data, ensure_ascii=False, indent=2))


def parallel_estimation(data):
    estimate_model_on_range(data[0], data[1], data[2], data[3])


def estimate_model(model_path):
    """ Оценивает работу по рейнджу записей из модели корпуса """

    # Готовим задания по анализу ло
    corpus_model = CorpusModel().load(model_path)
    doc_chunks = get_chunks(len(corpus_model.corpus), chunk_size=200)

    # Создаем папку, в которую положим результаты
    base_logs_path = "logs"
    logs_count = len(os.listdir(base_logs_path))

    logfile_folder = os.path.join(base_logs_path, "{0}_{1}".format(model_path.replace("/", "_"), "0000{}".format(logs_count + 1)[-4:]))
    os.makedirs(logfile_folder, exist_ok=True)

    chunk_data = [(model_path, logfile_folder, doc_chunk[0], doc_chunk[1]) for doc_chunk in doc_chunks]


    with Pool() as p:
        p.map(parallel_estimation, chunk_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="Путь до модели корпуса")

    args = parser.parse_args()

    if not args.model:
        sys.stderr.write("Не указан путь до модели (параметр --model)\n")
        sys.exit(1)

    if not os.path.exists(args.model):
        sys.stderr.write("Папка \"{0}\" не найдена\n".format(args.model))
        sys.exit(1)

    estimate_model(args.model)





