import sys
import os
import argparse
import copy
import numpy as np

from multiprocessing import Pool

from corpus_model import CorpusModel
from knn_model import KNNModel
from tools import get_morph_predictor, get_word2vec_model, get_chunks


def estimate_model_on_range(model_path, index_from, index_to):
    """

    :param model:
    :return:
    """

    sys.stdout.write("\n--------------------------------------------------------------------------------\n")
    sys.stdout.write("Загружаем модель корпуса\n")
    sys.stdout.write("--------------------------------------------------------------------------------\n")

    corpus_model = CorpusModel().load(model_path)

    sys.stdout.write("\n--------------------------------------------------------------------------------\n")
    sys.stdout.write("Загрузка морфологического предиктора\n")
    sys.stdout.write("--------------------------------------------------------------------------------\n")
    predictor = get_morph_predictor()

    sys.stdout.write("\n--------------------------------------------------------------------------------\n")
    sys.stdout.write("Загрузка word2vec модели\n")
    sys.stdout.write("--------------------------------------------------------------------------------\n")
    word2vec_model = get_word2vec_model(corpus_model.meta.get("word2vec"))

    sys.stdout.write("\n--------------------------------------------------------------------------------\n")
    sys.stdout.write("Создаем папку для лог файла\n")
    sys.stdout.write("--------------------------------------------------------------------------------\n")
    logfile_folder = os.path.join("logs", "")
    os.makedirs(logfile_folder, exist_ok=True)

    correct = 0
    subcorrect = 0
    incorrect = 0

    for doc_index in range(index_from, index_to):
        corpus_model_copy = copy.deepcopy(corpus_model)

        corpus_model_copy.distances = np.delete(corpus_model_copy.distances, doc_index, axis=0)
        corpus_model_copy.distances = np.delete(corpus_model_copy.distances, doc_index, axis=1)

        del corpus_model_copy.corpus[doc_index]

        knn = KNNModel(predictor=predictor, word2vec=word2vec_model)
        knn.fit(corpus_model_copy)

        prediction = knn.predict(corpus_model.corpus[doc_index]["orig_text"])

        if prediction.category == corpus_model.corpus[doc_index]["category"]:
            correct += 1
        else:
            if corpus_model.corpus[doc_index]["category"] in [p[0] for p in prediction.probable_categories[0:3]]:
                subcorrect += 1
            else:
                incorrect += 1

        print("----------------------")
        print("{0} -> {1}".format(corpus_model.corpus[doc_index]["category"], [p[0] for p in prediction.probable_categories[0:3]]))

        print("Correct: {0}".format(correct))
        print("In top 3: {0}".format(subcorrect))
        print("Wrong: {0}".format(incorrect))


def estimate_model(model_path):
    """ Оценивает работу по рейнджу записей из модели корпуса """
    
    corpus_model = CorpusModel().load(model_path)

    doc_chunks = get_chunks(len(corpus_model.corpus), chunk_size=15)

    def parallel_function (doc_chunk): 
        estimate_model_on_range(model_path, doc_chunk[0], doc_chunk[1])

    with Pool() as p:
        p.map(parallel_function, doc_chunks)



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





