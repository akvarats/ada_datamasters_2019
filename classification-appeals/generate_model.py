import sys
import os
import argparse
import shutil
import json

import numpy as np

import gensim
import datetime

from tools import read_corpus, get_morph_predictor, get_word2vec_model, get_distance_matrix_for_corpus


def generate_model(model_folder, corpus_filepath, word2vec_filepath):
    """

    :param model_folder:
    :return:
    """

    sys.stdout.write("\n--------------------------------------------------------------------------------\n")
    sys.stdout.write("Загрузка морфологического предиктора\n")
    sys.stdout.write("--------------------------------------------------------------------------------\n")
    predictor = get_morph_predictor()

    sys.stdout.write("\n--------------------------------------------------------------------------------\n")
    sys.stdout.write("Загрузка word2vec модели\n")
    sys.stdout.write("--------------------------------------------------------------------------------\n")
    word2vec_model = get_word2vec_model(word2vec_filepath)

    sys.stdout.write("\n--------------------------------------------------------------------------------\n")
    sys.stdout.write("Чтение корпуса\n")
    sys.stdout.write("--------------------------------------------------------------------------------\n")

    t = datetime.datetime.now()
    corpus = read_corpus(corpus_filepath, predictor, word2vec_model)
    sys.stdout.write("Корпус загружен за {0:.3f} сек\n".format((datetime.datetime.now() - t).total_seconds()))
    sys.stdout.write("Документов в корпусе: {0}\n".format(len(corpus)))

    sys.stdout.write("\n--------------------------------------------------------------------------------\n")
    sys.stdout.write("Построение матрицы дистанций\n")
    sys.stdout.write("--------------------------------------------------------------------------------\n")

    t = datetime.datetime.now()
    D = get_distance_matrix_for_corpus(corpus, word2vec_model)
    sys.stdout.write("Матрица дистанций вычислена за {0:.3f} сек\n".format((datetime.datetime.now() - t).total_seconds()))

    sys.stdout.write("\n--------------------------------------------------------------------------------\n")
    sys.stdout.write("Сохранение модели\n")
    sys.stdout.write("--------------------------------------------------------------------------------\n")

    meta_file = os.path.join(model_folder, "meta.json")
    model_meta = dict(
        corpus=corpus_filepath,
        docs_count=len(corpus),
        word2vec=word2vec_filepath,
    )

    with open(meta_file, "wt") as f:
        f.write(json.dumps(model_meta, ensure_ascii=False, indent=2))

    out_corpus_file = os.path.join(model_folder, "corpus.json")
    sys.stdout.write("Сохранение корпуса документов в {0}\n".format(out_corpus_file))
    with open(out_corpus_file, "wt") as f:
        f.write(json.dumps(corpus, ensure_ascii=False, indent=2))

    out_distance_file = os.path.join(model_folder, "distances")
    sys.stdout.write("Сохранение матрицы дистанций в {0}\n".format(out_distance_file))

    np.save(out_distance_file, D)

    sys.stdout.write("Готово\n")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus", help="Путь до текстов исходного корпуса")
    parser.add_argument("--word2vec", help="Путь до файла c word2vec моделью")
    parser.add_argument("--out", help="Папка, в которую нужно положить сгенерированную модель")
    parser.add_argument("--clear", action="store_true", help="Если в папке out уже есть файлы, то удалить их")

    args = parser.parse_args()

    if not args.corpus:
        sys.stderr.write("Не указан путь до текстов исходного корпуса (параметр --corpus)\n")
        sys.exit(1)

    if not os.path.exists(args.corpus):
        sys.stderr.write("Файл с корпусом \"{0}\" не найден\n".format(args.corpus))
        sys.exit(1)

    if not args.word2vec:
        sys.stderr.write("Не указан путь к word2vec модели (параметр --word2vec)\n")
        sys.exit(1)

    if not os.path.exists(args.word2vec):
        sys.stderr.write("Файл с word2vec моделью \"{0}\" не найден\n".format(args.word2vec))
        sys.exit(1)

    if os.path.exists(args.out):
        # папка уже существует
        if os.path.isdir(args.out):
            if args.clear:
                shutil.rmtree(args.out)
            else:
                sys.stderr.write("Папка {0} уже существует. Если нужно очистить, то передай ключ --clear.\n".format(args.out))
                exit(1)
        else:
            sys.stderr.write("По пути {0} уже распологается что-то, что не является папкой.\n".format(args.out))
            sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    generate_model(model_folder=args.out, corpus_filepath=args.corpus, word2vec_filepath=args.word2vec)
