import os
import sys
import json
import argparse


class AggregatedLog(object):
    def __init__(self):

        self.total = 0

        self.category_success = 0
        self.category_top3 = 0
        self.category_fail = 0
        self.category_unknown = 0

        self.theme_success = 0
        self.theme_top3 = 0
        self.theme_fail = 0
        self.theme_unknown = 0

        self.executor_success = 0
        self.executor_top3 = 0
        self.executor_fail = 0
        self.executor_unknown = 0


def analyze_log(log_path):

    log_files = os.listdir(log_path)

    print("Обнаружено {0} файлов с логами".format(len(log_files)))

    results = AggregatedLog()

    for log_file in log_files:
        with open(os.path.join(log_path, log_file), "rt") as f:
            log_data = json.loads(f.read())
            for entry in log_data:
                results.total += 1

                category_status = entry["prediction"]["category"]["status"]
                if category_status == "success":
                    results.category_success += 1
                elif category_status == "top3":
                    results.category_top3 += 1
                elif category_status == "fail":
                    results.category_fail += 1
                else:
                    results.category_unknown += 1

                theme_status = entry["prediction"]["theme"]["status"]
                if theme_status == "success":
                    results.theme_success += 1
                elif theme_status == "top3":
                    results.theme_top3 += 1
                elif theme_status == "fail":
                    results.theme_fail += 1
                else:
                    results.theme_unknown += 1

                executor_status = entry["prediction"]["executor"]["status"]
                if executor_status == "success":
                    results.executor_success += 1
                elif executor_status == "top3":
                    results.executor_top3 += 1
                elif executor_status == "fail":
                    results.executor_fail += 1
                else:
                    results.executor_unknown += 1

    print("Всего документов: {}".format(results.total))
    
    print("Категории:")
    print("  успех: {} ({:.1f}%)".format(results.category_success, (100.0 * results.category_success / results.total) if results.total else 0))
    print("  топ-3: {} ({:.1f}%)".format(results.category_top3, (100.0 * results.category_top3 / results.total) if results.total else 0))
    print("  ошибка: {} ({:.1f}%)".format(results.category_fail, (100.0 * results.category_fail / results.total) if results.total else 0))

    print("Исполнители:")
    print("  успех: {} ({:.1f}%)".format(results.executor_success, 100.0 * (results.executor_success / results.total) if results.total else 0))
    print("  топ-3: {} ({:.1f}%)".format(results.executor_top3, 100.0 * (results.executor_top3 / results.total) if results.total else 0))
    print("  ошибка: {} ({:.1f}%)".format(results.executor_fail, 100.0 * (results.executor_fail / results.total) if results.total else 0))

    print("Темы:")
    print("  успех: {} ({:.1f}%)".format(results.theme_success, 100.0 * (results.theme_success / results.total) if results.total else 0))
    print("  топ-3: {} ({:.1f}%)".format(results.theme_top3, 100.0 * (results.theme_top3 / results.total) if results.total else 0))
    print("  ошибка: {} ({:.1f}%)".format(results.theme_fail, 100.0 * (results.theme_fail / results.total) if results.total else 0))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--log", help="Путь до модели корпуса")

    args = parser.parse_args()

    if not args.log:
        sys.stderr.write("Не указан путь до папки с логами (параметр --log)\n")
        sys.exit(1)

    if not os.path.exists(args.log):
        sys.stderr.write("Папка \"{0}\" не найдена\n".format(args.log))
        sys.exit(1)

    analyze_log(args.log)