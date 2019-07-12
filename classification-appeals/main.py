import nltk
import csv
import gensim
import pymorphy2


nltk.download('punkt')
model = gensim.models.KeyedVectors.load_word2vec_format("models/180/model.bin", binary=True)
# sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
# sentence_president = 'The president greets the press in Chicago'.lower().split()
# similarity = model.wmdistance(['глухой_ADJ'], ['темнота_NOUN'])
# print(similarity)


morph = pymorphy2.MorphAnalyzer()

stop_words = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его',
              'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
              'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже',
              'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
              'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их',
              'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда',
              'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти',
              'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
              'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про',
              'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой',
              'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю',
              'между',

              'быть', 'мой', 'наш', 'ваш', 'их', 'его', 'её', 'их',
              'этот', 'тот', 'где', 'который', 'либо', 'нибудь', 'нет', 'да'
              ]

punctuation_signs = [',', '.']


def morph_parse(w):
    # TODO: вероятно не разбирает ещё какие-то части речи
    grammars = {
        'ADJF': 'ADJ',
        'ADVB': 'ADV',
        'INFN': 'VERB',
    }
    p = morph.parse(w)
    try:
        p = max(p, key=lambda x: (x.score, x.methods_stack[0][2]))  # взято из статьи на habr
    except Exception:
        p = p[0]
    return p.normal_form, '_' + grammars.get(p.tag.POS, p.tag.POS) if p.tag.POS else None


def get_clean_rusvectores_words(text):
    # text = """Уже почти два месяца не работают светодиодные светильники по ул. Боголюбова от дома №84  до да №105. Заявку подавали неоднократно ,но увы ни каких изменений.Ходить в вечернее время суток по улице становиться страшно т.к бегают бездомные собаки , которые охраняю стройку на улице.Надеемся что Вы нам поможите решить нашу проблему.А то живем как в глухой деревне в темноте."""
    words = []
    text = text.replace('.', '. ')
    sentences = nltk.sent_tokenize(text, language="russian")
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence, language="russian")
        for t in tokens:
            w, POS = morph_parse(t.strip(',.')) #
            if not POS:  # TODO: просто пропускаем слова без части речи
                continue
            if w and w not in stop_words and w not in punctuation_signs:
                words.append(w+POS)
    return words


def get_distance_matrix(input_list):
    m = {}
    for d1 in input_list:
        for d2 in input_list:
            if d1['id'] > d2['id']:
                row = m.get(d1['id'], {})
                # не все слова будут в модели
                distance = model.wmdistance(d1['cleaned_rusvectores_words'], d2['cleaned_rusvectores_words'])
                row[d2['id']] = distance
                m[d1['id']] = row
    return m


def run():
    input_list = []
    with open('input/NashDomRyazan-29-03-2019.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # skip header
        for row in csv_reader:
            orig_text = row[1]
            cleaned_rusvectores_words = get_clean_rusvectores_words(orig_text)
            input_list.append(
                dict(
                    id=row[0],
                    orig_text=orig_text,
                    category=row[2],
                    theme=row[3],
                    executor=row[4],
                    cleaned_rusvectores_words=cleaned_rusvectores_words
                )
            )
            if row[0] == '3':
                break
    print(get_distance_matrix(input_list))


if __name__ == '__main__':
    run()
