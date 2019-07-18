pip install -r requirements.txt

Положить https://github.com/Mottl/ru_punkt в .venv/nltk_data (TODO: git clone)

Положить распакованную модель http://vectors.nlpl.eu/repository/11/180.zip в models (TODO: wget, unzip)

## Подготовка модели

```bash
python ./generate_model.py --corpus="input/NashDomRyazan-29-03-2019.csv" --word2vec="models/180/model.bin" --out="models/md_717_180"
```

## Запуск оценщика

```bash
python ./estimator.py --model=models/md_717_180 --range=0:300
```