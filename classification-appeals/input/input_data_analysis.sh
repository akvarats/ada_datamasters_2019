# получаем справочник категорий
clickhouse-local \
     --file=NashDomRyazan-29-03-2019.csv \
     --input-format=CSVWithNames \
     --table=appeals \
     --structure='id String, text String, category String, theme String, executor String' \
     --query='SELECT category f FROM appeals GROUP BY f ORDER BY f' > categories.csv

# есть запись где не заполнен исполнитель
clickhouse-local \
     --file=NashDomRyazan-29-03-2019.csv \
     --input-format=CSVWithNames \
     --table=appeals \
     --structure='id String, text String, category String, theme String, executor String' \
     --query="SELECT id, text, category, theme, executor FROM appeals WHERE executor=''"
# 3247	Жители военного городка живут без тепла и горячей воды. Мерзнут дети, взрослые и старики.	Многоквартирные дома	Перебои/Отсутствие теплоснабжения     

# получаем справочник исполнителей
clickhouse-local \
     --file=NashDomRyazan-29-03-2019.csv \
     --input-format=CSVWithNames \
     --table=appeals \
     --structure='id String, text String, category String, theme String, executor String' \
     --query="SELECT executor FROM appeals WHERE executor!='' GROUP BY executor ORDER BY executor" > executors.csv
