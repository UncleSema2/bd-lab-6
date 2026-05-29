# OpenFoodFacts KMeans Clustering on PySpark

Кластеризация продуктов из OpenFoodFacts с помощью PySpark и алгоритма K-средних.

## Установка

```bash
pip install -r requirements.txt
```

Данные положить в `data/food.parquet` (или указать свой путь в `config.yaml`).

## WordCount

Проверка работоспособности Spark:

```bash
python test/wordcount.py
```

## Кластеризация

```bash
python src/main.py --config config.yaml
```

Гиперпараметры задаются в `config.yaml`.
