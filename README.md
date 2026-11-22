# Bank Marketing

ДЗ 1 курса «Инженерные практики в ML и анализе данных». Проект обучает
базовую модель для прогноза подписки клиента на срочный вклад (`y`).

## Данные

Используется `bank-full.csv` из [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
45 211 наблюдений. Авторы — S. Moro, P. Rita и P. Cortez.
[DOI](https://doi.org/10.24432/C5K306). Лицензия — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Скрипт скачивает исходный архив без токена, проверяет схему и число строк
и сохраняет CSV в `data/raw/`.

## Запуск

Нужны Git, [uv](https://docs.astral.sh/uv/getting-started/installation/) и доступ
к интернету. Выполните команды из корня клонированного репозитория.
`uv` при необходимости установит Python 3.12.

```sh
uv sync --locked
uv run python -m src.get_data
uv run pre-commit install
uv run pre-commit run --all-files
uv run python -m src.train
```

Для отдельного запуска линтера и проверки форматирования:

```sh
uv run ruff check
uv run ruff format --check
```

Обучение сохраняет полный pipeline с преобразованиями признаков в
`artifacts/model.pkl` и метрики отложенной выборки в `artifacts/metrics.json`.
Для числовых признаков применяется масштабирование, для категориальных — one-hot
кодирование. Логистическая регрессия оценивается на стратифицированном разбиении
80/20 с `random_state=42`. Признак `duration` исключён, так как он известен после
звонка. ROC AUC и average precision оценивают ранжирование, F1 использует
стандартный порог модели. Случайное разбиение служит для сравнения экспериментов
на этой таблице. Для оценки на будущих кампаниях нужна отдельная временная проверка.

[Отчёт по ДЗ 1](reports/hw-1.md)
