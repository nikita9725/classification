# Financial News Sentiment Classification

Учебный ML mini-product для классификации коротких англоязычных финансовых
фраз по трём классам: `negative`, `neutral` и `positive`.

Проект реализует единый воспроизводимый workflow:

```text
Financial PhraseBank
        ↓
валидация и очистка текста
        ↓
TF-IDF-признаки
        ↓
baseline и сравнение улучшений
        ↓
оценка и анализ ошибок
        ↓
сохранённая модель и inference
```

## Результат

Основная метрика — macro F1: она одинаково учитывает каждый класс при
несбалансированном датасете.

| Модель | CV macro F1 | Test accuracy | Test macro F1 | Прирост к baseline |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression (baseline) | 0.7226 | 0.8333 | 0.7561 | — |
| Настроенный TF-IDF + Logistic Regression | 0.7871 | 0.8565 | 0.7936 | 4.95% |
| TF-IDF + balanced Logistic Regression | 0.7982 | 0.8478 | 0.8000 | 5.81% |
| **TF-IDF + balanced LinearSVC** | **0.8070** | **0.8623** | **0.8096** | **7.07%** |

Победитель выбран по среднему macro F1 пятифолдовой cross-validation только на
train-части. Test set использован один раз для итоговой оценки, поэтому выбор
модели не подстраивался под отложенные данные.

Финальный артефакт `models/best_model.joblib` содержит тот же pipeline,
переобученный на всех 3448 очищенных примерах.

## Быстрый старт

Требуется Python 3.11 или новее и установленный
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
```

Готовая модель уже находится в репозитории, поэтому inference не требует
скачивания датасета или повторного обучения:

```python
from finnews_sentiment.models.predict import load_model, predict_texts

model, metadata = load_model("models/best_model.joblib")
predictions = predict_texts(
    model,
    [
        "The company reported strong profit growth.",
        "The board will meet on Tuesday.",
        "Operating loss increased and the company issued a profit warning.",
    ],
)
print(predictions)
```

`predict_texts` принимает одну строку или последовательность строк и возвращает:

- исходный и очищенный текст;
- предсказанный класс;
- decision score каждого класса;
- разницу между двумя наибольшими score (`decision_margin`).

Decision scores и margin полезны для ранжирования неоднозначных примеров, но не
являются вероятностями или калиброванной уверенностью.

### Inference для CSV

Во входном CSV обязательна колонка `text`. Остальные колонки, дубликаты и порядок
строк сохраняются:

```python
from finnews_sentiment.models.predict import predict_for_file

predictions = predict_for_file(
    "models/best_model.joblib",
    "new_financial_news.csv",
    "predictions.csv",
)
```

Пустые и нестроковые тексты отклоняются с явной ошибкой.

## Данные и preprocessing

Используется
[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank),
конфигурация `sentences_75agree`. После очистки датасет содержит 3448 уникальных
примеров:

| Класс | Примеров |
|---|---:|
| negative | 420 |
| neutral | 2141 |
| positive | 887 |

Очистка приводит текст к нижнему регистру, декодирует HTML, удаляет HTML-теги и
нормализует пробелы. Числа, валюты, проценты и пунктуация сохраняются как важные
для финансового домена признаки. Также удаляются пустые строки и дубликаты,
проверяются допустимые классы и конфликтующие метки.

Данные распространяются по лицензии
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Загруженные
CSV исключены из Git и воспроизводятся ноутбуками. При первом запуске загрузки
нужен доступ к Hugging Face; затем может использоваться локальный кэш.

## Обучение и воспроизведение

Все ноутбуки можно открыть через `uv run jupyter lab`. Они хранятся выполненными,
поэтому выводы и графики доступны непосредственно в репозитории.

| Этап | Ноутбук | Основной результат |
|---|---|---|
| Данные и EDA | `notebooks/01_data_eda.ipynb` | raw CSV и графики дня 1 |
| Preprocessing | `notebooks/02_text_preprocessing.ipynb` | `data/processed/clean_news.csv` |
| Baseline | `notebooks/03_baseline_model.ipynb` | baseline-модель и метрики |
| Улучшения | `notebooks/04_model_improvement.ipynb` | сравнение моделей и победитель |
| Error analysis | `notebooks/05_error_analysis.ipynb` | confusion matrix и ошибки |
| Финальный inference | `notebooks/06_inference_pipeline.ipynb` | финальная модель и demo |

Для воспроизведения конкретного этапа без интерфейса:

```bash
uv run jupyter nbconvert \
  --execute \
  --to notebook \
  --inplace notebooks/01_data_eda.ipynb
```

Замените имя ноутбука на нужный этап. Ноутбуки 3–6 самостоятельно загружают
исходный датасет и выполняют preprocessing; запуск предыдущих ноутбуков для них
не обязателен. Для полного финального обучения выполните
`notebooks/06_inference_pipeline.ipynb`.

## Анализ ошибок

Итоговый LinearSVC правильно классифицировал 595 из 690 test-примеров:

- accuracy — `0.8623`;
- macro F1 — `0.8096`;
- ошибок — `95` (`13.77%`);
- главный сценарий — `positive → neutral`, 41 ошибка.

Косвенный положительный смысл часто выглядит для unigram TF-IDF как нейтральное
фактическое сообщение. Ошибочные ответы также имеют меньший средний decision
margin (`0.606` против `1.287` у правильных), поэтому margin пригоден для
очереди ручной проверки. Выводы error analysis не использовались для повторной
настройки модели на test set.

## Demo

Полный результат находится в `reports/day06/demo_predictions.csv`.

| Текст | Предсказание | Margin |
|---|---|---:|
| Strong profit growth and improved outlook | positive | 2.155 |
| The board will meet to review the report | neutral | 1.022 |
| Operating loss increased and profit warning issued | negative | 0.680 |
| Sales rose, but operating profit declined | positive | 0.032 |

Последний пример намеренно содержит смешанные сигналы. Почти нулевой margin
показывает, что такое предсказание следует интерпретировать осторожно.

## Структура проекта

```text
src/finnews_sentiment/
├── data/load_data.py              # загрузка и валидация датасета
├── features/preprocess.py         # очистка и подготовка признаков
├── models/train_model.py          # baseline (train_baseline)
├── models/train_improved.py       # сравнение улучшений и LinearSVC
├── models/error_analysis.py       # метрики, ошибки и confusion matrix
├── models/predict.py              # загрузка модели и inference
└── visualization/plots.py         # EDA-графики
notebooks/                         # воспроизводимые этапы 1–6
models/                            # сохранённые sklearn pipelines
reports/                           # метрики, графики и demo
tests/                             # unit и regression-тесты
```

Имена модулей соответствуют рекомендуемой структуре задания; `train_model.py`
реализует baseline-обучение, то есть выполняет роль `train_baseline.py`.

## Регрессионные проверки

Полный набор тестов работает без сети:

```bash
uv run pytest -q
```

Только сквозные regression-проверки:

```bash
uv run pytest -q -m regression
```

Regression-набор проверяет единый workflow дней 1–6 на детерминированных данных,
согласованность опубликованных JSON/CSV-отчётов и воспроизводимость demo с
готовой моделью. Полная сетка гиперпараметров и скачивание датасета намеренно не
запускаются при каждом тесте.

## Ограничения

- Модель предназначена для коротких англоязычных финансовых фраз. Русский язык,
  длинные документы и другие предметные области находятся вне её домена.
- TF-IDF учитывает лексические признаки, но не моделирует глубокий контекст,
  причинность, сарказм и сложную смешанную тональность.
- В датасете преобладает класс `neutral`; `class_weight="balanced"` уменьшает,
  но не устраняет последствия дисбаланса.
- Метрики получены на одном фиксированном стратифицированном разбиении
  Financial PhraseBank и не гарантируют такое же качество на новых источниках.
- LinearSVC не выдаёт вероятности. Для вероятностных решений потребуется
  отдельная калибровка на независимых данных.
- `joblib` может выполнить произвольный код при загрузке. Загружайте модельные
  артефакты только из доверенного источника.
