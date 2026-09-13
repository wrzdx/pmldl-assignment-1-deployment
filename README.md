# PMLDL Assignment 1 — Automated Iris Deployment

Полный MLOps-процесс: **обработка данных → обучение и оценка → Docker API + веб-приложение**.
`python pipeline.py` немедленно запускает все три этапа, затем повторяет их каждые **300 секунд**.

## Что реализовано

| Критерий задания | Реализация |
|---|---|
| Data engineering | DVC: чтение CSV, очистка, стратифицированный train/test split, сохранение CSV |
| Model engineering | StandardScaler + LogisticRegression, accuracy и macro-F1, MLflow, файл joblib |
| Deployment | Два Dockerfile, отдельные FastAPI и Streamlit контейнеры, запросы приложения к API |
| Automation | Последовательный планировщик: DVC `--force`, сборка и обновление контейнеров, проверки |
| Structure | Отдельные модули данных, модели, API, приложения, тесты, CI, инструкции |

## Быстрый старт

Требования: **Python 3.12**, Git, **Docker Engine / Docker Desktop с Linux containers** и Docker Compose v2+.
Docker должен быть запущен. Интернет требуется при первой установке Python-зависимостей и сборке образов.
Команды выполняются из корня этого проекта. Файл raw CSV уже включён.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pipeline.py
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python pipeline.py
```

После сообщения `All three stages passed`:

- Приложение: [http://localhost:8501](http://localhost:8501)
- Документация API: [http://localhost:8000/docs](http://localhost:8000/docs)
- Состояние API: [http://localhost:8000/health](http://localhost:8000/health)
- Версия модели и метрики: [http://localhost:8000/model](http://localhost:8000/model)

В приложении заполните четыре поля и нажмите **«Определить вид»**. Будут показаны
вид ириса, вероятности трёх классов и ID обучения. Значения 5.1, 3.5, 1.4, 0.2 дают `setosa`.
Приложение отправляет HTTP-запрос в API; модель загружается только в API-контейнере.

Для остальных команд ниже `python` означает интерпретатор активированного `.venv`;
в PowerShell можно всегда использовать `.\.venv\Scripts\python.exe`.

## Расписание и остановка

```bash
python pipeline.py                  # сразу, затем каждые 5 минут
python pipeline.py --once           # один полный запуск
python pipeline.py --max-runs 2     # два полных запуска с интервалом 5 минут
python pipeline.py --interval 600   # интервал 10 минут для медленной машины
```

Планировщик работает на хосте: терминал нужно оставить открытым, компьютер — включённым.
Это независимый от Codex процесс. Повторные запуски выполняют обработку, обучение,
сборку и развёртывание даже при неизменных исходных данных. Новый MLflow run ID
попадает в модель внутри нового API-образа. После развёртывания проверяется совпадение ID.
`--force-recreate` обновляет оба контейнера; в момент обновления возможен короткий перерыв.

Если запуск занимает больше интервала, пропущенные слоты не догоняются: следующий
запуск переносится на ближайший будущий слот. Одновременные запуски блокируются
файловой блокировкой. При ошибке обработки/обучения развёртывание не начинается.
При ошибке сборки/развёртывания успех не записывается; автоматического отката нет.
Обычный планировщик записывает ошибку и повторяет попытку в следующем слоте;
`--once` и ограниченный `--max-runs` завершаются с ненулевым кодом.

`Ctrl+C` останавливает планировщик, оставляя приложение работающим. Затем остановить контейнеры:

```bash
docker compose -f iris_ml/deployment/docker-compose.yml down
```

Если порты заняты, задайте `API_PORT` и `APP_PORT` в окружении терминала **до запуска**:

```powershell
$env:API_PORT = '18000'
$env:APP_PORT = '18501'
.\.venv\Scripts\python.exe pipeline.py
```

В Linux/macOS: `API_PORT=18000 APP_PORT=18501 python pipeline.py`.
Оба сервиса публикуются только на localhost; между контейнерами используется `http://api:8000`.

## Данные и отсутствие утечки

Выбран разрешённый датасет Iris: 150 исходных примеров, 4 измерения, 3 класса.
Источник и способ экспорта описаны в [data/raw/README.md](data/raw/README.md).

1. Чтение файла, проверка столбцов, преобразование признаков в числа.
2. Нечисловые, бесконечные и неположительные измерения считаются пропусками;
   записи с неизвестными метками и точные дубликаты удаляются.
3. Стратифицированное разбиение 80/20 с `random_state=42`.
4. Медианы считаются **только на train**, ими заполняются пропуски в train и test.
5. Из train удаляются строки за границами `[Q1−3×IQR, Q3+3×IQR]`, рассчитанными на train.
   Test-выбросы сохраняются для честной оценки. Очистка фиксируется в отчёте.
6. StandardScaler обучается только на train внутри sklearn Pipeline; тот же
   преобразователь применяется к test и включён в упакованную модель для API.

Accuracy и macro-F1 относятся к небольшому фиксированному учебному test split.
Гиперпараметры не подбираются по test. Для демонстрации MLOps сложная модель не нужна.

## Артефакты и MLflow

| Файл / каталог | Содержимое |
|---|---|
| `data/raw/iris.csv` | Исходные данные, включены в Git |
| `data/processed/train.csv`, `test.csv` | Очищенные train/test |
| `models/model.joblib` | StandardScaler + классификатор и метаданные |
| `models/metadata.json` | Run ID, UTC-время, версии, хеш raw, метрики |
| `metrics/metrics.json` | Accuracy, macro-F1 |
| `metrics/evaluation.json` | Матрица ошибок и отчёт по классам |
| `metrics/data_quality.json` | Пропуски, выбросы, размеры, параметры очистки |
| `mlruns/` | Локальные MLflow experiments: параметры, метрики, артефакты модели |
| `logs/runs.jsonl` | История запусков: UTC-время, длительность, статус, run ID |
| `logs/pipeline.log` | Журнал этапов планировщика |

MLflow сохраняет joblib как артефакт `model/model.joblib`; это собственный формат
упаковки проекта, а не MLflow pyfunc. Для задания достаточно `mlflow-skinny`:
отдельный tracking server не требуется. Историю можно прочитать через `MlflowClient`
или установить **опционально** `mlflow==2.22.0` и открыть UI:

```bash
python -m pip install mlflow==2.22.0
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

MLflow URI вычисляется как абсолютный `file://` URI текущей папки проекта.
При переносе проекта запускайте обучение заново; сохранённые локальные эксперименты
могут ссылаться на старые абсолютные пути. В Git они не включаются.
Генерируемые модели также исключены из Git и воспроизводятся первой командой запуска.
DVC работает в `no_scm` режиме, поэтому распакованный ZIP запускается без `git init`.

## Проверки и демонстрация для TA

```bash
python -m pytest -q
python pipeline.py --max-runs 2
docker compose -f iris_ml/deployment/docker-compose.yml ps
docker compose -f iris_ml/deployment/docker-compose.yml logs --tail 50
```

Тесты проверяют очистку грязных данных, отсутствие утечки из test, сохранение модели,
MLflow-метрики и артефакты, API и ошибки входных данных, кнопку Streamlit и ошибку сети,
пропуск просроченных слотов и прекращение процесса до deployment при ошибке обучения.
Полный запуск дополнительно проверяет health обоих контейнеров, предсказание,
совпадение текущего run ID и доступность API из контейнера приложения.

Для защиты покажите:

1. Файл `dvc.yaml` и обработанные train/test CSV.
2. Метрики и запись MLflow, файл модели.
3. Два работающих контейнера и предсказание через веб-форму.
4. Две строки `success` в `logs/runs.jsonl` с разными run ID и интервалом около 300 секунд.
5. Обновлённый run ID в `/model` после второго запуска.

GitHub Actions запускает тесты и **один полный Docker-процесс** на push/PR.
Это CI-проверка; постоянное пятиминутное расписание обеспечивает `pipeline.py`
на машине, где будет демонстрация. GitHub Actions не используется как постоянный хост.

## Структура

```text
iris_ml/                         # Python-пакет; имя не конфликтует со stdlib code
  common.py
  datasets/{export_raw,prepare}.py
  models/train.py
  deployment/
    api/{main.py,Dockerfile}
    app/{main.py,Dockerfile}
    docker-compose.yml
data/{raw,processed}/
models/
metrics/
tests/test_pipeline.py
.github/workflows/ci.yml
.dvc/config
dvc.yaml
pipeline.py
requirements*.txt
```

Airflow и notebooks не нужны: обработка и обучение организованы через DVC,
расписание и deployment — через Python. Все три этапа объединены одним запуском.

## Публикация для сдачи

Задание сдаётся ссылкой на **public GitHub repository**. Если репозиторий ещё не опубликован,
из этой папки выполните (после `gh auth login`, если требуется):

```bash
git init -b main
git add .
git commit -m "Implement automated Iris training and Docker deployment"
gh repo create pmldl-assignment-1-deployment --public --source . --remote origin --push
```

Если локальный Git уже инициализирован, пропустите `git init`; если commit уже создан,
публикуйте текущий commit. Если имя на GitHub занято, выберите другое свободное имя.

## Документация инструментов

- [DVC repro](https://dvc.org/doc/command-reference/repro) — воспроизведение этапов и `--force`.
- [Docker Compose up](https://docs.docker.com/reference/cli/docker/compose/up/) — сборка, обновление и ожидание health.
- [MLflow Tracking API](https://mlflow.org/docs/latest/ml/tracking/tracking-api) — параметры, метрики и артефакты.
- [scikit-learn Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).
