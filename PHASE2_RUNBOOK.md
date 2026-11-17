# Phase 2: что, где и как запускать

Внутренняя инструкция для команды. Код уже готов в ветке `phase2`. Осталось прогнать обучение на Kaggle, собрать `results/`, построить таблицы и закоммитить результаты. Время прогонов указано по оценке, его никто не замерял.

## 0. Подготовка (один раз)

1. Залить ветку `phase2` на GitHub. Ноутбуки в Kaggle загружать из этой ветки: *File → Import Notebook* или копированием файла.
2. Во всех ноутбуках выставить *Accelerator = GPU T4 ×2* (или P100).
3. Квота Kaggle — около 30 GPU-часов в неделю на аккаунт, а одна сессия идёт максимум 12 часов. Прогоны можно раскидать по аккаунтам участников, все они независимы. Публиковать веса (шаг с `PUBLISH_MODELS = True`) лучше с аккаунта `eeee13`: на него уже ссылается ансамбль. Если публикуете с другого аккаунта, поменяйте `KAGGLE_USERNAME` и пути в `ensemble-pipeline.ipynb`.

## 1. Прогоны обучения

| # | Ноутбук | Что поменять перед *Run All* | Internet | Время (оценка) | Что скачать из Output |
|---|---|---|---|---|---|
| 1 | `mbart50.ipynb` | `SEED = 42`, `PUBLISH_MODELS = True` | ON | ~1–1.5 ч | `results/` |
| 2 | `mbart50.ipynb` | `SEED = 43` | OFF | ~1–1.5 ч | `results/` |
| 3 | `mbart50.ipynb` | `SEED = 44` | OFF | ~1–1.5 ч | `results/` |
| 4 | `byte-level-seq2seq.ipynb` | `SEED = 42`, `PUBLISH_MODELS = True` | ON | ~4–6 ч | `results/` |
| 5 | `byte-level-seq2seq.ipynb` | `SEED = 43` | ON | ~2–3 ч | `results/` |
| 6 | `byte-level-seq2seq.ipynb` | `SEED = 44` | ON | ~2–3 ч | `results/` |
| 7 | `pipeline_QwenQLoRa.ipynb` | в последней ячейке `PUBLISH_MODELS = True` | ON | ~4–8 ч | `results/` |
| 8 | `ensemble-pipeline.ipynb` | см. раздел 2 | OFF | ~1–2 ч | `results/`, `submission.csv` |

Входные данные:
- везде: соревновательный датасет (*Add Input → Competitions → deep-past-initiative-machine-translation*);
- для mBART дополнительно: модель `nensipansuriya1311/facebookmbart-large-50-many-to-many-mmt`.

Что важно не перепутать:
- `USE_TRACKING` оставляйте `False`, тогда Secrets не нужны. `True` включайте, только если хотите логировать в MLflow; тогда нужны Secrets из README.
- `EXECUTION_MODE` в mBART оставьте `"offline"`. Базовая модель берётся из Input, и это работает даже при включённом интернете (интернет в прогоне 1 нужен только для загрузки весов).
- В начале выполнения ноутбук печатает строку `[shared split] train=1405 rows, val=156 rows (156 documents) ...`. Если вместо неё ошибка `Validation split differs`, значит подключён не тот `train.csv`. Не продолжайте прогон.
- Если ByT5 падает с CUDA OOM: в `MODEL_CONFIGS` поставьте `"batch_size": 1, "grad_accum": 8` (эффективный батч останется 8). Если и это не помогает, уменьшите `MAX_TARGET_LEN` до 1024. Об изменении обязательно напишите мне: оно должно попасть в отчёт. Для mBART при OOM аналогично `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`.
- Из каждого прогона скачивайте папку `results/` целиком. Имена файлов уникальные (`__seed<k>`), поэтому всё можно складывать в одну папку.

## 2. Ансамбль (после прогонов 1, 4 и 7)

1. Прогоны 1, 4 и 7 с `PUBLISH_MODELS = True` создают модели на Kaggle с такими вариациями:
   - `akkadian-mbart/pytorch/mbart50-seed42`
   - `akkadian-byt5/pytorch/byt5-seed42`
   - `akkadika/pytorch/qwen-lora-seed42`
2. Откройте `ensemble-pipeline.ipynb` и подключите через *Add Input*:
   - эти три модели;
   - `eeee13/qwen2.5-7b-instruct` (`base-v1`);
   - датасет `eeee13/bitsandbytes`;
   - соревновательные данные.
3. Сверьте пути в ячейке с `MBART_OFFLINE_PATH` с панелью Input. Если номер версии не `/1`, поправьте его.
4. Internet OFF, *Run All*. Скачайте `results/` и `submission.csv`. По желанию отправьте `submission.csv` в соревнование и запишите Public score: он пойдёт в отчёт.

## 3. Сбор результатов (локально)

```bash
git switch phase2
# положить train.csv соревнования в корень репозитория (в git его не добавлять)
# сложить содержимое всех скачанных results/ в ./results/
pip install -r requirements.txt
python tools/evaluate_all.py results/preds_val_*.csv --out results
python tools/analyze_runs.py --results results
python -m pytest tests/
python tools/sync_shared_cells.py --check
git add results/ && git commit -m "Phase 2 results"
```

После этого в `results/` появятся:
- `summary.md`: главная таблица сравнения (mean ± std по сидам и 95% CI);
- `paired_tests.csv`: значимость отличия лучшей модели от остальных;
- `experiment_registry.md`: какой прогон дал какую строку (это для W3);
- `figures/curves_*.png`: кривые обучения;
- `error_analysis.md`: анализ ошибок (длина документа, повреждённый текст, числа, имена, худшие примеры).

Пришлите мне `results/` или запушьте его. Дальше по этим файлам пишутся отчёт и `fixes.pdf`.

## 4. Что изменено в коде и зачем (для fixes.pdf)

| Слабость | Что исправлено | Где |
|---|---|---|
| W1: разные сплиты | Один сплит по документам для всех моделей с проверкой SHA-256. Qwen раньше делил по строкам, ByT5 использовал `GroupShuffleSplit`. | `common/split.py`, ячейки `SHARED CELL` во всех ноутбуках, `data/val_ids.csv` |
| W1: разные метрики | У mBART и ансамбля был chrF без словесных n-грамм. Теперь везде chrF++, идентичный sacreBLEU. | `common/metrics.py`, `tests/test_metrics.py` |
| W1: неравные условия | ByT5-бейзлайн обучается на полном train, а не на 85%. Длины выбраны по правилу p95 для каждой модели (раньше ByT5 обрезал 66–71% документов). Итоговый балл считается по полным эталонам, а не по обрезанным. Чекпоинт выбирается по GM (у mBART раньше по chrF). | ноутбуки ByT5 и mBART, README → *Length limits* |
| W1: слабая статистика | 3 сида для mBART и ByT5, bootstrap CI, парный bootstrap-тест | `tools/evaluate_all.py` |
| W1: утечка в ансамбле (нашли сами) | ByT5 и Qwen обучались на других сплитах, поэтому часть валидации ансамбля была у них в train. Теперь все модели переобучены на общем сплите. | `ensemble-pipeline.ipynb` |
| W1: баги ансамбля (нашли сами) | Ансамбль подавал Qwen другой промпт, чем при обучении. Для mBART не задавался язык `en_XX`, а `forced_bos` был функцией вместо id. Длины генерации не совпадали с обучением. | `ensemble-pipeline.ipynb`, ячейка с `GEN_LIMITS` |
| W1: баг ByT5 (нашли сами) | `range(N_MODELS-1)`: ансамбль ByT5 фактически состоял из одной модели | `byte-level-seq2seq.ipynb` |
| W1: баг Qwen (нашли сами) | При батче больше 1 промпт обрезался неправильно (правый паддинг) | `pipeline_QwenQLoRa.ipynb`, `generate_predictions` |
| W2: `HF_REPO_NAME` офлайн | Секрет читается только при `IS_ONLINE` | `mbart50.ipynb` |
| W2: загрузка Qwen | Загрузка выполняется только при `PUBLISH_MODELS = True` | `pipeline_QwenQLoRa.ipynb`, последняя ячейка |
| W2: `test_preds_per_model` | Тестовые предсказания теперь генерируются | `byte-level-seq2seq.ipynb` |
| W2: веса ByT5 | Публикуются на Kaggle Models, ручной шаг больше не нужен | `PUBLISH_MODELS`, README |
| W2: версии пакетов | Убран `pip -U`, версии и GPU каждого прогона пишутся в `results/env_*.json` | `common/env.py` |
| W3: связь таблиц с прогонами | JSON-запись на каждый прогон и реестр экспериментов | `common/runlog.py`, `tools/analyze_runs.py` |
| W3: анализ ошибок | Скрипт анализа ошибок и кривые обучения | `tools/analyze_runs.py` |
| Прочее | Старый CLI Qwen (`src/`) с отдельным сплитом перенесён в `legacy/` | `legacy/README.md` |
