# RAG Pharma Dissolution Prep

## Открытые экспериментальные данные ВЭЖХ

### Открытые сигналы, ширины пиков, Rs и полные тексты

```bash
# Расширенный сбор: QC-хроматограммы, статьи, дополнительные сигнальные наборы
.venv/bin/python scripts/run_public_pipeline.py --extended-data
# Отчёт об объёмах и действительной готовности данных
.venv/bin/python scripts/report_extended_data.py
```

`collect_chromatograms.py` создаёт индекс всех 2870 сырых сигналов U Liège
(13,47 ГБ) и по умолчанию загружает все 306 QC-DAD сигналов (1,86 ГБ).
Для полного архива всех отдельных стандартов, холостых проб и ELSD есть
`--scope all`; это отдельная многогигабайтная загрузка.
SHA256 проверяется по метаданным источника; большие файлы скачиваются с
возобновлением, файлы и истинные повторы не удаляются.

`extract_chromatogram_peaks.py` читает оба варианта экспорта Empower `.arw`,
учитывает pH-коды 35/65 и извлекает ширины пиков на половине высоты относительно
локальной базовой линии. Выбирается один общий DAD-канал для всех QC-компонентов
инъекции. Rs рассчитывается только между пиками этой же реальной QC-смеси
по half-height формуле 1,18·ΔtR/(wh1+wh2). На разных файлах смесь не собирается.
Идентичность трёх компонентов — предварительная; часть инъекций дополнительно
сопоставляется с таблицей авторов с проверкой всех трёх tR. При разных количествах
запусков соответствие не навязывается. Результаты — кандидаты для проверки,
не авторские значения Rs, и не добавляются в обучение автоматически.
QC состоит из uracil/3-cyanopyridine/ibuprofen; это не TFV/FTC и не набор сложных
близких разделений. Оценки tailing, давления и количественной чистоты не выдумываются.

`collect_open_methods.py` получает полные тексты через Europe PMC и сохраняет
только тексты с явно подтверждёнными CC BY / CC BY-SA / CC0. NC/ND и неясные
лицензии фиксируются в manifest как исключённые. Таблицы сохраняют исходную
структуру ячеек; числовые фрагменты с DOI/node_id требуют проверки связи
вещество–метод–измерение и не являются готовыми training labels.

Дополнительно загружаются три CC BY 4.0 набора для обработки реальных сигналов:
Zenodo 10082436 (реакционные HPLC-хроматограммы), Zenodo 14236878 (разработка
метода анализа продуктов окисления глицерина), Figshare 22182694 (HPLC-DAD
многокомпонентных образцов). Архивы проверяются, их файлы индексируются;
код из архивов не исполняется, данные не подмешиваются к QSRR без адаптера.

`import_reported_suitability.py` импортирует проверенную структуру таблицы 8
PMC12216858: 3 записи tR/tailing/N/k и 2 авторских значения Rs для FTC/DAC/LDV.
Эти значения отделены от производных Rs по сигналам и от RSD% робастности.
Записи остаются summary уровня метода с run_id=null; не считаются смесью TFV/FTC.
Из авторского k можно вычислить log10(k), но это не измеренный t0.

Выходные файлы: `data/external/chromatogram_inventory.json`,
`data/external/chromatogram_downloads.json`, `data/external/open_methods/`,
`data/external/signal_benchmarks/`, `data/processed/public_retention/chromatograms/`.
Реальный пример сигнала — `qc_real_chromatogram.png`, итоговый отчёт —
`reports/open_data_expansion/REPORT.md`.

### Полный сбор подключённых открытых источников

```bash
.venv/bin/python scripts/run_public_pipeline.py
# Дополнительно обучить первые модели (до 12000 строк каждого источника):
.venv/bin/python scripts/run_public_pipeline.py --train
```

Сейчас подключены все 421 системы RepoRT (183818 наблюдений), METLIN SMRT
(79957 пригодных наблюдений; 81 запись с невалидной структурой/RT осталась
в исходном CSV) и сводная таблица U Liège сразу по трём приборам (2718
наблюдений со структурами; 2 строки с названием `arginie` сохранены отдельно).
PubChem используется для идентификации 96 названий U Liège; автоматическое
сопоставление требует проверки солей и стереохимии.
Europe PMC предоставляет библиографию по тенофовиру/эмтрицитабину (1724 записи
при текущей выгрузке), а не автоматически подтверждённые экспериментальные метки.

Источники сохранены раздельно: `data/external/`; нормализованные данные —
`data/processed/public_retention/{report,metlin_smrt,uliege}/`.
Условия системы хранятся один раз в `systems.json`, наблюдения — в
`observations.jsonl`. Каталог лицензий и доступности —
`data/external/public_sources.json`; пересечения —
`data/processed/public_retention/corpus_report.json`.
Нельзя складывать размеры этих таблиц: METLIN уже частично входит в RepoRT.
PredRet представлен через 88 систем RepoRT; сайт прямой выгрузки требует входа.

Сырые сигналы U Liège `.arw` — CSV-экспорты Empower, а не изображения камеры.
Полный многогигабайтный набор сигналов доступен через
`collect_public_data.py --include-signals`; обычная команда берёт таблицы и
сохраняет полный индекс файлов, не загружая сигналы. Программы сокращённых
градиентов U Liège ещё требуют связи с сырыми файлами; их нельзя восстанавливать
из одного поля nominal gradient_time. Геометрия колонки и постоянные условия
перенесены из первичной статьи, DOI 10.1016/j.dib.2022.108017.

Первичные модели сохранены отдельно в `models/public_retention/`.
RepoRT и METLIN обучены на воспроизводимых подвыборках по 12000 строк,
U Liège — на всех 2718 сопоставленных строках. Это smoke baseline, не обучение
на всём собранном корпусе и не валидация готовой методики.
Для полного обучения отдельного источника не задавайте `--max-rows`.
RP и HILIC оцениваются отдельно (`train_open_retention.py --mode HILIC`).
Большое число систем проверяется GroupKFold по источнику/публикации;
алиасы SMRT объединены. U Liège группируется по физическому прибору.
Неполная provenance остаётся ограничением для исключения всех дублей.

Проверка импорта:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_public_retention.py'
```

Пайплайн работает без данных компании и API-ключей. Первый подключённый источник —
[RepoRT](https://github.com/michaelwitting/RepoRT), CC BY-SA 4.0.
Скачивается фиксированная версия: повторный запуск использует сохранённый commit
и проверяет SHA-256 файлов. Исходники, лицензия и описание хранятся отдельно от PDF.

```bash
.venv/bin/python scripts/open_retention.py download
.venv/bin/python scripts/open_retention.py build
.venv/bin/python scripts/train_open_retention.py
.venv/bin/python scripts/predict_open_retention.py --system report:0001 --smiles 'CC(=O)Oc1ccccc1C(=O)O'
```

По умолчанию скачиваются первые шесть систем (это проверка интеграции, не весь RepoRT).
Расширение: `download --systems 0007 0008 0009`; затем повторить build и обучение.
Для воспроизводимого повторного обучения сохранить `data/external/report/manifest.json`.

Наблюдения: `data/processed/open_retention/observations.jsonl`.
Отчёт качества: `data/processed/open_retention/quality_report.json`.
Модель и независимые прогнозы: `models/open_retention/`.
В `evaluation.json` две проверки: новые молекулы с группировкой по connectivity
InChIKey и leave-one-system-out. Baseline использует только обучающую часть.
Bootstrap-интервал MAE характеризует оценку ошибки, не интервал отдельного прогноза.

Для пары добавить `--second-smiles '...'`: выводятся два tR и разность времени.
Rs остаётся null: публичный system_id не подтверждает совместное введение веществ,
а ширина пиков отсутствует. Условия импортированной системы выводятся полностью.
Эта модель пока не подключена к прежнему эвристическому ranking: его результаты
нельзя выдавать за проверенные прогнозы новой модели.

### Дальнейшее развитие по техническому брифу

Целевая демонстрация TFV/FTC с исключением обоих веществ и TDF/TAF из обучения:

```bash
.venv/bin/python scripts/demo_tenofovir.py
```

Отчёт Word, Markdown, исходные прогнозы и CSV появляются в `reports/tenofovir_demo/`.
В текущем тесте порядок выхода в системе report:0382 предсказан неверно;
демонстрация показывает состояние прототипа и ограничения, не готовое разделение.

Скрининговый кандидат полного метода на Acclaim RSLC 120 C18:

```bash
.venv/bin/python scripts/propose_pair_method.py
```

Файлы в `reports/tenofovir_method/`: Word, Markdown, JSON всех кандидатов и
CSV градиента. Текущий статус ранжирования — UNRESOLVED; стартовая программа
выбирается правилом из открытого метода, а не объявляется оптимумом CatBoost.
Параметры ввода и детекции явно помечены как скрининговые допущения.

Исследовательская проверка расширенных химических признаков и интервалов:

```bash
.venv/bin/python scripts/evaluate_realism.py
.venv/bin/python scripts/propose_pair_method.py --research-evaluation reports/realism_evaluation/evaluation.json --out reports/tenofovir_method_calibrated
```

Эксперимент использует одинаковые 12000 наблюдений для сравнения двух моделей.
Молекулы разделены на обучение, выбор версии, калибровку и тест, без пересечения.
TFV/FTC/TDF/TAF исключены; знакомый пример TFV/FTC считается диагностикой разработки,
а не нетронутым финальным тестом. На фиксированном тесте новые признаки дали MAE
1,487 вместо 1,600 мин; для TFV/FTC улучшение не подтверждено (1,550 против 1,521).
Номинально 90%-ные интервалы калибруются по максимальной ошибке внутри молекулярной
группы. Радиус 3,92 мин показывает ограниченную практическую точность. Покрытие
групп новых источников составило 76,9%, поэтому перенос этих интервалов на новую
систему не валидирован. Дополнительный анализ на scaffold holdout ещё нужен.
Калиброванный кандидат не заменяет предыдущие отчёты; абсолютные интервалы
распространяются на оценку разницы tR, а не на Rs. Ширины пиков пока не моделируются.

Гидродинамика и затраты цикла (v1 остаётся совместимой с прежними моделями):

```bash
.venv/bin/python scripts/audit_method_efficiency.py
.venv/bin/python scripts/evaluate_realism.py --feature-version chemistry_ph_hydrodynamics_v2 --out reports/hydrodynamics_evaluation
.venv/bin/python scripts/propose_pair_method.py --research-evaluation reports/hydrodynamics_evaluation/evaluation.json --out reports/tenofovir_method_hydrodynamics
```

В v2 добавлены линейная поверхностная скорость (мм/с), явная неизвестная пористость,
крутизна %B/мин, отдельные длительности анализа/промывки/уравновешивания.
При F в мл/мин и d в мм: u_s = (1000 F / 60)/(πd²/4). Скорость u_s/ε вычисляется
только при известной пористости; необходимо различать общую и межчастичную ε.
Не вводится фиктивный строгий оптимум Ван-Деемтера по одному размеру частиц.
Оптимум эффективности требует H/N и сведений о массообмене; это не таргет tR.

На том же тесте v2 дала MAE 1,485 мин против 1,487 у v1 — существенный прирост
не подтверждён. Для TFV/FTC: 1,485 против 1,550; на новых источниках MAE ухудшилась
с 2,696 до 2,809 мин. Эти результаты не доказывают исправление физики прибора.
Расходы и длительности выводятся для аудита, но не заменяют оценку Rs.
`audit_method_efficiency.py` предлагает потоки 0,2/0,35/0,4/0,5 мл/мин с сохранением
объёмной программы и минимумом 3/5 пустых геометрических объёмов для промывки/
уравновешивания. Это явная общая скрининговая политика; неизвестный dwell volume
и давление требуют проверки, никакие tR/Rs новым потокам автоматически не приписываются.

1. Связать сокращённые градиенты U Liège с сырыми сигналами и проверить
   автоматически сопоставленные структуры; затем обучить полные модели.
2. Завершить согласование дублей между источниками и добавить scaffold holdout.
3. Из собранной библиографии Europe PMC извлечь лицензированные методики для
   TFV, TDF, TAF и эмтрицитабина; хранить форму вещества, цитату и DOI.
4. Добавить оценку области применимости по химическому окружению и диапазону
   условий, генерацию кандидатов и ranking с проверкой ограничений колонки.
5. Прогноз разделения проверять только на открытых наблюдениях одной смеси
   с Rs или ширинами пиков. Недостающие измерения не заменять оценками.
6. Добавить интерпретацию с источниками и измеренной ошибкой модели.

Данные компании не являются условием запуска этих этапов. Перенос на её конкретную
систему остаётся отдельной экспериментальной проверкой.

Pipeline for extracting molecule dissolution / solubility / chromatographic solution
information from source documents and preparing it for embeddings.

## 1. Put documents here

Copy PDF, DOCX, TXT, or MD files into:

```bash
data/raw
```

## 2. Install dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 3. Extract chunks without embeddings

```bash
.venv/bin/python scripts/prepare_rag.py
```

The result will be:

```text
data/processed/dissolution_chunks.jsonl
```

Each line contains:

- `id`
- `text`
- `metadata.source`
- `metadata.start_page` / `metadata.end_page`
- `metadata.section`
- matched `metadata.terms`

## 4. Create embeddings in local Chroma

```bash
export OPENAI_API_KEY="..."
.venv/bin/python scripts/prepare_rag.py --embed
```

By default this writes to:

```text
vector_store
```

Collection name:

```text
molecule_dissolution
```

Default embedding model:

```text
text-embedding-3-small
```

## 5. Query the RAG store

Retrieve the most relevant chunks:

```bash
export OPENAI_API_KEY="..."
.venv/bin/python scripts/query_rag.py "Какая среда растворения используется?"
```

Retrieve chunks and generate an answer from them:

```bash
.venv/bin/python scripts/query_rag.py "Какая среда растворения используется?" --answer
```

## 6. Predict chromatography conditions for a new molecule

A predictive scaffold turns the extracted НД text into structured HPLC method
labels and predicts likely conditions for an unseen molecule from its structure.

Pipeline:

```text
dissolution_chunks.jsonl
  -> scripts/extract_conditions.py   structured condition blocks (regex, no API key)
  -> scripts/compound_enrichment.py  optional PubChem/ChEMBL properties cache
  -> scripts/build_dataset.py        one labelled row per molecule + RDKit/external features
  -> scripts/train_model.py          nearest-analog model + leave-one-out report
  -> scripts/predict_conditions.py   conditions for a new SMILES / INN
```

Run it:

```bash
python3 scripts/extract_conditions.py     # -> data/processed/chromatography_conditions.jsonl
python3 scripts/compound_enrichment.py --name ibuprofen
python3 scripts/build_dataset.py          # -> data/processed/labels.csv + dataset.json
python3 scripts/train_model.py            # -> models/nearest_analog.json (+ LOO metrics)
python3 scripts/predict_conditions.py --smiles "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
python3 scripts/predict_conditions.py --inn ibuprofen
python3 scripts/predict_conditions.py --name ketoprofen
```

If `--name` / `--inn` is not found in `data/molecule_map.csv`, the script tries
PubChem and uses the returned SMILES automatically. To only use the local map:

```bash
python3 scripts/predict_conditions.py --name ketoprofen --no-lookup-pubchem
```

Optional external enrichment:

```bash
python3 scripts/compound_enrichment.py --name celecoxib
python3 scripts/compound_enrichment.py          # enrich every mapped molecule
python3 scripts/predict_conditions.py --name celecoxib
```

This writes `data/processed/compound_properties.json` with PubChem / ChEMBL
properties, source names and retrieval dates. Use the all-molecule command only
when it is acceptable to send the mapped INN names to external PubChem/ChEMBL
services. The predictor also works without this cache; it falls back to local
RDKit descriptors.

Predicted output, most actionable first:

- **Reagents** — the chemicals the method needs, grouped by role: organic
  solvents, buffer salts, acids / bases for pH, ion-pairing agents. This is the
  primary output — the reagent shopping list for the method.
- **Mobile phase** — elution mode (isocratic / gradient), the solutions and their
  ratio (e.g. `вода : ацетонитрил = 95 : 5`) parsed into structured components,
  and the gradient step profile when the source table is legible. Falls back to
  the ratio alone when the solution names are unclear.
- **Conditions** — stationary phase, column length / ID / particle size, column
  temperature, detector + wavelength, organic modifier, pH, flow rate, injection
  volume.

Because prediction is *transfer of one real method* (not per-field regression),
the reagents, the mobile-phase ratio, and the conditions come from the same НД
and stay mutually consistent. Missing fields stay missing and are reported
explicitly instead of being borrowed from unrelated methods.

### How it works

- **Labels (y):** `extract_conditions.py` parses the "Хроматографические условия"
  blocks; `build_dataset.py` selects, per document, the single **most complete
  real method** (rather than blending blocks, which produced non-physical values
  and lost the mobile-phase recipe) and back-fills any missing field from the
  document's other blocks.
- **Features (X):** RDKit physicochemical descriptors + a Morgan fingerprint,
  computed from each molecule's SMILES. If available, PubChem and ChEMBL
  properties are added as a separate external profile with provenance.
- **Model:** with only a few dozen labelled molecules, a trained regressor would
  overfit, so the baseline is *nearest-analog transfer* — it returns one coherent
  method from the closest known molecule. Analog similarity combines Morgan
  fingerprint similarity with a descriptor correction for chromatographically
  relevant properties such as LogP, TPSA, MW, formal charge and H-bonding. It
  Structures are first standardized with RDKit `rdMolStandardize` (cleanup,
  largest fragment, neutral parent and canonical tautomer). The model also
  compares explicit functional-group counts (acids, amines, amides,
  phenols, sulfonamides, phosphates, heteroaromatics and halogens), scaffold and
  ionization class. When ChEMBL supplies calculated acidic/basic pKa values they
  participate in the ionization similarity. Dimorphite-DL enumerates plausible
  net-charge states at pH 2, 3, 5 and 7; these are possible states rather than
  quantitative species fractions. RDKit itself does not predict pKa,
  so missing pKa is never replaced with a made-up numeric estimate. The model does not fill missing
  parameters from other molecules because that creates non-physical "chimera"
  methods. It reports missing fields, confidence components and a screening
  range around the transferred method. The class in `scripts/model.py` is
  model-shaped (`fit`/`predict`/`save`/`load`) so a learned estimator can replace
  it once more labelled data exists.

### To improve accuracy

1. **Fill `data/molecule_map.csv`.** Each row maps a source document to its INN
   and **SMILES**. Only rows with SMILES enter the model. Add the missing SMILES,
   verify the pre-filled ones, and flip `verified` to `true`. More molecules =
   better predictions. Rows marked `EXCLUDE` (combinations, allergens,
   homeopathic) are not single small molecules and should stay unmapped.
2. Re-run `build_dataset.py` and `train_model.py`; check the leave-one-out report.

> The prediction is a **method starting point** by analogy, not a validated
> analytical procedure.

## 7. Prepare a QSRR retention dataset

QSRR is trained on `molecule + HPLC conditions -> log(k)`, not directly on
`SMILES -> method`. Build auditable retention candidates with:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/build_qsrr_dataset.py
```

This creates `qsrr_observations.csv` with extraction candidates, an empty
human-owned `data/qsrr_curated.csv` template (never overwritten), a strict
`qsrr_training.csv`, a `qsrr_provisional.csv` with geometry-estimated `t0`, and
`qsrr_report.json`. Copy verified candidate rows into
the curated table and set `review_status=approved` only after checking source
evidence. A row is training-ready only when exact
`tR`, measured `t0`, column, modifier, organic percentage and flow are present
(plus pH for a buffer). The target is calculated as
`log10((tR - t0) / t0)`; run time is never substituted for `t0`.
When measured `t0` is absent, the provisional table estimates it from column
length, internal diameter, flow and `--porosity` (default `0.68`). Such rows are
marked `t0_source=estimated_column_geometry` and are excluded from strict training.

### Provisional direct retention-time model (without t0)

For early pipeline testing, build a deduplicated dataset and train CatBoost
directly on `tR`:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/build_tr_dataset.py
PYTHONPATH=scripts .venv/bin/python scripts/train_tr_model.py
```

This model accepts missing condition fields and downweights approximate or
incomplete observations. It is kept separate from strict QSRR because `tR`
depends on column geometry, flow and the instrument. Evaluate it with grouped
cross-validation by molecule; do not treat training-set fit as evidence of
predictive accuracy.

## Target-specific source audit (TFV/FTC)

Reindex already downloaded, SHA256-verified licensed XML without network access,
including JATS tables stored outside the article body, and build a target review queue:

```bash
.venv/bin/python scripts/collect_open_methods.py --reindex-cached
.venv/bin/python scripts/curate_target_evidence.py
```

Outputs are in `reports/target_data_curation/`: source-linked retention candidates,
same-method (not proven same-injection) pair candidates, quarantined experimental
rows, conflicting author claims, a CSV review queue and a Russian summary.
The current audit finds 21 RepoRT target observations in 7 systems containing both
analytes, and 13 FTC/lamivudine experimental rows (not TFV/FTC). PMC12114340 has
contradictory narrative/table retention times and elution order; PMC3929521 has
contradictory method pH and injection volume. Neither is silently approved for
training. No measured target-pair Rs is established by this audit. Development
examples are not an untouched final validation set; existing models are unchanged.

## Multi-column TFV/FTC research comparison and customer report

```bash
.venv/bin/python scripts/evaluate_realism.py --feature-version chemistry_ph_hydrodynamics_v2 --extra-training-rows 18000 --iterations 500 --out reports/expanded_evaluation
.venv/bin/python scripts/evaluate_realism.py --feature-version chemistry_ph_hydrodynamics_v2 --extra-training-rows 18000 --domain-weighting --iterations 700 --out reports/domain_weighted_evaluation
.venv/bin/python scripts/curate_target_evidence.py
.venv/bin/python scripts/compare_columns.py
```

The additional 18,000 training rows use system-balanced sampling, excluding frozen
holdout molecules and target/prodrug families. Original selection/calibration/test
IDs are unchanged. The source-transfer experiment also excludes held-out sources
from added training rows. Domain weights use target structures, never target RTs.
Model choice uses only the original selection macro-MAE, not target/test errors.

Outputs: `reports/customer_multicolumn/Отчёт_для_заказчика.docx`, `REPORT.md`,
`comparison.json`, `candidates.csv`, and source gradient CSVs. Current run: 84
programs on 10 column products. Methods with conflicting column labels (CORTECS
versus Raptor Biphenyl), invalid solvent balances or missing critical metadata
are rejected. Superseded generated gradients are preserved in `superseded_gradients/`.
Except for the checked Acclaim geometry, manufacturer limits remain unverified.

Important negative result: broader test MAE improved from 1.485 to 1.364 minutes,
but the historical 21-row connectivity-based TFV/FTC diagnostic worsened from
1.485 to 2.479 minutes. Full InChIKey review identifies 17 exact identities;
after removing METLIN alias 0209, the selected model's diagnostic MAE is 2.185
minutes on 16 remaining records. It gets the order wrong in all three unambiguous
exact-identity pair comparisons. These are development diagnostics, not final
validation. The expanded model is NOT promoted to production and all separation
rankings remain unresolved. No peak-width/Rs/pressure model or automated execution
is claimed. This run compares column+method packages, not isolated column effects.

## First laboratory experiment (evidence-first, not an ML optimum)

`compare_columns.py` now also produces `lab_recommendation.json`, `LAB_PLAN.md`,
`Задание_в_лабораторию.docx`, and primary/backup gradient CSVs. Rebuild just the
laboratory packet from an existing comparison without training or network access:

```bash
.venv/bin/python scripts/recommend_lab.py
```

The policy selects an unmodified traceable source program with complete fields,
preferring checked column geometry, published provenance, the existing low-pH
Acclaim reference, and coverage. Predicted tR gap is deliberately NOT used.
Backups use different column products. A primary must include a return to initial
composition and an explicit equilibration hold; missing programs/conditions or
source links cause rejection, never silent imputation. A backup without an explicit
equilibration hold is marked blocked until an analyst defines/verifies it.
If no primary qualifies, output is `BLOCKED_NO_COMPLETE_SOURCE_METHOD`.

The packet separates source conditions from proposed injection/concentration/UV
defaults. It requires analyst sign-off on pressure limits, dwell volume, phase pH,
equilibration, sample solubility and detection before testing. All recommendations
retain `instrument_execution_authorized=false` and `separation_confirmed=false`.
This is a practical first-experiment proposal, not a validated separation method.

## Experiment → local correction loop (TFV/FTC)

```bash
# Once: create a laboratory-owned DRAFT, six adaptation conditions + two reserved validation conditions.
.venv/bin/python scripts/experimental_loop.py init
# After an analyst fills and approves protocol.json: freeze predictions BEFORE running experiments.
.venv/bin/python scripts/experimental_loop.py predict
# Validate and append ONE reviewed real injection (does not rewrite existing measurements).
.venv/bin/python scripts/experimental_loop.py record --observation path/to/reviewed_injection.json
# After adding real reviewed observations: audit, estimate local corrections, propose next experiment.
.venv/bin/python scripts/experimental_loop.py update
```

Protocol, recipe/observation templates, raw observation log and frozen predictions
live in `data/lab/tfv_ftc/` (git-ignored). `init` refuses to overwrite an existing
protocol. Current status/guide: `reports/lab_cycle/{state.json,CYCLE.md}`.
No experimental values are generated by these commands. Until actual instrument,
measured dwell volume, fixed sample parameters, recipes and approval are supplied,
the loop returns `WAITING_FOR_APPROVED_PROTOCOL` and does not fit a model.
Frozen predictions are never overwritten: repeated `predict` with the same model
and protocol reuses them; a changed protocol/model requires a new campaign folder.
The loop refuses to establish a predictive baseline after measurements have
already been recorded. `record` checks the complete log and the new injection
before append, rejecting duplicate run/raw-injection IDs without changing it.

The proposed pH 2.6/3.6 levels are draft suggestions for an analyst, not executable
recipes. Source methods 0385/0386 differ in buffer and initial solvent composition
and must not be used as a pH-only contrast. The plan fixes one column, flow and
temperature and changes only analytical ramp duration (0.8/1.0/1.2), retaining wash,
return and equilibration durations. Two intermediate conditions are reserved for
validation; alternative names or repeat injections cannot cross the split.
The base model uses molecular/column/phase/gradient conditions; measured dwell
volume is recorded and exact-context gated, NOT a newly learned dwell correction.

Each accepted real injection needs a reviewed record, exact target InChIKeys,
actual method/instrument snapshot matching frozen predictions, pressure, raw-file
SHA256 and injection identifier. Blank/standard injections are retained but do
not substitute for a mixture run. Coelution/non-detection are stored without fake
tR labels. Rs requires an actual reported measurement with method recorded or
two identified same-injection peaks with half-height widths. Source t0 is never
copied into the laboratory measured-t0 field.

With at least three DIFFERENT adaptation conditions per recipe/analyte, the loop
evaluates a median residual correction to the frozen base tR. Replicates are
averaged per condition. The correction is retained only if leave-condition-out
MAE improves on the uncorrected prediction, and applies only inside that recipe's
sampled gradient range on the same instrument/column serial/matrix/sample setup.
Reserved validation remains excluded from fitting/selection. Training residuals
and two validation conditions do not yield a claimed calibrated 90% interval.

The scheduler first fills the grid, then requests repeats and reserved validation.
Best-observed conditions use measured Rs, pressure and analyst-defined retention,
tailing/SNR/carryover criteria; unknown criteria are explicitly listed. A pressure
violation or retention at/before measured t0 can require review before further
runs. This is not a trained Bayesian optimizer, universal QSRR, or certified
method. Future Bayesian optimization needs sufficient measured outcomes and
validated constraints. Global CatBoost models and earlier metrics are unchanged.

## Notes

The extractor searches for Russian and English terms around dissolution and
chromatographic preparation, including:

- раствор, растворение, растворимость, растворитель
- solubility, dissolution, dissolve, solvent, diluent
- mobile phase, sample solution, standard solution, stock solution

If the source document has a very specific heading structure, tune
`DISSOLUTION_TERMS` and `HEADING_RE` in `scripts/prepare_rag.py`.

## Scanned PDFs

If a PDF is a scan, normal PDF text extraction can return almost no text.
The script then automatically falls back to OCR.

For Russian scanned documents, install Tesseract and Russian language data:

```bash
brew install tesseract tesseract-lang
```

Then run:

```bash
.venv/bin/python scripts/prepare_rag.py --ocr-engine tesseract --ocr-dpi 180
```

Higher `--ocr-dpi` values can improve recognition, but make processing slower.

## Physical retention calibration (research, isocratic only)

The selected TFV/FTC gradient is preserved. This independent module does not
modify candidate ranking, simulate that gradient or claim improved accuracy.

```bash
.venv/bin/python scripts/audit_physical_retention.py
.venv/bin/python scripts/physical_retention.py --measurements reviewed_series.json --out reports/physical_retention/calibration.json
.venv/bin/python -m unittest discover -s tests -p 'test_physical_retention.py'
```

The audit writes `reports/physical_retention/REPORT.md` and `audit.json` with
dataset hashes. A positive `column.t0` is not proof of measured void time;
a one-point gradient table is not evidence of isocratic elution. The current
normalized corpus has no constant-composition candidate passing this audit.
This is not a claim that no suitable measurements exist in the literature.

Calibration input is a JSON array. Each row requires `observation_id`,
`full_inchikey`, `phi` (organic volume fraction, 0–1), `t0_min`, `tr_min`,
`source`, `t0_evidence`, `t0_origin: "measured"`, `label_origin: "experiment"`,
`mode: "isocratic"`, `reviewed: true`, and a preassigned
`split: "calibration"` or `"validation"`. The reviewer must verify these claims
against the source; a boolean is not independent proof. Every row must have the
same `context` object with `column` (exact product and geometry), `modifier`,
`aqueous_recipe` (buffer/additives/concentrations), numeric `pH`,
`pH_convention`, numeric `temperature_c`, numeric `flow_ml_min`, and
`instrument_id`. Use one exact molecular identity per fit.

The model fits `log10(k) = log10(kw) - S * phi`, where
`k = (tr-t0)/t0`. At least three distinct calibration compositions are required.
Replicates receive equal weight per composition; validation compositions cannot
overlap calibration compositions. Validation labels never enter the fit.
Out-of-domain validation compositions are reported without extrapolated times.
No validation observations means no accuracy estimate. Reported validation MAE
is conditional on the supplied measured t0; it does not validate prediction of t0.

This is a calibration foundation, not a gradient simulator or a peak model.
No synthetic experimental dataset, widths, Rs, or uncertainty intervals are
created. Existing synthetic unit tests verify mathematics and input guards only.
For the chosen gradient, further work requires a solvent-propagation model,
dwell-volume evidence and independently calibrated retention parameters.

### Gradient simulation research module

`scripts/gradient_retention.py` now implements the ideal equilibrium LSS migration
integral with gradient delay and piecewise-linear solvent programs. It can fit
retention parameters from reviewed gradient measurements and evaluate held-out
programs, without requiring an isocratic calibration series. Timing evidence and
at least three informative calibration programs are required. There is no real
TFV/FTC calibration yet and no claim of improved accuracy or simulated peak shape.

See [simulation workflow and assumptions](reports/physical_retention/SIMULATION_WORKFLOW.md).
The selected source gradient is preserved in the unfilled campaign template.
Synthetic output is labelled explicitly and rejected by experimental data loaders.
No synthetic augmentation or retraining is enabled automatically.

```bash
.venv/bin/python scripts/gradient_retention.py template \
  --input reports/customer_multicolumn/lab_recommendation.json \
  --out reports/physical_retention/gradient_campaign_template.json
.venv/bin/python -m unittest discover -s tests -p 'test_gradient_retention.py'
```

The template command refuses to overwrite an existing file.

### Learning simulator parameters from open measured retention factors

New source: [Horner et al., Retention data, version 3](https://data.mendeley.com/datasets/x292z4gd25/3),
CC BY 4.0, DOI `10.17632/x292z4gd25.3`. Downloaded source files and API metadata
are in `data/external/horner_retention`. The importer verifies all three published
SHA-256 checksums before processing; it does not execute source spreadsheet code.

```bash
.venv/bin/python scripts/build_lss_training.py
.venv/bin/python scripts/train_lss_parameters.py
```

The source provides 26,420 measured-k records for 105 compounds on Waters BEH C18
with ACN/aqueous phosphoric acid. CSV column 2 is mass fraction; column 3 is volume
fraction (already 0–1, do not divide again by 100). Temperature series remain
separate. Author-reported k is used directly without inventing t0, tR or pH.
The provenance/method of the authors' k determination still needs source review.

547 series support a preliminary LSS fit. Entire compositions and their replicates
are held out for interpolation checks. Log(kw) extrapolates to water and is not
a directly measured water-retention label. LSS can fail outside the observed range.

The first structure model uses 261 series / 49 molecular connectivity groups from
unambiguous exact-name matches in the existing corpus. RDKit checks that SMILES
reproduces the matched full InChIKey. This is still an automatic identity proposal,
not a complete audit of source substance identity; ambiguous matches are excluded.
Names/typos are not silently corrected. Columns cannot be learned as a feature
from this one-column dataset, so this experiment uses structure and temperature.

Five-fold validation holds every temperature of a molecule together. The fitted
parameters of held-out molecules never enter model training. End-to-end predictions
are compared with actual log10(k) on held-out compositions of held-out molecules.
The reference uses median training-compound parameters at the same temperature.
No hyperparameter search or production promotion is performed.

Results and research model: `reports/lss_parameter_learning/`. Current MAE log10(k)
is 0.249 versus 0.529 for the reference; equal-molecule macro-MAE is 0.276.
These are log-retention-factor units, not minutes, and no calibrated uncertainty
interval follows from this score. This is a provisional research CV, not final
validation. It does not establish transfer to TFV/FTC, Acclaim C18 or methanol.
The selected source gradient and production CatBoost model remain unchanged.

### PubChem cross-check and expanded LSS research comparison

```bash
.venv/bin/python scripts/audit_lss_identities.py prepare
curl --rate 2/s --config data/external/horner_retention/pubchem/requests.curl
.venv/bin/python scripts/audit_lss_identities.py build
.venv/bin/python scripts/train_lss_parameters.py \
  --data data/processed/lss_identity_audited/parameter_labels.jsonl \
  --out reports/lss_parameter_learning_pubchem \
  --split-reference data/processed/lss_research/parameter_labels.jsonl \
  --baseline-predictions reports/lss_parameter_learning/predictions.jsonl
```

PubChem responds to 96/105 exact source names with one internally consistent
structure; nine names return NotFound and remain excluded. This is an external
name cross-check, not proof of the identity of the original experimental material.
No silent spelling corrections, salt removal, or stereochemical substitutions.
Every response URL and checksum is retained in `reports/lss_identity_audit/audit.json`.

Expanded fitting uses 500 series, 94 molecular connectivity groups. Original
molecule test-fold assignments are frozen; new groups are assigned by deterministic
hash. The same-case comparison uses the same 794 molecule/temperature/composition
cases and folds: MAE log10(k) changes from 0.24935 to 0.21325 (about 14.5% lower).
On all 1640 expanded cases MAE is 0.18265; this is a different evaluation population
and must not be presented as a direct improvement over 0.24935.

Reports: `reports/lss_parameter_learning_pubchem/`. Original labels, model and
evaluation remain in their original directories. All results remain research-only,
limited to the source BEH C18 / ACN system, without TFV/FTC or cross-column validation.

### Direct log(k) comparison and transfer-series audit

```bash
.venv/bin/python scripts/audit_transfer_series.py
.venv/bin/python scripts/evaluate_direct_logk.py
```

The transfer audit groups source-publication/column/recipe contexts and counts
distinct full gradient programs per exact molecular identity. It excludes RP/HILIC
mixing and does not count duplicate source systems as additional gradient programs.
Groups are review candidates only: missing temperatures, instrument equivalence,
void-time provenance and dwell/post-column delays remain blockers. Different wash
segments do not necessarily imply informative independent calibration programs.
Results: `reports/transfer_series_audit/`.

### MCMRT: multi-column/methanol transfer benchmark

Author workbook: `data/external/mcmrt/Supplementary Data 1-14.xlsx`, supplementary
data of DOI `10.1038/s42004-024-01135-0`, CC BY 4.0. Git blob and SHA-256 verified;
license evidence and provenance are stored alongside it. Separate ScienceDB DOI
`10.57760/sciencedb.15823` has CC0 metadata, but identical file contents are not assumed.

```bash
.venv/bin/python scripts/import_mcmrt.py
.venv/bin/python scripts/evaluate_mcmrt_transfer.py
```

Importer reads only structures/methods/experimental times on sheets 1–3. It checks
SMILES against source InChI, parses geometry in either orientation, converts phases
and gradients explicitly, and preserves original cell references. Ten structure
disagreements and five formula-derived tR cells are excluded. 217 n.d. outcomes
are stored without numerical labels. CM19 is quarantined because its phase-A
description disagrees with the recipe code (0.1% vs 0.01% formic acid).

9,768 numeric observations / 333 structures / 30 methods / six columns imported;
9,443 rows in 29 methods enter RT research. Twenty-four methods use methanol in B,
six acetonitrile. Initial A can contain 10% methanol, so %B is not assumed equal
to organic %. No pH, measured t0, pressure, width or Rs is invented. This supplement
does not contain TFV or FTC measurements.

The evaluation freezes data hashes, hyperparameters and train/test IDs before fitting.
Whole methods, whole column products, and joint column+molecular-connectivity groups
are held out in separate experiments. Only MCMRT is used for training, avoiding
unresolved cross-source duplicate leakage. The comparator uses a training-only
median normalized tR for known molecules, or a global median for unseen molecules.

Initial MAE (minutes): unseen method 2.727, unseen column with familiar molecules
2.824, joint unseen column+molecule 3.516. The first two aggregate scores are worse
than the simple comparator (2.368 / 2.483). This is a meaningful negative result;
no production promotion or claim of solved TFV/FTC transfer. Per-column and solvent
results, source-overlap candidates and full predictions: `reports/mcmrt_transfer/`.
Full-condition held-out models are not calibrated for a different laboratory.

### MCMRT error audit: analytical-time normalization

The first error audit identified severe time-scale failures: CM16 (100-minute
cycle) was underpredicted by about 29.7 minutes on average, while CM01 (10-minute
cycle) was overpredicted by about 8.8 minutes. This motivated one research change:
fit `log1p(tR / s)` instead of `log1p(tR)`, where `s` is the method's first time
at maximum actual organic fraction. Reconstruct minutes by multiplying the
inverse-transformed prediction by `s`. This is empirical target scaling, not k
or measured void time. Wash/return durations after the maximum do not set `s`.
The original feature vector and hyperparameters are unchanged.

```bash
.venv/bin/python scripts/evaluate_mcmrt_transfer.py \
  --target-mode analytical_scaled --out reports/mcmrt_transfer_scaled
.venv/bin/python scripts/audit_mcmrt_errors.py \
  --candidate reports/mcmrt_transfer_scaled
```

The paired audit checks identical dataset hashes, split IDs, observed labels,
features and hyperparameters. On the same development evaluation cases, MAE changes:
unseen methods 2.727 → 1.330 min; unseen columns with familiar molecules
2.824 → 1.662 min; joint unseen column+molecules 3.516 → 2.475 min.
Some individual columns/methods worsen; all breakdowns are retained, not only wins.
For joint transfer the 90th percentile absolute error remains 5.18 min: this is
an empirical error quantile, not a calibrated prediction interval.

These folds informed development and are no longer an untouched final validation.
No guarantee of arbitrary-gradient extrapolation, TFV/FTC accuracy or Rs follows.
Previous outputs and the selected laboratory gradient are preserved. Reports:
`reports/mcmrt_error_audit/` and `reports/mcmrt_transfer_scaled/`.

### External-source check of analytical-time scaling

```bash
.venv/bin/python scripts/evaluate_external_transfer.py
```

Both compared models train on MCMRT only, with unchanged 300-iteration parameters.
Predeclared external RepoRT methods: Brungs/MSnLib `0390`, `0391` (ACN) and
Han `0400`–`0403` (MeOH). `0401` is rejected for inconsistent flow metadata;
`0402`/`0403` have an ammonium-hydroxide additive absent from the feature schema
and are excluded rather than represented as additive-free phases.

The adapter validates explicit solvent percentages and additive units. Source
column labels are preserved. Duplicate-time nodes encode a declared instantaneous
return step, preserved without inventing elapsed time. One repeated external
structure/method/time candidate is removed; potential structure+rounded-RT overlaps
with MCMRT are conservatively excluded and recorded (none in this run).

External results: 17,498 records, MAE 4.325 → 1.285 min. Most molecular connectivities
(17,330 records) are absent from MCMRT training. Separate methanol method: 480 records,
MAE 2.112 → 0.831 min. Overall signed bias is +1.136 min and empirical 90th percentile
absolute error is 2.489 min; neither gives a calibrated individual prediction interval.
No correction is fitted using these external answers.

This is external to the two MCMRT-only models, not a globally untouched final test
of the project: RepoRT was used in earlier research. One exact FTC observation is
present (measured 0.6455 min, scaled prediction 2.2430 min); no exact TFV record.
That FTC measurement uses a different BEH C18/ACN method and does not validate
the selected TFV/FTC gradient. No production promotion. Reproducible frozen plan,
source exclusions, full predictions and research model files: `reports/external_transfer/`.

### Early-elution diagnostic (not a measured void-time classification)

```bash
.venv/bin/python scripts/evaluate_mcmrt_transfer.py \
  --target-mode analytical_log --out reports/mcmrt_transfer_log
.venv/bin/python scripts/audit_early_exit.py
.venv/bin/python scripts/evaluate_early_external.py
```

One fixed candidate changes only the training target from `log1p(tR/s)` to
`log(tR/s)`, where s is the same analytical time scale. It emphasizes relative
errors of short retention times without inventing t0 or using external FTC labels.
Early is defined diagnostically as observed tR/s ≤0.10; this is not proof of weak
retention or a classification used to select the prediction model at inference.

Before fitting, the research gate required improved early-case MAE for jointly
unseen columns/molecules with no more than 5% overall MAE degradation in that
scenario. It passes narrowly: early MAE 2.720 → 1.808 min (61 cases), overall
2.475 → 2.596 min (about 4.9% worse). New-method overall MAE also worsens
1.330 → 1.438 min even though early cases improve 1.205 → 0.656 min.
This is a tradeoff, not a universal improvement or production approval.

The external comparison is run only after this MCMRT criterion. It reuses the
previously inspected external cases as a diagnostic, not a new final holdout.
Original data hashes, training/test IDs and hyperparameters are checked; no
external retention labels enter fitting or tuning. Results are kept separately
in `reports/early_exit_audit/` and `reports/early_exit_external/`.

External diagnostic: overall MAE 1.285 → 1.206 min; early cases (244 rows)
1.863 → 1.583 min. The methanol source worsens 0.831 → 1.201 min, so the
candidate is not promoted for the selected methanol task. FTC prediction changes
2.243 → 2.126 min against measured 0.6455 min: improvement is small and its
remaining error is about 1.48 min. No correction is fitted to this single point.

The direct-log(k) experiment trains CatBoost on molecular features, temperature,
and organic volume fraction, using author-reported experimental k. It uses all
compositions of training molecules and excludes all temperatures/compositions
of test molecules. Evaluation is on exactly the same 1640 cases/folds as the
expanded LSS model; the script asserts comparator case equality. Because both
the label representation and available training compositions change, this is
a comparison of complete approaches, not an isolated architecture experiment.

Direct MAE log10(k): 0.16085; LSS: 0.18265. However ketoprofen worsens from 0.145
to 0.204. No automatic winner promotion, selective per-molecule oracle, or new
claim about TFV/FTC is made. The direct model has no monotonicity constraint,
is not a gradient simulator, and has no calibrated uncertainty. Full predictions,
model and limitations: `reports/direct_logk_comparison/`.

### Physical-feature transfer experiment (research only)

`early_retention_features.py` adds geometric column volume/flow, superficial
velocity, initial solvent/additives and molecular polarity interactions. Volume/flow
is not measured t0; formal charge is not a pH-dependent ionization profile.
The analytical-scaled target, MCMRT observations, folds and CatBoost parameters
are unchanged from `reports/mcmrt_transfer_scaled`.

```bash
.venv/bin/python scripts/evaluate_mcmrt_transfer.py --target-mode analytical_scaled --feature-version physical_early_v1 --out reports/mcmrt_transfer_physical
.venv/bin/python scripts/audit_physical_features.py
```

The predeclared gate failed: MAE for unseen methods increased from 1.330 to
1.557 min; unseen columns from 1.662 to 1.726; jointly unseen columns/molecules
from 2.475 to 2.522. Early joint cases improved slightly (2.720 to 2.595 min,
61 observations), insufficient to offset the overall regression. Early means
observed tR within 10% of the analytical time scale, not proven weak retention.
No external candidate evaluation or production promotion follows this failed
gate. MCMRT contains neither TFV nor FTC; their chosen gradient is unchanged.
These repeatedly studied folds are development diagnostics, not a new blind test.
Full paired results: `reports/physical_features_audit/REPORT.md`.

### Local calibration on public measurements (research only)

Run `.venv/bin/python scripts/evaluate_local_calibration.py` to reproduce
`reports/local_calibration/REPORT.md`. This uses the frozen MCMRT-only scaled
model predictions, without refitting CatBoost. Select 20 molecular connectivities
per complete method by a fixed structure hash, excluding TFV/FTC; fit the median
of per-molecule median residuals. Identical full methods share a calibration;
column identity alone is insufficient. Exclude every calibration connectivity
from testing across all sources. The plan is frozen before fitting offsets.

On 17,444 remaining observations MAE decreased from 1.286 to 0.846 min. Sources
0390/0391 improved; methanol method 0400 worsened from 0.836 to 0.856 min.
FTC (not a calibrant) changed from 2.243 to 1.469 min versus measured 0.6455 min;
TFV has no test observation. Four negative corrected outputs were clipped to zero
and counted as boundary failures. These are retrospective results on previously
inspected public data, not independent final validation, calibrated uncertainty,
or proof of performance on the customer instrument. Panel sensitivity remains
untested. Production models and the selected TFV/FTC gradient are unchanged.

### Response to changes in gradient programs

`.venv/bin/python scripts/evaluate_gradient_changes.py` audits frozen scaled-model
predictions on MCMRT. Eleven pairs match recorded metadata, instrument and
constant flow but differ in full gradient programs. Both predictions in a pair
come from the same held-out-column fold; the joint scenario also excludes test
molecular connectivities globally. No retraining or customer-gradient changes.

For 695 paired observations in the joint scenario, MAE of the predicted change
is 3.168 min versus 9.258 for analytical-time scaling anchored to one measured
retention time. Direction accuracy (97.1%) does not beat that simple baseline
(97.3%). In the diagnostic subset with measured changes of 0.1–2 min (84 pairs),
model error is 1.384 min, worse than assuming no change (0.901 min). This subset
is selected by observed response, not by small gradient edits. Thus precision
for fine method optimization is not demonstrated. Pairs are statistically
dependent and reuse development folds; unreported pH/dwell differences remain
possible. TFV/FTC and Rs are not evaluated. Full breakdown and limitations:
`reports/gradient_change_audit/REPORT.md`.

### Generate original R&D gradient hypotheses for TFV and FTC

Run `.venv/bin/python scripts/design_rnd_gradients.py`. Outputs are in
`reports/rnd_gradient_design/`: all 162 predictions, a frozen design plan,
a human-readable report, and gradient CSVs for the source control and three
exploratory candidates. This changes initial %B, hold, intermediate composition,
knee time and time to 95% B independently; it is not uniform time scaling.
Column, phase recipes, flow and temperature stay fixed. Existing column checks
apply. Source wash/return/equilibration durations are preserved as assumptions,
not proof of adequate cleaning or equilibration at a different composition.

Selection uses explicit heuristic roles (gap/time, shorter program, larger gap)
and parameter diversity. Predicted gap is not Rs, uncertainty is unvalidated,
and tiny score differences do not establish a better method. Two candidates
have 25-minute cycles instead of 30; a third has virtually the same predicted
peak gap as the source control. These are experiments to compare, not a new
validated recommendation. Negative/nonfinite predictions or peaks after the
analytical ramp are excluded from ranking; measured t0 and pressure are absent.
Novelty means no exact numeric match in parseable stored RepoRT programs only.
Predictions are marked as model-derived and not approved for training. No model
retraining, production method replacement, or instrument execution occurs.

R&D confidence diagnostics are generated automatically by `rnd_confidence.py`
when running `design_rnd_gradients.py`. Each generated candidate now has a
qualitative evidence level, explicit reasons, exact-label column and phase
recipe coverage in eligible MCMRT training metadata, and deterministic
sensitivity diagnostics. Eight scenarios vary middle %B by ±2 percentage points,
hold by ±0.1 min, and knee/end times by ±5%, one factor at a time. All generated
programs are compared under each common perturbation using gap/analytical-time.
Rank ranges are numerical sensitivity, not probabilities or statistical intervals.
Local empirical validation is absent, so no stable result can receive high
confidence. Probability and calibrated interval fields remain null.

For the current three candidates the rank ranges are 1–32, 5–14, and 1–31 among
162 programs. All are low-confidence research hypotheses: there is no exact
column-label or phase-recipe match in the 29 eligible MCMRT methods, and no
local TFV/FTC validation. Exact-label absence does not establish complete
chemical dissimilarity. The deterministic time ranges must not be presented
as prediction intervals. Selection and production conditions are unchanged.

### Local gradient adaptation with a held-out program

Run `.venv/bin/python scripts/evaluate_local_gradient_model.py`. Two matching
column/phase/instrument/constant-flow series contain 4 and 3 programs. Each
molecule is locally calibrated on 2–3 programs, then evaluated on the omitted
program. The frozen base CatBoost excluded the entire column and the molecule's
connectivity. Local adaptation intentionally uses measurements of that same
molecule: this is unseen-program testing, not unseen-molecule testing after
calibration. Source folds and local fit/test IDs are preserved in the plan.

Across 440 predictions for 126 molecules, MAE is 3.640 min for CatBoost, 2.689
for a per-molecule median residual correction, and 2.874 for a regularized
residual slope versus log analytical duration (fixed slope penalty 0.1). Thus
the more complex correction does not beat the simple calibration baseline.
Twenty-four local predictions were negative and clipped to zero, a reported
failure mode. Full per-program results include regressions. These observations
are dependent, and previously studied development data are not a new blind
validation. A single analytical-duration feature does not represent all gradient
shape changes. There are no TFV/FTC calibration measurements here. Production
models, confidence levels and the customer method are unchanged.
Results: `reports/local_gradient_model/REPORT.md`.

### Gradient shape adapter and feedback snapshot

Run `.venv/bin/python scripts/evaluate_gradient_shape.py` for the fixed comparison
on the same 440 local-calibration cases. The ridge residual adapter uses actual
organic fraction on absolute and normalized time grids, truncated at the first
maximum, plus log analytical duration. Fixed scaling and alpha=0.1 are declared
before fitting; no test labels set preprocessing or hyperparameters.

MAE: CatBoost 3.640, constant correction 2.689, duration correction 2.874, shape
correction 2.669 min. The 0.020-min advantage over constant correction is small
and not demonstrated robust. CM01 worsens from 2.442 to 4.646 min; 18 shape
predictions were negative and clipped. This does not justify promotion or
application to unmeasured TFV/FTC. Shape features approximate the program and
are not a mechanistic simulator. Full results: `reports/local_gradient_shape/`.

`reports/customer_feedback_snapshot/` contains the customer discussion document,
four experimental program CSVs, evidence JSON snapshots and hashes. This fixes
the current research state for feedback; it does not change production models,
the original control, or the low confidence of generated TFV/FTC hypotheses.

### Customer feedback experiment plan — 9 November 2026

Run `.venv/bin/python scripts/build_feedback_experiment_plan.py` to produce
`reports/lab_plan_revision_2026_11_09/experiment_plan.json` and the measurement
template. The declared working assumption is pure TFV/FTC standards and UV/DAD;
the customer's actual detector and matrix are still unconfirmed. Four phase
systems are compared on the same control gradient before testing generated
programs: source acid control, 10 mM added ammonium formate at aqueous pH 3/4,
and 10 mM added ammonium acetate at pH 5. Organic B is neat methanol for buffer
candidates. Salt concentration therefore decreases along the gradient; neither
constant ionic strength nor constant mixed-solvent pH is claimed. New phase
candidates have null model scores/tR/Rs; current pH and peak-shape prediction
is not validated. No production model is retrained or promoted.

The DOCX and REPORT.md specify condition-specific diluents, 100 ug/mL stocks
conditional on verified dissolution, 5/10/20 ug/mL working mixtures, 1 uL input,
proposed R&D criteria (not compendial validation limits), identification controls,
replicates and carryover checks. Initial phase screening is 28 injections/14
hours of programmed cycles, excluding equilibration. Aqueous pH convention,
added-salt vs total-buffer concentration, salt solubility checks and MS changes
are explicitly documented. Unknown/unmeasurable results are not passes.
