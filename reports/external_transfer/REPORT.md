# Внешняя проверка нормализации времени удерживания

Обе модели обучены только на MCMRT. Внешние источники и ограничения записаны до обучения; параметры моделей одинаковы.
Это проверка переноса между источниками, не полностью нетронутая финальная проверка проекта: RepoRT использовался в предыдущих исследованиях.

| Группа | Наблюдений | Без нормализации, мин | С нормализацией, мин |
|---|---:|---:|---:|
| Все | 17498 | 4.325 | 1.285 |
| Новые молекулы | 17330 | 4.328 | 1.287 |
| report:0390 | 7333 | 3.735 | 0.952 |
| report:0391 | 9685 | 4.881 | 1.560 |
| report:0400 | 480 | 2.112 | 0.831 |

Исключённые методы: {"report:0401": "metadata_program_flow_conflict", "report:0402": "unrepresented_additive_or_solvent", "report:0403": "unrepresented_additive_or_solvent"}
Средние ошибки относятся к этим источникам и не определяют неопределённость отдельного прогноза.
Прогнозы не обрезались до экспериментального окна, не калибровались по внешним ответам. Рабочая модель и выбранный градиент не заменены.
RepoRT has appeared in earlier project research; this is external to the current MCMRT models, not a global final holdout.
Fixed configurations; no tuning on external errors.
Source column labels preserved without post-hoc alias optimization.
Conservative structure+rounded RT exclusions can remove coincidental matches; their IDs are retained.
No independent widths/Rs or calibrated prediction intervals. No claim of TFV/FTC accuracy.

Проверка целевых веществ по полному InChIKey:
TFV: измерений в этой внешней проверке нет.
FTC, report:0391: измерено 0.6455 мин, прогноз 2.2430 мин. Это отдельный метод, не выбранный метод пары TFV/FTC.
