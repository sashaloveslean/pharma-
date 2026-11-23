# Локальная модель изменения градиента

Для каждого вещества и колонки использованы 2–3 измеренные программы. Отдельная программа исключена из настройки. Базовый CatBoost не обучался на этой колонке и этих молекулах.
После локальной настройки молекулы уже известны по измерениям: проверяется новая программа, а не новое вещество.
Поправка: линейная зависимость ошибки CatBoost от логарифма времени первого максимума органики; коэффициент регуляризации 0,1 зафиксирован до расчёта.

| Проверка | n | CatBoost MAE мин | Постоянная поправка | Локальная зависимость |
|---|---:|---:|---:|---:|
| Все | 440 | 3.640 | 2.689 | 2.874 |
| interpolation | 188 | 2.608 | 1.984 | 2.217 |
| extrapolation | 252 | 4.411 | 3.215 | 3.363 |
| mcmrt:01 | 62 | 2.948 | 2.442 | 5.018 |
| mcmrt:02 | 64 | 1.925 | 1.443 | 1.256 |
| mcmrt:06 | 62 | 1.789 | 2.939 | 2.696 |
| mcmrt:08 | 64 | 1.845 | 0.727 | 0.725 |
| mcmrt:12 | 64 | 3.022 | 1.502 | 1.101 |
| mcmrt:13 | 62 | 4.213 | 2.327 | 3.278 |
| mcmrt:16 | 62 | 9.874 | 7.583 | 6.220 |

Отрицательные прогнозы локальной модели обрезаны до нуля: 24. Это сбои переноса, а не достоверное отсутствие удерживания.
Вывод текущего опыта: локальная зависимость улучшает исходный CatBoost в среднем, но уступает постоянной поправке. В рабочий предиктор не включена.
Все варианты сравниваются на одних строках; улучшение оценивается относительно обеих базовых моделей.
Для TFV/FTC такой поправки пока нет: отсутствуют локальные измерения выбранного метода.
Retrospective development folds, not a new independent validation.
Each analyte must have measured RT at 2 or 3 calibration programs on this column.
No measurements from the held-out program enter fitting. Same molecules across calibration and test are intentional.
Program changes are compressed to one analytical time scale; not a general gradient simulator.
Interpolation refers only to time scale, not the full multivariate gradient shape.
Same molecules recur across folds; rows are not independent trials.
No TFV/FTC measurements, widths, Rs or calibrated uncertainty. Customer method unchanged.
