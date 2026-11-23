# Проверка признаков формы градиента

Те же 440 контрольных случаев; 2–3 локальные программы для настройки, отдельная программа для проверки.
Модель поправки учитывает реальную долю органики на абсолютной и нормированной временных сетках до первого максимума. Поздняя промывка исключена из признаков поправки.

| Программа | n | CatBoost | Постоянная поправка | По длительности | По форме |
|---|---:|---:|---:|---:|---:|
| Все | 440 | 3.640 | 2.689 | 2.874 | 2.669 |
| mcmrt:01 | 62 | 2.948 | 2.442 | 5.018 | 4.646 |
| mcmrt:02 | 64 | 1.925 | 1.443 | 1.256 | 1.257 |
| mcmrt:06 | 62 | 1.789 | 2.939 | 2.696 | 2.277 |
| mcmrt:08 | 64 | 1.845 | 0.727 | 0.725 | 0.725 |
| mcmrt:12 | 64 | 3.022 | 1.502 | 1.101 | 1.101 |
| mcmrt:13 | 62 | 4.213 | 2.327 | 3.278 | 2.766 |
| mcmrt:16 | 62 | 9.874 | 7.583 | 6.220 | 6.071 |

Все ошибки — MAE tR в минутах. Рабочая модель не заменена.
Only 2–3 local programs per molecule; high dimensional shape is heavily regularized.
Same 440 retrospective cases, no new blind validation or hyperparameter tuning.
Known measured molecule on new program; not prediction for unmeasured TFV/FTC.
Shape grids are approximations, not a mechanistic column simulator.
No peak width, Rs or calibrated uncertainty. Negative predictions counted and clipped.
