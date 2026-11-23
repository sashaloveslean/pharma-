# Проверка переноса на MCMRT

Обучение и проверка только внутри нового набора, без смешивания с RepoRT. План и параметры зафиксированы до запуска.

| Проверка | Строк проверки | MAE, мин | Простой ориентир, мин |
|---|---:|---:|---:|
| unseen_column_and_molecules | 1871 | 2.596 | 5.672 |
| unseen_column_known_molecules | 9443 | 1.728 | 2.483 |
| unseen_method | 9443 | 1.438 | 2.368 |

Сценарии: новая программа; новая колонка с преимущественно знакомыми молекулами; одновременно новая колонка и новые молекулы.
Режим целевой переменной: analytical_log. Масштабирование, если включено, не является расчётом k или t₀.
В последнем сценарии все измерения тестируемых молекул удалены из обучения на всех колонках. В первом сценарии одинаковые полные условия группируются вместе.
Все значения в минутах. У методов разные длительности; общий MAE следует читать вместе с результатами отдельных колонок и растворителей.
TFV/FTC в данных нет: это проверка переноса подхода, не подтверждение выбранного метода заказчика.

Single laboratory/publication; held-out column is not held-out laboratory.
No TFV or FTC observations in this dataset.
pH, measured void time and dwell volume absent in the downloaded supplementary methods.
No widths, pressure or resolution forecast. RT classification and author predictions excluded.
Only method CMs without metadata conflicts and molecules with agreeing SMILES/InChI were used.
Matching structure+rounded RT across sources is a duplicate candidate, not proof; no cross-source pooling.
