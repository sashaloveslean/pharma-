# Ранний выход: проверка изменения целевой переменной

| Сценарий / часть | Было MAE, мин | Стало MAE, мин | Число |
|---|---:|---:|---:|
| unseen_column_and_molecules / all | 2.475 | 2.596 | 1871 |
| unseen_column_and_molecules / early | 2.720 | 1.808 | 61 |
| unseen_column_and_molecules / other | 2.467 | 2.623 | 1810 |
| unseen_column_known_molecules / all | 1.662 | 1.728 | 9443 |
| unseen_column_known_molecules / early | 0.939 | 0.590 | 332 |
| unseen_column_known_molecules / other | 1.689 | 1.770 | 9111 |
| unseen_method / all | 1.330 | 1.438 | 9443 |
| unseen_method / early | 1.205 | 0.656 | 332 |
| unseen_method / other | 1.334 | 1.467 | 9111 |

Предварительный критерий MCMRT пройден: True.
Критерий: улучшение ранних пиков для новых колонок и молекул при ухудшении общего MAE не более 5%. Он записан в плане до обучения.
Early = first 10% of analytical scale, not measured void time or proven weak retention.
MCMRT groups already informed development. Gate is a research screen, not independent validation.
No external FTC prediction is used to choose the change.
