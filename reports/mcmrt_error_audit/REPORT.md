# Разбор ошибок переноса MCMRT

Совпадение строк, измерений, разбиений, признаков и гиперпараметров проверено при парном сравнении.
Проверочные группы уже изучались при разработке: результат не заменяет новую внешнюю валидацию.

| Сценарий / группа | Было, мин | Стало, мин | Изменение, мин |
|---|---:|---:|---:|
| unseen_column_and_molecules / all all | 3.516 | 2.475 | -1.041 |
| unseen_column_and_molecules / cycle_min 21-60 | 2.758 | 2.304 | -0.454 |
| unseen_column_and_molecules / cycle_min <=20 | 1.913 | 1.912 | -0.000 |
| unseen_column_and_molecules / cycle_min >60 | 30.855 | 9.874 | -20.981 |
| unseen_column_and_molecules / system mcmrt:01 | 2.004 | 2.948 | +0.944 |
| unseen_column_and_molecules / system mcmrt:16 | 30.855 | 9.874 | -20.981 |
| unseen_column_known_molecules / all all | 2.824 | 1.662 | -1.161 |
| unseen_column_known_molecules / cycle_min 21-60 | 2.053 | 1.444 | -0.609 |
| unseen_column_known_molecules / cycle_min <=20 | 1.586 | 1.520 | -0.066 |
| unseen_column_known_molecules / cycle_min >60 | 27.267 | 7.326 | -19.941 |
| unseen_column_known_molecules / system mcmrt:01 | 2.963 | 2.676 | -0.286 |
| unseen_column_known_molecules / system mcmrt:16 | 27.267 | 7.326 | -19.941 |
| unseen_method / all all | 2.727 | 1.330 | -1.397 |
| unseen_method / cycle_min 21-60 | 1.427 | 1.180 | -0.247 |
| unseen_method / cycle_min <=20 | 2.999 | 1.516 | -1.483 |
| unseen_method / cycle_min >60 | 29.700 | 3.511 | -26.189 |
| unseen_method / system mcmrt:01 | 8.792 | 2.171 | -6.622 |
| unseen_method / system mcmrt:16 | 29.700 | 3.511 | -26.189 |

Положительное изменение означает ухудшение. Все методы сохранены в JSON, включая ухудшившиеся.
Переход к tR / времени первого максимума органики — эмпирическая нормализация, а не физическая модель градиента. Промывка и возврат после первого максимума не задают масштаб.
