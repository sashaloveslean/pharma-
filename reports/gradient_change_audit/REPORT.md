# Проверка реакции модели на изменение градиента

Сравниваются реальные измерения одного вещества при совпадающих колонке, фазах, температуре, постоянном потоке и указанном приборе.
Для каждой пары оба прогноза получены одной моделью без обучения на этой колонке.

| Сценарий | Пар наблюдений | Ошибка ΔtR, мин | Без изменения, мин | Масштабирование по времени, мин | Верное направление* |
|---|---:|---:|---:|---:|---:|
| unseen_column_known_molecules | 3591 | 2.315 | 14.032 | 9.427 | 96.6% |
| unseen_column_and_molecules | 695 | 3.168 | 14.399 | 9.258 | 97.1% |

*Для измеренного изменения более 0,1 мин; это порог анализа, не установленная экспериментальная погрешность.
unseen_column_known_molecules, небольшие измеренные изменения: MAE модели 1.378 мин; предположение без изменения 0.874 мин; n=488.
unseen_column_and_molecules, небольшие измеренные изменения: MAE модели 1.384 мин; предположение без изменения 0.901 мин; n=84.
Вывод: для тонкой настройки метода преимущества пока не показано. Высокая общая доля верного направления не превосходит простое масштабирование.
Ошибка ΔtR оценивает изменение удерживания одного вещества между программами, а не расстояние между пиками TFV и FTC.
Масштабирование использует измеренный tR исходной программы. Пары и все ухудшения сохранены в evaluation.json.
Для небольших измеренных изменений (0,1–2 мин) результаты приведены отдельно в JSON; это диагностическая подгруппа, не оценка точности малых правок градиента.
unseen_column_known_molecules: направление по простому масштабированию верно в 96.9% случаев.
unseen_column_and_molecules: направление по простому масштабированию верно в 97.3% случаев.

Repeated compounds/pairs are dependent observations, not independent trials.
Both programs predicted by the same model with the entire column held out.
Known-molecule scenario can use the same molecules on other columns; joint scenario excludes their connectivity globally.
Time-scaling baseline uses one measured starting tR; it is an anchored baseline, not a zero-shot prediction.
Matching recorded conditions cannot exclude unreported pH/dwell differences.
TFV/FTC absent; changes to the customer gradient are not validated.
Entire programs differ: this does not isolate one gradient segment or establish Rs.
