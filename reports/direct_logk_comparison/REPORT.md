# Прямой прогноз log(k) и прогноз через параметры LSS

На 1640 одинаковых проверочных случаях: прямая модель MAE 0.161; LSS-модель 0.183.
Все температуры и составы тестируемой молекулы исключены из обучения. Использованы все составы обучающих молекул, поэтому сравнение отражает весь новый способ обучения, а не только архитектуру.

| Вещество | Прямая модель | LSS |
|---|---:|---:|
| caffeine | 0.368 | 0.442 |
| ketoprofen | 0.204 | 0.145 |
| ibuprofen | 0.342 | 0.414 |
| methylparaben | 0.169 | 0.236 |

Это исследовательский результат для BEH C18/ACN. Автоматической замены рабочей модели нет. Выбранный градиент TFV/FTC сохранён.

One fixed BEH C18/ACN system; no methanol or new-column validation.
Research folds already inspected; not a new final holdout.
Direct model does not impose monotonicity, and is not a calibrated gradient simulator.
Provisional PubChem name resolution; source substance identity remains unverified.
