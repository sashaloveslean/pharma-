# Локальная калибровка по открытым измерениям

20 веществ на полный метод. Поправка — медиана ошибок; выбор веществ по хешу структуры, без использования времени удерживания.
Все калибровочные структуры исключены из проверки на всех методах. Базовая модель не переобучалась.

| Группа | n | MAE до, мин | MAE после, мин | Прогнозов на границе 0 |
|---|---:|---:|---:|---:|
| Все | 17444 | 1.286 | 0.846 | 4 |
| report:0390 | 7326 | 0.952 | 0.647 | 1 |
| report:0391 | 9660 | 1.561 | 0.996 | 3 |
| report:0400 | 458 | 0.836 | 0.856 | 0 |

Целевые вещества (отдельные условия источника, не метод заказчика):
XQSPYNMVSIKCOC-NTSWFWBYSA-N, report:0391: измерено 0.6455; до 2.2430; после 1.4687 мин.
TFV: измерений для проверки нет.

Retrospective adaptation on previously inspected public sources; not a fresh blind test.
Same complete method only. Does not calibrate arbitrary gradients, columns or the customer instrument.
No TFV/FTC labels enter calibration; target results are diagnostics, not selection criteria.
One fixed structure-based calibration panel; sensitivity to panel choice remains untested.
No measured t0, peak widths, Rs or validated prediction intervals.
Zero-clipped predictions are boundary failures, not evidence of correct retention.

Рабочая модель и выбранный градиент TFV/FTC не изменены.
