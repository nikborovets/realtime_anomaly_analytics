## Что стоит вынести в recording‑rules (при появлении доступа)

| PromQL | Причина |
|--------|---------|
| `histogram_quantile(0.9, sum(rate(common_event_delay_bucket[1m])) by (le))` | тяжёлый quantile, вызывается часто |
| `rate(event_counter_total[1m])` | можно кэшировать, уменьшив нагрузку на API |