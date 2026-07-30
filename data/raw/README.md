# data/raw/

Place the **original unmodified CSV file** here before running the pipeline.

Expected file
-------------
```
data/raw/dataset.csv     (or use input/dataset.csv — the pipeline reads from both)
```

The raw dataset should match this schema:

| Column | Type | Description |
|--------|------|-------------|
| measured_on | datetime | Timestamp at 1-minute intervals |
| ac_current__319 | float32 | AC current (A) |
| ac_power__315 | float32 | AC power — **target variable** (W) |
| ac_voltage__318 | float32 | AC voltage (V) |
| ambient_temp__320 | float32 | Ambient air temperature (°C) |
| das_battery_voltage__326 | float32 | Data logger battery voltage (V) |
| das_temp__325 | float32 | Data logger temperature (°C) |
| dc_pos_current__317 | float32 | DC positive current (A) |
| dc_pos_voltage__316 | float32 | DC positive voltage (V) |
| dc_power__314 | float32 | DC power from array (W) |
| inverter_temp__324 | float32 | Inverter heat-sink temperature (°C) |
| module_temp_1__321 | float32 | Module surface temperature sensor 1 (°C) |
| module_temp_2__322 | float32 | Module surface temperature sensor 2 (°C) |
| module_temp_3__323 | float32 | Module surface temperature sensor 3 (°C) |
| poa_irradiance__313 | float32 | Plane-of-array irradiance (W/m²) |
| power_factor__327 | float32 | Power factor (dimensionless) |
| system_id | int8 | Plant system identifier |

Notes
-----
* The pipeline automatically handles the CSV → Parquet conversion.
* After the first run, `data/processed/processed_data.parquet` serves as the cache.
* Original CSV is never modified.
