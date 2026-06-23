# Data Directory

Place raw weather CSV files here. The notebook expects `london_weather.csv`.

## Source
Historical London weather data can be downloaded from:
- [Kaggle: London Weather Data](https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data)

## Expected Columns
| Column | Description |
|--------|-------------|
| date | Date in YYYYMMDD format |
| cloud_cover | Cloud cover (oktas) |
| sunshine | Sunshine hours |
| global_radiation | Global radiation (W/m²) |
| max_temp | Maximum temperature (°C) |
| mean_temp | **Target** — daily mean temperature (°C) |
| min_temp | Minimum temperature (°C) |
| precipitation | Precipitation (mm) |
| pressure | Pressure (hPa) |
| snow_depth | Snow depth (cm) |
