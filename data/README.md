# Diccionario de datos

## Identificación del conjunto

| Elemento | Detalle |
|---|---|
| Nombre del archivo | `bronze_meteorologicos_aranzazu_raw.csv` |
| Ubicación | `Proyecto/data/` |
| Descripción | Registros de variables meteorológicas y ambientales observadas en Aranzazu, Colombia. |
| Número de registros | 19.190 |
| Número de columnas | 20 |
| Fuente de referencia | [Datos meteorológicos Aranzazu - datos.gov.co](https://www.datos.gov.co/dataset/Datos-meteorol-gicos-Aranzazu/nqj3-4xmv/about_data) |
| Formato de codificación | UTF-8 |


## Diccionario de variables

| # | Campo | Descripción | Unidad | Tipo en CSV | Tipo esperado para análisis | Valores o notas |
|---:|---|---|---|---|---|---|
| 1 | `Time` | Fecha y hora de la observación. | Fecha-hora | Texto | `datetime64[ns]` mediante `Time_dt` | No presenta faltantes en el archivo revisado. |
| 2 | `Interval` | Intervalo reportado entre mediciones. | Minutos, según el nombre del campo del dispositivo | Texto/número | Numérico entero | El archivo contiene distintos valores; debe validarse antes de asumir una frecuencia fija. |
| 3 | `Indoor Temperature(°C)` | Temperatura medida en el interior. | °C | Texto numérico | Numérico | Valores con coma decimal; se convierte en el EDA. |
| 4 | `Indoor Humidity(%)` | Humedad relativa medida en el interior. | % | Texto numérico | Numérico | No presenta faltantes observados. |
| 5 | `Outdoor Temperature(°C)` | Temperatura medida en el exterior. | °C | Texto numérico | Numérico | Valores con coma decimal; se convierte en el EDA. |
| 6 | `Outdoor Humidity(%)` | Humedad relativa medida en el exterior. | % | Texto numérico | Numérico | No presenta faltantes observados. |
| 7 | `Relative Pressure(hpa)` | Presión relativa registrada por la estación. | hPa | Texto numérico | Numérico | El nombre usa `hpa`; se conserva la unidad declarada por la fuente. |
| 8 | `Absolute Pressure(hpa)` | Presión atmosférica absoluta registrada. | hPa | Texto numérico | Numérico | Valores con coma decimal; se convierte en el EDA. |
| 9 | `Wind Speed(km/h)` | Velocidad del viento. | km/h | Texto numérico | Numérico | Valores con coma decimal; se convierte en el EDA. |
| 10 | `Gust(km/h)` | Velocidad máxima de ráfaga registrada. | km/h | Texto numérico | Numérico | Valores con coma decimal; se convierte en el EDA. |
| 11 | `Wind Direction` | Dirección desde la que sopla el viento. | Dirección cardinal | Texto categórico | Categórico | Incluye abreviaturas cardinales/intercardinales y `---` en 3 registros. |
| 12 | `DewPoint(°C)` | Temperatura de rocío. | °C | Texto numérico | Numérico | Valores con coma decimal; se convierte en el EDA. |
| 13 | `WindChill(°C)` | Temperatura aparente asociada al efecto del viento. | °C | Texto numérico | Numérico | Valores con coma decimal; se convierte en el EDA. |
| 14 | `Hour Rainfall(mm)` | Precipitación acumulada durante la última hora. | mm | Texto numérico | Numérico | Se convierte en el EDA. |
| 15 | `24 Hour Rainfall(mm)` | Precipitación acumulada durante las últimas 24 horas. | mm | Texto numérico | Numérico | Se convierte en el EDA. |
| 16 | `Week Rainfall(mm)` | Precipitación acumulada durante la semana. | mm | Texto numérico | Numérico | Se convierte en el EDA. |
| 17 | `Month Rainfall(mm)` | Precipitación acumulada durante el mes. | mm | Texto numérico | Numérico | Se convierte en el EDA. |
| 18 | `Total Rainfall(mm)` | Precipitación acumulada total reportada por la estación. | mm | Texto numérico | Numérico | Se convierte en el EDA; el periodo acumulado debe confirmarse con la fuente. |
| 19 | `Light(lux)` | Nivel de iluminación registrado. | lux | Texto numérico | Numérico | Se convierte en el EDA. |
| 20 | `UVI` | Índice de radiación ultravioleta. | Índice | Texto numérico | Numérico | En la revisión del archivo solo se observó el valor `0`; es una variable constante en este dataset. |