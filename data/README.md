# Diccionarios de Datos
Detalles técnicos del archivo y diccionario de datos de cada fase del procesamiento:

- **Bronze:** Datos crudos originales.
- **Silver:** Datos limpios.
- **Gold:** Datos preparados para el modelamiento.

<br>

# Bronze

## Detalles del Archivo

| Elemento | Detalle |
|---|---|
| Nombre del archivo | `meteorologicos_aranzazu_raw.csv` |
| Ubicación | `Proyecto/data/01_bronze/` |
| Descripción | Registros de variables meteorológicas y ambientales observadas en Aranzazu, Colombia, durante los años 2018 y 2019 |
| Estado | Datos crudos |
| Número de registros | 19.190 |
| Número de columnas | 20 |
| Fuente de referencia | [Datos meteorológicos Aranzazu - datos.gov.co](https://www.datos.gov.co/dataset/Datos-meteorol-gicos-Aranzazu/nqj3-4xmv/about_data) |
| Formato de codificación | UTF-8 |


## Diccionario de Variables

| # | Campo | Descripción | Unidad | Tipo en CSV | Valores o notas |
|---:|---|---|---|---|---|
| 1 | `Time` | Fecha y hora de la observación. | Fecha-hora | Texto | No presenta faltantes observados. |
| 2 | `Interval` | Intervalo reportado entre mediciones. | Minutos, según el nombre del campo del dispositivo | Texto/número | El archivo contiene distintos valores debe validarse antes de asumir una frecuencia fija. |
| 3 | `Indoor Temperature(°C)` | Temperatura medida en el interior. | °C | Texto | Valores con coma decimal  |
| 4 | `Indoor Humidity(%)` | Humedad relativa medida en el interior. | % | Texto | No presenta faltantes observados. |
| 5 | `Outdoor Temperature(°C)` | Temperatura medida en el exterior. | °C | Texto | Valores con coma decimal  |
| 6 | `Outdoor Humidity(%)` | Humedad relativa medida en el exterior. | % | Texto | No presenta faltantes observados. |
| 7 | `Relative Pressure(hpa)` | Presión relativa registrada por la estación. | hPa | Texto | El nombre usa `hpa` se conserva la unidad declarada por la fuente. |
| 8 | `Absolute Pressure(hpa)` | Presión atmosférica absoluta registrada. | hPa | Texto | Valores con coma decimal  |
| 9 | `Wind Speed(km/h)` | Velocidad del viento. | km/h | Texto | Valores con coma decimal  |
| 10 | `Gust(km/h)` | Velocidad máxima de ráfaga registrada. | km/h | Texto | Valores con coma decimal  |
| 11 | `Wind Direction` | Dirección desde la que sopla el viento. | Dirección cardinal | Texto | Incluye abreviaturas cardinales/intercardinales y `---` en 3 registros. |
| 12 | `DewPoint(°C)` | Temperatura de rocío. | °C | Texto | Valores con coma decimal  |
| 13 | `WindChill(°C)` | Temperatura aparente asociada al efecto del viento. | °C | Texto | Valores con coma decimal.  |
| 14 | `Hour Rainfall(mm)` | Precipitación acumulada durante la última hora. | mm | Texto | No presenta faltantes observados. |
| 15 | `24 Hour Rainfall(mm)` | Precipitación acumulada durante las últimas 24 horas. | mm | Texto | No presenta faltantes observados. |
| 16 | `Week Rainfall(mm)` | Precipitación acumulada durante la semana. | mm | Texto | No presenta faltantes observados. |
| 17 | `Month Rainfall(mm)` | Precipitación acumulada durante el mes. | mm | Texto | No presenta faltantes observados. |
| 18 | `Total Rainfall(mm)` | Precipitación acumulada total reportada por la estación. | mm | Texto |  |
| 19 | `Light(lux)` | Nivel de iluminación registrado. | lux | Texto | No presenta faltantes observados. |
| 20 | `UVI` | Índice de radiación ultravioleta. | Índice | Texto | En la revisión del archivo solo se observó el valor `0` es una variable constante en este dataset. |