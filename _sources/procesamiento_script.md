# Script de Procesamiento de Datos

A continuación se muestra el contenido del script utilizado para el procesamiento de datos, primero para regresion y luego para clasificacion.

## Código del Script
## procesamiento Regresion
```python

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Cargar datos y filtrar California
df = pd.read_csv("US_Accidents_March23.csv")
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

df_california = df[
    (df["State"] == "CA") 
    & (df["Start_Time"].notna())
    & (df["Start_Lat"].notna()) 
    & (df["Start_Lng"].notna())
].copy()

# 2. Parámetros clave
GRID_SIZE = 0.05  # Tamaño de celda en grados (~5.5 km)
MIN_ACCIDENTS = 3  # Mínimo de accidentes por zona-semana
MIN_YEARS = 2      # Mínimo de años con datos para considerar una zona

# 3. Bins espaciales mejorados
df_california["lat_bin"] = (df_california["Start_Lat"] // GRID_SIZE).astype(int)
df_california["lng_bin"] = (df_california["Start_Lng"] // GRID_SIZE).astype(int)
df_california["zone_id"] = df_california["lat_bin"].astype(str) + "_" + df_california["lng_bin"].astype(str)

# 4. Procesamiento temporal mejorado
df_california["year_week"] = df_california["Start_Time"].dt.strftime("%Y-%U")

# 5. Variables derivadas
df_california["is_weekend"] = df_california["Start_Time"].dt.dayofweek.ge(5).astype(int)
df_california["is_night"] = df_california["Start_Time"].dt.hour.between(20, 6, inclusive="right").astype(int)

# 6. Variables climáticas extremas
clim_vars = ["Temperature(F)", "Wind_Speed(mph)", "Precipitation(in)", "Humidity(%)"]
for var in clim_vars:
    threshold = df_california[var].quantile(0.95)
    df_california[f"{var}_extreme"] = df_california[var].gt(threshold).astype(int)

# 7. Agregación inicial
grouped = df_california.groupby(["zone_id", "year_week"]).agg(
    accident_count=("ID", "count"),
    Start_Lat=("Start_Lat", "mean"),
    Start_Lng=("Start_Lng", "mean"),
    lat_bin=("lat_bin", "first"),  # Mantener los bins
    lng_bin=("lng_bin", "first"),  # Mantener los bins
    **{var: (var, "mean") for var in clim_vars},
    **{f"{var}_extreme": (f"{var}_extreme", "sum") for var in clim_vars},
    **{var: (var, "sum") for var in ["Junction", "Railway", "Roundabout", "Station", "Stop", "Traffic_Signal"]},
    is_night=("is_night", "sum"),
    is_weekend=("is_weekend", "sum")
).reset_index()

# 8. Generar semanas completas
min_year = df_california["Start_Time"].dt.year.min()
max_year = df_california["Start_Time"].dt.year.max()
all_weeks = [f"{year}-{week:02d}" for year in range(min_year, max_year+1) for week in range(0, 53)]

# 9. Rellenar semanas faltantes
full_grid = []
for zone in grouped["zone_id"].unique():
    zone_data = grouped[grouped["zone_id"] == zone]
    complete_weeks = pd.DataFrame({"year_week": all_weeks, "zone_id": zone})
    merged = pd.merge(complete_weeks, zone_data, on=["zone_id", "year_week"], how="left")
    
    # Rellenar valores faltantes
    merged = merged.fillna({
        "accident_count": 0,
        **{var: 0 for var in ["Junction", "Railway", "Roundabout", "Station", "Stop", "Traffic_Signal"]},
        **{f"{var}_extreme": 0 for var in clim_vars},
        "is_night": 0,
        "is_weekend": 0
    })
    
    # Llenar valores geográficos y climáticos
    for var in ["Start_Lat", "Start_Lng", "lat_bin", "lng_bin"] + clim_vars:
        merged[var] = merged[var].fillna(zone_data[var].mean())
    
    full_grid.append(merged)

grouped = pd.concat(full_grid).reset_index(drop=True)

# 10. Calcular límites geográficos
grouped["lat_min"] = grouped["lat_bin"].astype(float) * GRID_SIZE
grouped["lat_max"] = (grouped["lat_bin"].astype(float) + 1) * GRID_SIZE
grouped["lng_min"] = grouped["lng_bin"].astype(float) * GRID_SIZE
grouped["lng_max"] = (grouped["lng_bin"].astype(float) + 1) * GRID_SIZE

# 11. Filtrar zonas con suficiente historia
zone_years = grouped.groupby("zone_id")["year_week"].nunique() // 52
valid_zones = zone_years[zone_years >= MIN_YEARS].index
grouped = grouped[grouped["zone_id"].isin(valid_zones)]

# 12. Escalado de variables climáticas
scaler_clim = StandardScaler()
grouped[clim_vars] = scaler_clim.fit_transform(grouped[clim_vars])

# 13. Exportar dataset final
grouped.to_csv("california_accidents_cleaned_complete.csv", index=False)

print("Procesamiento completado exitosamente!")
print("Columnas en el dataset:", grouped.columns.tolist())

```
## Procesamiento Clasificacion



```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("--- Iniciando la generación del dataset 'grouped' para modelado de severidad ---")

# 1. Cargar datos y filtrar California
df_raw = pd.read_csv("accidentes.csv") # Ensure correct path
df_raw["Start_Time"] = pd.to_datetime(df_raw["Start_Time"], errors="coerce")

df_california = df_raw[
    (df_raw["State"] == "CA")
    & (df_raw["Start_Time"].notna())
    & (df_raw["Start_Lat"].notna())
    & (df_raw["Start_Lng"].notna())
].copy()

if 'Severity' not in df_california.columns:
    print("ERROR: Columna 'Severity' no encontrada. No se puede proceder.")
    exit()
df_california['severity_class'] = df_california['Severity'].astype(int)
print("Distribución de 'severity_class' en datos crudos de CA:")
print(df_california['severity_class'].value_counts(normalize=True).sort_index())

# 2. Parámetros clave
GRID_SIZE = 0.05
MIN_YEARS = 2 # Para filtrar zonas después de la agregación

# 3. Bins espaciales
df_california["lat_bin"] = (df_california["Start_Lat"] // GRID_SIZE).astype(int)
df_california["lng_bin"] = (df_california["Start_Lng"] // GRID_SIZE).astype(int)
df_california["zone_id"] = df_california["lat_bin"].astype(str) + "_" + df_california["lng_bin"].astype(str)

# 4. Procesamiento temporal
df_california["year_week"] = df_california["Start_Time"].dt.strftime("%Y-%U")

# 5. Variables derivadas
df_california["is_weekend"] = df_california["Start_Time"].dt.dayofweek.ge(5).astype(int)
df_california["is_night"] = df_california["Start_Time"].dt.hour.between(20, 6, inclusive="right").astype(int)

# 6. Variables climáticas extremas
clim_vars = ["Temperature(F)", "Wind_Speed(mph)", "Precipitation(in)", "Humidity(%)"]
for var in clim_vars:
    if var in df_california.columns:
        df_california[var] = pd.to_numeric(df_california[var], errors='coerce')
        if df_california[var].notna().any():
            threshold = df_california[var].quantile(0.95)
            df_california[f"{var}_extreme"] = df_california[var].gt(threshold).astype(int)
        else:
            df_california[f"{var}_extreme"] = 0
    else:
        df_california[f"{var}_extreme"] = 0

# 7. Agregación inicial (Esta es la base para tu modelado)
agg_dict = {
    "accident_count": ("ID", "count"),
    "max_severity_class": ("severity_class", "max"),
    "Start_Lat": ("Start_Lat", "mean"),
    "Start_Lng": ("Start_Lng", "mean"),
    "lat_bin": ("lat_bin", "first"),
    "lng_bin": ("lng_bin", "first"),
    # Ensure only existing columns are aggregated
    **{var: (var, "mean") for var in clim_vars if var in df_california.columns},
    **{f"{var}_extreme": (f"{var}_extreme", "sum") for var in clim_vars}, # _extreme columns were created
    **{var: (var, "sum") for var in ["Junction", "Railway", "Roundabout", "Station", "Stop", "Traffic_Signal"] if var in df_california.columns},
    "is_night": ("is_night", "sum"),
    "is_weekend": ("is_weekend", "sum")
}
final_agg_dict = {k:v for k,v in agg_dict.items() if v[0] in df_california.columns or k=="accident_count"} # Simplified check

if not final_agg_dict:
    print("ERROR: No valid aggregation operations. Check input columns.")
    exit()

grouped = df_california.groupby(["zone_id", "year_week"]).agg(**final_agg_dict).reset_index()

if 'max_severity_class' not in grouped.columns:
    print("ERROR: 'max_severity_class' no fue agregada. Verifique 'severity_class' en los datos crudos y la lógica de agregación.")
    exit()

grouped['severity_binary'] = grouped['max_severity_class'].apply(lambda x: 1 if pd.notna(x) and x >= 3 else 0)
grouped.drop('max_severity_class', axis=1, inplace=True, errors='ignore')

print("\nDistribución de 'severity_binary' en 'grouped' (solo zone-weeks con accidentes):")
print(grouped['severity_binary'].value_counts(normalize=True).sort_index())
print(grouped['severity_binary'].value_counts().sort_index())

# Add year and week for filtering and sorting
temp_split_yw = grouped['year_week'].str.split('-', expand=True)
grouped['year'] = temp_split_yw[0].astype(int)
grouped['week'] = temp_split_yw[1].astype(int)
del temp_split_yw

# 11. Filtrar zonas con suficiente historia (aplicado a 'grouped')
zone_years = grouped.groupby("zone_id")["year"].nunique() #nunique on year column
valid_zones = zone_years[zone_years >= MIN_YEARS].index
modeling_df = grouped[grouped["zone_id"].isin(valid_zones)].copy() # Use .copy()

# Optional: Calcular límites geográficos (si los necesitas como features)
if 'lat_bin' in modeling_df.columns and 'lng_bin' in modeling_df.columns:
    modeling_df["lat_min"] = modeling_df["lat_bin"].astype(float) * GRID_SIZE
    modeling_df["lat_max"] = (modeling_df["lat_bin"].astype(float) + 1) * GRID_SIZE
    modeling_df["lng_min"] = modeling_df["lng_bin"].astype(float) * GRID_SIZE
    modeling_df["lng_max"] = (modeling_df["lng_bin"].astype(float) + 1) * GRID_SIZE

# Guardar este DataFrame para el modelado
output_filename = "revisedds.csv"
modeling_df.to_csv(output_filename, index=False)

print(f"\nDataset para modelado guardado en: {output_filename}")
print("Columnas:", modeling_df.columns.tolist())
print("Forma:", modeling_df.shape)
print("Distribución final de 'severity_binary' en el CSV guardado (solo semanas activas):")
print(modeling_df['severity_binary'].value_counts(normalize=True).sort_index())
print(modeling_df['severity_binary'].value_counts().sort_index())
```