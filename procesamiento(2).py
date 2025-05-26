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