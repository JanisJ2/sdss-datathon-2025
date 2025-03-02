import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

neighborhoods = gpd.read_file("Neighbourhoods - 4326.geojson")

df = pd.read_csv("real-estate-data.csv")

df['geometry'] = df.apply(lambda row: Point(row['lg'], row['lt']), axis=1)

gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

if neighborhoods.crs != gdf.crs:
    neighborhoods = neighborhoods.to_crs(gdf.crs)

gdf = gpd.sjoin(gdf, neighborhoods[['AREA_NAME', 'geometry']], how="left", predicate="within")

df['neighbourhood'] = gdf['AREA_NAME']

df.drop(columns='geometry', inplace=True)

df.to_csv("real-estate-data-with-neighbourhood.csv", index=False)
