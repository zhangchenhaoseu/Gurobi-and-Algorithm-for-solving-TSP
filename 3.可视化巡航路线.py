# 靡不有初，鲜克有终
# 开发时间：2023/4/13 16:23
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import json
from shapely import geometry


data_df = pd.read_csv(r"C:\Users\张晨皓\Desktop\博一课程\数学模型\第三次作业\data\TSP_Data.txt")
optimal_path_df = pd.read_csv(r"C:\Users\张晨皓\Desktop\博一课程\数学模型\第三次作业\result\optimal_path.txt",header=None, names=["node_id"])
print(optimal_path_df)
data_gdf = gpd.GeoDataFrame(data_df, geometry=gpd.points_from_xy(data_df.longitude, data_df.latitude))
with open('cities.json', 'w') as f:
    f.write(data_gdf.to_json())
with open("cities.json",'r',encoding='utf8') as fp:
    node_json = json.load(fp)
    node_gdf = gpd.GeoDataFrame.from_features(node_json["features"])

MultiLine_lst = []
for i in range(0, len(optimal_path_df)-1):
    start_index = optimal_path_df.loc[i, "node_id"]-1
    end_index = optimal_path_df.loc[i+1, "node_id"]-1
    start_node_x = data_df.loc[start_index, 'longitude']
    start_node_y = data_df.loc[start_index, 'latitude']
    end_node_x = data_df.loc[end_index, 'longitude']
    end_node_y = data_df.loc[end_index, 'latitude']
    MultiLine_lst.append([(start_node_x, start_node_y), (end_node_x,end_node_y)])
    Path_gdf = gpd.GeoSeries([geometry.MultiLineString(MultiLine_lst)], index=['a'])
    with open('Path.json', 'w') as f:  # 将最优路径保存为json
        f.write(Path_gdf.to_json())
    with open("Path.json",'r',encoding='utf8') as fp:
        link_json = json.load(fp)
        link_gdf = gpd.GeoDataFrame.from_features(link_json["features"])

# 画标签
fig, ax = plt.subplots(figsize=(25, 20), dpi=150)
ax = node_gdf.plot(ax=ax, lw=2, edgecolor='blue', facecolor='yellow', markersize = 60, zorder=2)
ax = link_gdf.plot(ax=ax, lw=2, edgecolor='red', facecolor=None, zorder=1)
for i in range(0, len(data_df)):
    if i == 0:
        x = data_df.loc[i, 'longitude'] + 0.25
        y = data_df.loc[i, 'latitude'] + 0.25
        label = "1-start"
        plt.text(x, y, str(label), family='serif', style='italic', fontsize=20, verticalalignment="bottom", ha='left',color='red')
    else:
        x = data_df.loc[i, 'longitude'] + 0.25
        y = data_df.loc[i, 'latitude'] + 0.25
        label = i+1
        plt.text(x, y, str(label), family='serif', style='italic', fontsize=15, verticalalignment="bottom", ha='left',color='k')
plt.savefig(r"C:\Users\张晨皓\Desktop\博一课程\数学模型\第三次作业\figure\3.飞机巡航TSP最短路示意图.png")
plt.show()


