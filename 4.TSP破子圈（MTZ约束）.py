# 靡不有初，鲜克有终
# 开发时间：2023/8/21 9:21
# 调用Gurobi求解器求解

import numpy as np
import pandas as pd
from gurobipy import *

np.seterr(invalid='ignore')


# 根据经纬度输出实际球面距离:输入两点在df中的索引，输出两点的球面距离，单位千米
def distance_calculation(i_index, j_index):
    # 载入经纬度并将其变为弧度
    x1 = data_df.loc[i_index, 'longitude']
    y1 = data_df.loc[i_index, 'latitude']
    x2 = data_df.loc[j_index, 'longitude']
    y2 = data_df.loc[j_index, 'latitude']
    x1_radians = x1*math.pi/180
    y1_radians = y1*math.pi/180
    x2_radians = x2*math.pi/180
    y2_radians = y2*math.pi/180
    outcom = round(math.cos(x1_radians-x2_radians)*math.cos(y1_radians)*math.cos(y2_radians)+math.sin(y1_radians)*math.sin(y2_radians), 4)
    Dis = 6370*math.acos(outcom)
    return Dis


if __name__ == "__main__":
    # 导入数据，建立距离矩阵
    data_df = pd.read_csv(r"TSP_Data.txt")
    N = len(data_df)
    # print('点的数量是:',N)
    dis_mtx = np.zeros((N+1, N+1))
    for i in range(0, N):
        dis_mtx[i, N] = distance_calculation(i, 0)  # 为节点0创建一个虚拟节点
        dis_mtx[N, i] = distance_calculation(0, i)
        for j in range(0, N):
            dis_mtx[i, j] = distance_calculation(i, j)

    # 建立下标索引,含虚拟节点
    index_lst = []
    i_set = [i for i in range(0, N)]  # i集合
    j_set = [j for j in range(1, N+1)]  # j集合
    u_set = [i for i in range(0, N+1)]  # u集合
    for i in i_set:
        for j in j_set:
            if i != j:  # 排除掉i=j的情况
                index_lst.append((i, j))
    index_tplst = tuplelist(index_lst)

    '''________________________________建立优化函数_________________________________'''
    # 建立优化模型
    m = Model()
    m.setParam(GRB.Param.MIPGap, 0.01)  # 1%的gap

    # 建立变量
    x = m.addVars(index_tplst, vtype=GRB.BINARY, name='x')  # 二元整数变量
    u = m.addVars(u_set, vtype=GRB.CONTINUOUS, lb=0.0,  name='u')  # 非负连续变量
    m.update()
    # print(x)
    # print(u)

    # 建立目标函数
    m.setObjective(quicksum(dis_mtx[i, j]*x[i, j] for i, j in index_tplst))

    # 建立约束条件
    m.addConstrs(quicksum(x[i, j] for i, j in index_tplst.select('*', j)) == 1 for j in j_set)  # 流入约束
    m.addConstrs(quicksum(x[i, j] for i, j in index_tplst.select(i, '*')) == 1 for i in i_set)  # 流出约束
    m.addConstrs(u[i]-u[j] + N*x[i, j] <= N-1 for i, j in index_tplst)  # MTZ约束

    # 保存lp文件
    m.write('TSP_MTZ.lp')

    # 求解
    m.optimize()

    # 展示结果
    var_lst = m.getVars()
    print(var_lst)
    print("____________求解结果____________")
    print("变量值：")
    for i in range(0, len(var_lst)):
        if var_lst[i].X != 0:
            print(var_lst[i].VarName, var_lst[i].X)
    print("目标函数值：", m.ObjVal)
