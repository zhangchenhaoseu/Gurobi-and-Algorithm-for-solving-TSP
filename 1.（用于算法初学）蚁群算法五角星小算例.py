# 靡不有初，鲜克有终
# 开发时间：2023/4/13 13:56
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import time


# 按照轮盘赌算法随机选择下一个目标:输入各个概率组成的列表，输出所选择对象的索引
def random_selection(rate):
    # """随机变量的概率函数"""
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    rate_new = copy.deepcopy(rate)
    for i in range(0, len(rate_new)):
        rate_new[i] = (10**5)*round(rate[i], 5)  # 将概率扩大一定倍数，以防止出现下面int(sum(rate_new))的报错，并不影响最终的概率
    start = 0
    index = 0
    randnum = random.randint(1, int(sum(rate_new)))
    for index, scope in enumerate(rate_new):
        start += scope
        if randnum <= start:
            break
    return index


# 计算每一个OD结点对的选择概率:输入弗洛蒙浓度矩阵tau_m(tau_mtx)、能见度矩阵eta_mtx(eta_mtx)、搜索禁忌列表taboo_l(taboo_lst)、当前结点索引，输出选择概率矩阵
def calculate_probability(tau_m, eta_m, taboo_l, current_index):
    numerator_lst = [0 for i in range(0, city_number)]  # 初始化分子列表
    prob_lst = [0 for i in range(0, city_number)]  # 初始化选择概率列表
    for i in range(0, len(taboo_l)):
        if taboo_l[i] == 1:  # 根据禁忌表中，可以访问的城市，访问过的城市为0，没有的为1，1表示可以访问
            numerator_lst[i] = (tau_m[current_index, i] ** alpha) * (eta_m[current_index, i] ** beta)  # 分子的计算
    total = sum(numerator_lst)
    for i in range(0, len(prob_lst)):
        prob_lst[i] = numerator_lst[i] / total

        # print(prob_lst[i],numerator_lst[i],total)
    return prob_lst


# 蚁群算法，单只蚂蚁的视角:输入起点，和初始弗洛蒙矩阵，不重复地走完所有其他结点，最后回到起点，输出此时新的弗洛蒙矩阵，和距离dis
def ant_colony_optimization(start_index, tau_m):
    current_index = start_index
    path_index = []  # 用于存储走过的【结点】索引
    taboo_lst = [1 for i in range(0, city_number)]
    taboo_lst[start_index] = 0
    path_index.append(current_index)

    while sum(taboo_lst) != 0:  # 不重复地访问除了起点之外的所有结点
        next_index = random_selection(calculate_probability(tau_m, eta_mtx, taboo_lst, current_index))  # 下一个结点的索引
        taboo_lst[next_index] = 0
        path_index.append(next_index)
        current_index = next_index
    path_index.append(start_index)  # 由若干结点的索引，构成的路径

    return path_index


if __name__ == "__main__":

    alpha = 1  # =0时容易陷入局部最优解
    beta = 1   # =0时收敛过快，也无法最优
    Q = 1  # 佛罗蒙更新常量
    rho = 0.9  # 消散系数
    ant_number = 100  # 蚂蚁数
    rounds = 100  # 轮数
    city_number = 5
    times = 1  # 仿真轮次计数
    K = 20  # 邻边优化算法的次数

    dis_mtx = np.zeros((5, 5))  # 初始化距离矩阵
    tau_mtx = np.ones((5, 5))  # 初始化弗洛蒙矩阵（是一个有向非对称矩阵？）
    eta_mtx = np.zeros((5, 5))  # 初始化能见度矩阵

    dis_mtx[0, 1], dis_mtx[0, 2], dis_mtx[0, 3], dis_mtx[0, 4] = 1, 1, 1.62, 1.62
    dis_mtx[1, 0], dis_mtx[1, 2], dis_mtx[1, 3], dis_mtx[1, 4] = 1, 1.62, 1.62, 1
    dis_mtx[2, 0], dis_mtx[2, 1], dis_mtx[2, 3], dis_mtx[2, 4] = 1, 1.62, 1, 1.62
    dis_mtx[3, 0], dis_mtx[3, 1], dis_mtx[3, 2], dis_mtx[3, 4] = 1.62, 1.62, 1, 1
    dis_mtx[4, 0], dis_mtx[4, 1], dis_mtx[4, 2], dis_mtx[4, 3] = 1.62, 1, 1.62, 1

    for i in range(0, len(dis_mtx)):
        for j in range(0, len(dis_mtx)):
            if dis_mtx[i, j] != 0:
                eta_mtx[i, j] = 1 / dis_mtx[i, j]

    point_lst = []
    policy = 0
    average_dis_lst = []  # 单轮次所有蚂蚁的平均距离
    shortest_dis_lst = []  # 单轮次所有蚂蚁的最短距离
    shortest_till_now_lst = []  # 用来储存截止到目前的最短距离
    optimal_policy_round = []  # 初始化单轮次内，蚂蚁的最优行驶路径
    optimal_policy = []  # 初始化全局次内，蚂蚁的最优行驶路径

    # 调用函数进行求解
    tic = time.perf_counter()
    while times <= rounds:
        policy_mtx = []  # 初始化每只蚂蚁的行驶路径，里面具有蚂蚁数量个的路径（每一个都是由结点序列组成的列表）
        sigle_round_dis_lst = []  # 单轮次中，记录每只蚂蚁的访问总距离
        tau_mtx_round = copy.deepcopy(tau_mtx)  # 在同一轮的概率计算中所使用的不变的弗洛蒙矩阵
        tau_mtx = copy.deepcopy(tau_mtx * rho)
        # 每一只蚂蚁进行仿真
        for i in range(0, ant_number):
            start_index = random.randint(0, city_number-1)  # 随机生成蚂蚁的起点
            policy = ant_colony_optimization(start_index, tau_mtx_round)
            policy_mtx.append(policy)  # policy_mtx里面是在当前这一轮中，每一个蚂蚁跑出来的路径[1,17,34...]组成的二维列表[[],[],[]...]

        # 进行20次的2-邻边优化算法,调整最终的每一只蚂蚁的TSP
        for m in range(0, len(policy_mtx)):  # 每一只蚂蚁的TSP
            distance = 0  # 每一只蚂蚁调整后的TSP长度
            path = policy_mtx[m]  # 导出第k个蚂蚁路径[1,17,....,1],len(path)=102
            for k in range(0, K):  # 2-邻边算法循环，假设进行K轮改进
                flag = 0  # 2-邻边算法的退出标志
                for i in range(1, len(path)-4):  # len(path)=102 ; city_number = 101; len(dis_mtx)=101
                    for j in range(i+2, len(path)-2):
                        if (dis_mtx[path[i], path[j]] + dis_mtx[path[i+1], path[j+1]]) < (dis_mtx[path[i],path[i+1]] + dis_mtx[path[j],path[j+1]]):
                            path[i+1:j+1] = path[j:i:-1]  # [i+1:j+1]包括了[i+1,i+2,...,j];[j:i:-1]包括了[j,j-1,...,i+1],切片的左闭右开特性
                            flag = 1
                if flag == 0:
                    break
            for i in range(0, len(path)-1):
                distance += dis_mtx[path[i], path[i+1]]

            sigle_round_dis_lst.append(distance)  # 存储了每一只蚂蚁的行驶距离
            delta_tau = Q / distance  # 增加的弗洛蒙值
            for i in range(0, len(path)-1):
                tau_mtx[path[i], path[i+1]] = tau_mtx[path[i], path[i+1]] + delta_tau  # 更新弗洛蒙矩阵

        average_dis_lst.append(np.average(sigle_round_dis_lst))  # 用来存储这一轮中n个蚂蚁的平均行驶距离
        shortest_dis_lst.append(min(sigle_round_dis_lst))  # 用来存储这一轮中n个蚂蚁的最短距离
        shortest_till_now = min(shortest_dis_lst)  # 截止到目前的最短距离
        shortest_till_now_lst.append(shortest_till_now)  # 用来储存截止到目前的最短距离
        optimal_policy_index = sigle_round_dis_lst.index(min(sigle_round_dis_lst))  # 找到当前轮次最短行驶距离对应的蚂蚁索引
        optimal_policy_round = policy_mtx[optimal_policy_index]

        if min(sigle_round_dis_lst) == shortest_till_now:
            optimal_policy = optimal_policy_round  # 此时该轮最优策略即为optimal_policy
        # print(len(sigle_round_dis_lst))
        times += 1
    toc = time.perf_counter()

    print()
    print(f"最短距离为{min(shortest_dis_lst):0.4f}单位长度")
    print(f"最优策略为{optimal_policy}")
    print(f"计算耗时:{toc - tic:0.4f}秒")

    # 准备画布，保存并展示
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    ax.plot(average_dis_lst, color='r', linewidth=1, markersize=1, label="average length")  # 每一轮的【平均】距离
    ax.plot(shortest_till_now_lst, color='b', linewidth=1, markersize=1, label="shortest length")  # 每一轮的【最短】距离
    ax.set_title("Convergence curve", fontfamily="Times New Roman", fontsize=20, loc="left")
    plt.legend()
    plt.savefig(r"C:\Users\张晨皓\Desktop\博一课程\数学模型\第三次作业\figure\1.ACO五角星小案例收敛曲线.png", dpi=100)
    plt.show()