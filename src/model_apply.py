# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:01:51 2021

@author: 86159
"""

import math
import os
import numpy as np
# import matplotlib.pyplot as plt
import json
import pandas as pd
# import random
import time
import copy
# import datetime
# from threading import Timer
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def find_station(stops_route):
    return stops_route[stops_route["type"]=="Station"].index.to_list()[0]

def fullfillzone(stops_route):
    # 除了station，其他的stop都归到离他最近的zone中
    for s in stops_route.index.to_list():
        if s != find_station(stops_route):
            if pd.isnull(stops_route.loc[s,'zone_id'])==True:
                min_=100000
                for s2 in stops_route.index.to_list():
                    lat1=stops_route.loc[s,'lat']
                    lng1=stops_route.loc[s,'lng']
                    lat2=stops_route.loc[s2,'lat']
                    lng2=stops_route.loc[s2,'lng']
                    # print(lat1, lng1, lat2, lng2)
                    dis=getDistance(lat1, lng1, lat2, lng2)
                    # print(dis)
                    if dis<min_ and s != s2 and pd.isnull(stops_route.loc[s2,'zone_id'])==False: 
                        min_=dis
                        min_stop=s2
                stops_route.loc[s,'zone_id']=stops_route.loc[min_stop,'zone_id']
    return stops_route

# 返回和stop相同zone的一条route里的所有stop
def find_zone_stop(stop):
    zone=stops_route.loc[stop,'zone_id']
    if pd.isnull(zone)==False:
        stopls=stops_route.loc[stops_route['zone_id']==zone].index.to_list() 
    else: stopls=[stop]#若stop的zone是nan，则返回本身
    return stopls


def find_zone_center():
    zonels={}
    stopls=[]
    for s in data_travel_route.index.to_list():
        if stops_route.loc[s,'zone_id'] in zonels.keys(): continue
        stopls=find_zone_stop(s)
        sum_x=0
        sum_y=0
        for samezone_s in stopls:
            sum_x+=stops_route.loc[samezone_s,'lat']
            sum_y+=stops_route.loc[samezone_s,'lng']
        zonels[stops_route.loc[samezone_s,'zone_id']]=[]
        zonels[stops_route.loc[samezone_s,'zone_id']].append(sum_x/len(stopls))
        zonels[stops_route.loc[samezone_s,'zone_id']].append(sum_y/len(stopls))

    return zonels

    
def greedy_zone_seq():
    station=find_station(stops_route)
    lat1=stops_route.loc[station,'lat']
    lng1=stops_route.loc[station,'lng']
    zonels=find_zone_center()
    
    # 寻找离station最近的作为第一个zone
    min_=10000000
    for i in zonels.keys():
        lat2=zonels[i][0]
        lng2=zonels[i][1]
        dis=getDistance(lat1, lng1, lat2, lng2)
        if dis<min_:
            min_=dis
            zone1=i
    
    # 得到disX的dataframe
    disX={}
    for z1 in zonels.keys():
        disX[z1]={}
        lat1=zonels[z1][0]
        lng1=zonels[z1][1]
        for z2 in zonels.keys():
            lat2=zonels[z2][0]
            lng2=zonels[z2][1]
            disX[z1][z2]=getDistance(lat1, lng1, lat2, lng2)
    
    disX=pd.DataFrame(disX)
    return greedy(disX,zone1)
    
    
def rad(d):
    return math.pi/180.0*d

def getDistance(lat1,lng1,lat2,lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(
        math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    EARTH_RADIUS = 6378.137
    s = s * EARTH_RADIUS
    s = (s * 10000) / 10
    return s




class Instance:
    
    def __init__(self, file_name):
        self.file_name=file_name
        
        self.data_route, self.route_num, self.stops = self.get_route_data(file_name[0])
        self.data_travel = self.get_travel_times(file_name[1])
        self.data_package,self.window_stop_info = self.get_package_data(file_name[2])
        # self.data_sequences = self.get_actual_sequences(file_name[3])
        
        
    def get_route_data(self, file_name):
        with open(file_name,encoding='utf-8') as f1:
            line = f1.readline()
            d1 = json.loads(line)
            f1.close()
            
        data_route=pd.DataFrame(d1).T
        
        ## 去除掉stop的数据框
        # columnList=data_route.columns.to_list().remove('stops')
        # data_route2=data_route[['station_code','date_YYYY_MM_DD',
        #                         'departure_time_utc','executor_capacity_cm3','route_score']]
        
        data_route2=data_route[['station_code','date_YYYY_MM_DD',
                                'departure_time_utc','executor_capacity_cm3']]
        
        # print(data_route.columns.to_list())
        route_num=data_route.shape[0]
        
    
        ## stops
        stops={}
        for routeIdx in data_route.index.to_list():
            stops[routeIdx]={}
            for stopID,stopinfo in data_route.loc[routeIdx,'stops'].items():
                stops[routeIdx][stopID]=stopinfo
    
            stops[routeIdx]=pd.DataFrame(stops[routeIdx]).T
    
        return data_route2,route_num,stops
        
    def get_travel_times(self, file_name):
        with open(file_name,encoding='utf-8') as f2:
            line = f2.readline()
            d2 = json.loads(line)
            f2.close()
            
        data_travel={}
        for route in d2.keys():
            data_travel[route]={}
            for stop1 in d2[route].keys():
                data_travel[route][stop1]={}
                for stop2 in d2[route][stop1].keys():
                    data_travel[route][stop1][stop2]=d2[route][stop1][stop2]
            data_travel[route]=pd.DataFrame(data_travel[route])
        return data_travel
    
    def get_package_data(self, file_name):
        with open(file_name,encoding='utf-8') as f3:
            line = f3.readline()
            d3 = json.loads(line)
            f3.close()
            
        # 输出window_stop_info
        window_stop_info={}
        for route in d3.keys():
            window_stop_info[route]={}
            for stop in d3[route].keys():
                window_stop_info[route][stop]=[]
                for package in d3[route][stop].keys():
                    start_time_utc=d3[route][stop][package]["time_window"]["start_time_utc"] 
                    end_time_utc=d3[route][stop][package]["time_window"]["end_time_utc"] 
                    
                    # if start_time_utc!=nan or end_time_utc!=" :
                    window_stop_info[route][stop].append([start_time_utc,end_time_utc])
        
        
        return d3,window_stop_info
    
    def get_actual_sequences(self, file_name):
        with open(file_name,encoding='utf-8') as f4:
            line = f4.readline()
            d4 = json.loads(line)
            f4.close()
        
        data_sequences={}
        for route in d4.keys():
            data_sequences[route]={}
            sequence=d4[route]['actual']#一个route的sequences
            for stopID,orderID in sequence.items():
                if orderID>-1: # 忽略index为-1，即该route不会经过的stop
                    data_sequences[route][orderID]=stopID
            # 排序
            temp=[]
            data_sequences[route]=dict(sorted(data_sequences[route].items(),key=lambda d:d[0]))
            for i in data_sequences[route].values():
                temp.append(i)
                
            data_sequences[route] = temp
        
        return data_sequences
    
def cost_sum(w1,w2,data_travel_route,stops_route):
    data_distance_route={}
    for i in range(stops_route.shape[0]):
        stop1=stops_route.index.to_list()[i]
        data_distance_route[stop1]={}
        
        for j in range(stops_route.shape[0]):
            stop2=stops_route.index.to_list()[j]
            
            lat1=stops_route.loc[stop1,'lat']
            lng1=stops_route.loc[stop1,'lng']
            lat2=stops_route.loc[stop2,'lat']
            lng2=stops_route.loc[stop2,'lng']
            
            data_distance_route[stop1][stop2]=getDistance(lat1, lng1, lat2, lng2)
            
    data_distance_route=pd.DataFrame(data_distance_route)
    cost_sum=w1*data_travel_route+w2*data_distance_route
    return cost_sum




def greedy(cost,station):#stops_route,find_station
    sumpath=0
    seq=[]
    n=cost.shape[0]
    
    stop=station
    seq.append(stop)
    flag={}
    for s in cost.index.to_list():
        flag[s]=0
    flag[stop]=1
    
    while len(seq)<n:
        # 寻找min
        min_travel_time=100000
        # min_stop=stop
        for stop2 in cost.index.to_list():
            if flag[stop2]==0 and cost.loc[stop,stop2]<min_travel_time:#  and stop2!=stop
                min_travel_time=cost.loc[stop,stop2]
                min_stop=stop2
        
        stop=min_stop
        seq.append(stop)
        flag[stop]=1
        sumpath+=min_travel_time
        
    return seq



class DP:
    def __init__(self,X,start_node):
        self.cost=X
        self.X = np.array(X) #距离矩阵
        self.start_node =  start_node#开始的节点
        self.array = [[0]*(2**(len(self.X))) for i in range(len(self.X))] #记录处于x节点，未经历M个节点时，矩阵储存x的下一步是M中哪个节点 #len(self.X)-1
 
    def transfer(self, sets):
        su = 0
        for s in sets:
            su = su + 2**(s) # 二进制转换# s-1
        return su # int(su)
 
    # tsp总接口
    def tsp(self):
        s = self.start_node
        num = len(self.X)
        cities = list(range(num)) #形成节点的集合
        # past_sets = [s] #已遍历节点集合
        cities.pop(cities.index(s)) #构建未经历节点的集合
        node = s #初始节点
        return self.solve(node, cities) #求解函数
 
    def solve(self, node, future_sets):
        # 迭代终止条件，表示没有了未遍历节点，直接连接当前节点和起点即可
        if len(future_sets) == 0:
            return 0#self.X[node][self.start_node]
        d = 99999
        # node如果经过future_sets中节点，最后回到原点的距离
        distance = []
        # 遍历未经历的节点
        for i in range(len(future_sets)):
            s_i = future_sets[i]
            copy = future_sets[:]
            copy.pop(i) # 删除第i个节点，认为已经完成对其的访问
            # print(self.X)
            # print(node)
            # print(s_i)
            distance.append(self.X[node][s_i] + self.solve(s_i,copy))
        # 动态规划递推方程，利用递归
        d = min(distance)
        # node需要连接的下一个节点
        next_one = future_sets[distance.index(d)]
        # 未遍历节点集合
        c = self.transfer(future_sets)
        # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
        self.array[node][c] = next_one
        return d
    def main(self):
        # t=Timer(1, greedy(self.X,self.X.index.to_list()[self.start_node]))
        self.tsp()
        # 开始回溯
        # M = self.array
        lists = list(range(len(self.X)))
        start = self.start_node
        city_order = []
        # t=0
        while len(lists) > 0:
            # endTime=time.time()
            # if endTime-startTime>10: 
            #     t=1
            #     break
            if start in lists:
                # print("---")
                lists.pop(lists.index(start))
                m = self.transfer(lists)
                next_node = self.array[start][m]
                # print(start,"--->" ,next_node)
                city_order.append(start)
                start = next_node
        stopls=[]
        for i in city_order:
            stopls.append(self.cost.index.to_list()[i])
        # if t==1:
        #     stopls=greedy(self.cost,self.cost.index.to_list()[self.start_node])
        return stopls



class LS:
    
    def __init__(self, disX, initial):
        
        self.cost = disX
        self.num = disX.shape[0]
        self.initial_sequences=initial
        self.seqlist=[]
    
    def Linju(self,seq):
        # exchange
        seqlist=[]
        for i in range(len(seq)):
            for j in range(1,len(seq)):
                temp_seq=copy.deepcopy(seq)
                temp = temp_seq[i]
                temp_seq[i] = temp_seq[j]
                temp_seq[j] = temp
                if self.evaluate(temp_seq)<self.evaluate(seq):
                   seqlist.append(temp_seq)
        return seqlist
    
    # def Linju(self,seq):
    #     # exchange
    #     best_=10000000
    #     best_seq=copy.deepcopy(seq)
    #     for i in range(len(seq)):
    #         for j in range(1,len(seq)):
    #             temp_seq=copy.deepcopy(seq)
    #             temp = temp_seq[i]
    #             temp_seq[i] = temp_seq[j]
    #             temp_seq[j] = temp
    #             if self.evaluate(temp_seq)<best_:
    #                best_seq=temp_seq
    #                best_=self.evaluate(temp_seq)
    #     return best_seq

    def evaluate(self,seq):
        total_cost=0
        for i in range(1,self.num):
            total_cost+=self.cost[seq[i-1]][seq[i]]
        return total_cost
        
    def solve(self):

        initial_sequences = self.initial_sequences
        # initial_evaluation = self.evaluate(initial_sequences)
        best_sequences = self.initial_sequences
        best_evaluation = self.evaluate(best_sequences)
        # print(best_evaluation)
          
        self.seqlist.extend(self.Linju(initial_sequences))
        cnt=0
        while cnt<1:
            if len(self.seqlist)>0:
                temp_sequences = self.seqlist.pop()
            else: break
                
            temp_evaluation = self.evaluate(temp_sequences)
            
            if temp_evaluation < best_evaluation:
                best_sequences = temp_sequences
                best_evaluation = temp_evaluation
            else: cnt+=1
            # print(best_evaluation)
  			
            if len(self.Linju(temp_sequences))>0:
                self.seqlist.extend(self.Linju(temp_sequences))
            
        
        self.best_sequences = best_sequences
        self.best_evaluation = best_evaluation
        
        # cnt = 0
        # seqlist=[]
        # while len(seqlist) != 0 :
        #     temp=[]
        #     temp_sequences = self.Linju(initial_sequences)
        #     temp_evaluation = self.evaluate(temp_sequences)
        #     if temp_evaluation < self.evaluate(initial_sequences):
                
        #     if temp_evaluation < best_evaluation:
        #         best_sequences = temp_sequences
        #         best_evaluation = temp_evaluation
        #         initial_evaluation = temp_evaluation
                  
        #     else: cnt+=1
        #     print(best_evaluation)
  				
        # self.best_sequences = best_sequences
        # self.best_evaluation = best_evaluation
        
        
# 按照SA得到的zone的排序来安排seq
def order_by_zone3(cost0):
    zonels=find_zone_center()
    disX=[]
    for i in zonels.keys():
        temp=[]
        lat1=zonels[i][0]
        lng1=zonels[i][1]
        for j in zonels.keys():
            lat2=zonels[j][0]
            lng2=zonels[j][1]
            dis=getDistance(lat1, lng1, lat2, lng2)
            temp.append(dis)
        disX.append(temp)
    disX=pd.DataFrame(disX)
    zones=[]
    for i in zonels.keys():
        zones.append(i)
    
    # 找一条表现较好的zoneseq作为SA或者LS的初始解
    # 1、用order_by_zone2的结果
    # initialSol=[]
    # for i in find_zone_center_seq(order_by_zone2(cost0)).keys():
    #     initialSol.append(zones.index(i))
        
    # 2、greedy
    initialSol=[]
    for i in greedy_zone_seq():
        initialSol.append(zones.index(i))
    
    # 调用SA
    # sa=SA(disX,20,80,100.,0.95,initialSol)
    # sa.solve()
    # best_seq=sa.best_sequences
    
    ##############-------------------
    ls=LS(disX,initialSol)
    ls.solve()
    best_seq=ls.best_sequences
        
    ##############-------------------
    # 调用LP
    # if disX.shape[0]>9:
    #     ls=LS(disX,initialSol)
    #     ls.solve()
    #     best_seq=ls.best_sequences
    # else: 
    #     stop0id=initialSol[0]
    #     dp = DP(disX,stop0id)# stop0
    #     best_seq=dp.main()
    ##############-------------------
    
    zone_seq=[]
    for i in range(len(zones)):
        zone_seq.append(zones[best_seq[i]])
        
    stop0=find_station(stops_route)
    # print(stop0)
    total_seq=[]
    total_seq.append(stop0)
    # total_zone=[]
    cnt=0
    while len(total_seq)<cost0.shape[0]:
        next_zone=zone_seq[cnt]
        stopls=stops_route[stops_route['zone_id']==next_zone].index.to_list()
        # print(stopls)
        
        # if len(stopls)>0:
        # 找出seq末端的stop1
        stop1=total_seq[-1]
        # 找出离stop1最近的在next_zone中的stop
        min_=1000000
        for stop2 in cost0.columns.to_list():
            if cost0.loc[stop1,stop2]<min_ and stops_route.loc[stop2,'zone_id']==next_zone:#
                min_=cost0.loc[stop1,stop2]
                min_stop=stop2
                
        if min_ != 1000000:
            stop0=min_stop

        if len(stopls)>1:
            cost=cost0.loc[stopls,stopls]
            #####1
            # if len(stopls)>7:# 5
            #     # print("使用greedy")
            #     seq=greedy(cost,stop0)
            # else:
            #     # 调用DP
            #     # print("使用DP")
            #     stop0id=cost.index.to_list().index(stop0)
            #     # print("stop0id",stop0id)
            #     S = DP(cost,stop0id)# stop0
            #     seq=S.main()
            
            
            #####2
            temp_cost=cost.values
            seq=greedy(cost,stop0)
            temp_seq=[]
            for i in range(len(seq)):
                temp_seq.append(cost.index.to_list().index(seq[i]))
                
            ##############-------------------
            # # 调用LS
            # ls=LS(temp_cost,temp_seq)
            # ls.solve()
            # temp_seq=ls.best_sequences
            # seq=[]
            # for i in range(len(temp_seq)):
            #     seq.append(cost.index.to_list()[temp_seq[i]])
            
            ##############-------------------
            if len(stopls)>9:
                ls=LS(temp_cost,temp_seq)
                ls.solve()
                temp_seq=ls.best_sequences
                seq=[]
                for i in range(len(temp_seq)):
                    seq.append(cost.index.to_list()[temp_seq[i]])
            else: 
                stop0id=temp_seq[0]
                dp = DP(cost,stop0id)# stop0
                seq=dp.main()
                
            ##############-------------------
            
        if len(stopls)==1:
            seq=[stop0]
            
        if len(stopls)==0:
            seq=[]
            
        # if 'HU' in seq:
            # print(cost)
        # print(seq)
        #把每段的seq累加
        total_seq.extend(seq)
        
        # print(stop1)
        
        # temp=[]
        # for s in seq:
        #     temp.append(stops_route.loc[s,'zone_id'])
        # print(temp)
        
        cnt+=1
        
        
    return total_seq


def write_json(seq_dict,data_travel):
    proposed_sequences={}
    for route, seq in seq_dict.items():
        proposed_sequences[route]={}
        proposed_sequences[route]['proposed']={}
        
        for stopID in data_travel[route].index.to_list():
            if stopID not in seq:  print(route)
            proposed_sequences[route]['proposed'][stopID]=seq.index(stopID)
    
    # with open(r"D:\APP\jupyter\notebook\MIT\xy\scoring\data\model_apply_outputs\proposed_sequences.json","w") as f:
    #     json.dump(proposed_sequences,f)
    
    with open(BASE_DIR + r"/data/model_apply_outputs/proposed_sequences.json","w") as f:
        json.dump(proposed_sequences,f)

#%%
if __name__ == "__main__":
    start=time.time()
    # 导入数据build
    file_name = []
    file_name = []
    file_name.append(BASE_DIR + r"/data/model_apply_inputs/new_route_data.json")
    file_name.append(BASE_DIR + r"/data/model_apply_inputs/new_travel_times.json")
    file_name.append(BASE_DIR + r"/data/model_apply_inputs/new_package_data.json")

    instance = Instance(file_name)
    
#%%
    
        
    # 1. 所有的route
    routels=instance.data_route.index.to_list()
    
    # 2.开始测试
    count_miss=0
    count_miss_after=0
    seq_dict={}
    data_route=instance.data_route
    print('总耗时{:12.4f}s.数据载入和预处理耗时:{:12.4f}s。'.format(time.time()-start,time.time()-start))

    num_to_print = 100
    start_tmp = time.time()
    for i,route in enumerate(routels):
        # 2.1 数据准备
        cost=cost_sum(0.7,0.3,instance.data_travel[route],instance.stops[route])
        data_package_route=instance.data_package[route]
        data_travel_route=instance.data_travel[route]
        stops_route=instance.stops[route]
        stops_route=fullfillzone(stops_route)
        window_stop_info_route=instance.window_stop_info[route]
        
        # 2.2 排序
        # 2.2 (1) greedy
        # seq=greedy(cost,instance.stops[route])
        # 2.2 (2) order_by_zone
        seq=order_by_zone3(cost)

        
        # 2.4 输出最终排序
        seq_dict[route]=seq

        seq_dict[route]=seq
        if i%num_to_print==0:
            avg = (time.time()-start_tmp)/(i+1)
            remain = (len(routels)-i-1)*avg
            print('总耗时{:12.4f}s.已处理第{:4}/{:4},平均每条route耗时:{:12.4f}s.预计剩余{:12.4f}s'.format(time.time()-start,i+1,len(routels),avg,remain))
    print('总耗时{:12.4f}s.平均每条route耗时:{:12.4f}s.'.format(time.time()-start,(time.time()-start)/len(routels)))

    # 3. 写出json
    write_json(seq_dict, instance.data_travel)
    
    # 5. 输出所用时间
    end=time.time()
    print('总耗时{:12.4f}s.'.format(time.time()-start))
