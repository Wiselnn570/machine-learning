import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def one_data(data_raw,data_f):
    for i in range(len(data_raw[0])):
        for j in range(len(data_raw)):
            data_f[j][i] = (data_raw[j][i]-min(data_raw[:,i]))/(max(data_raw[:,i])-min(data_raw[:,i]))
    return data_f
def distance(data_f,a):
    pass


file_name = r'C:\Users\ASUS\Desktop\datingTestSet2.txt'
df = pd.read_csv(file_name,sep='\t',header=None,names=None)
data_fr = np.array(df.iloc[:,0:3])
data_l = np.array(df.iloc[:,3])
a = input("请输入一组数据：")
a = np.array(list(map(float,a.split()))).astype(np.float64).reshape(1,3)
data_now = abs(data_fr - a)
data_now = one_data(data_now,np.empty_like(data_now))
dis = np.sum(np.power(data_now,2),axis=1)
idx = dis.argsort()
ans = np.zeros(3)
for i in range(20):
    ans[data_l[idx[i]]-1] += 1
print("对应男性编号为{}".format(np.argmax(ans)+1))





