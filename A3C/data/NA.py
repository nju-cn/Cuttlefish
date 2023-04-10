#
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import interpolate
#
# #设置距离
# x =np.array(range(0,200))
# y=np.load("_car.npy")
# #y=np.load("VBJA_car.npy")
# #
#
# #插值法之后的x轴值，表示从0到10间距为0.5的200个数
# xnew =np.arange(0,10,0.1)
#
# #实现函数
# func = interpolate.interp1d(x,y,kind='cubic')
#
# #利用xnew和func函数生成ynew,xnew数量等于ynew数量
# ynew = func(xnew)
#
# # 原始折线
# plt.plot(x, y, "r", linewidth=1)
#
# #平滑处理后曲线
# plt.plot(xnew,ynew)
# #设置x,y轴代表意思
# plt.xlabel("The distance between POI  and user(km)")
# plt.ylabel("probability")
# #设置标题
# plt.title("The content similarity of different distance")
# #设置x,y轴的坐标范围
# plt.xlim(0,200)
# plt.ylim(-40,40)
# plt.show()


# str='It looked like this: I showed my masterpiece to the grown-ups, and asked them whether the drawing frightened them. But they answered:"Frighten? Why should any one be fright by a hat?"'
# print(len(str.split()))
# str=str.replace(':',' ')
# str=str.replace(',',' ')
# str=str.replace('.',' ')
# str=str.replace('?',' ')
# str=str.replace('"',' ')
# print(len(str.split()))

def isprime(num):
    if num<=1:
        return 0
    for i in range (2,int(num**0.5)+1):
        if num%i==0:
            return 0
    return 1

primeIndexs=[]
for i in range(0,1000):
    primeIndexs.append(isprime(i))

for i in range (2,1000):
    if primeIndexs[i]:
        temp=i
        while temp!=0:
            digit=temp%10
            temp//=10
            if primeIndexs[digit]==1:
                if temp==0:
                    print(i)
            else:break



