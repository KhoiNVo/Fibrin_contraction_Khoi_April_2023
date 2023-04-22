# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:19:31 2020

@author: Francesco
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:46:49 2020

@author: Francesco
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:57:26 2020

@author: Francesco
"""

# -*- coding: utf-8 -*-
"""
Created on June 30th 2022 

@author: francesco_pancaldi and Khoi Vo


"""
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
from scipy.spatial import ConvexHull
#from matplotlib.pyplot import probscale


from scipy import stats
from scipy.optimize import curve_fit
import os
import re
import pickle

plot.style.use('seaborn-deep')
from mpl_toolkits.mplot3d import Axes3D



def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
       print("Error during unpickling object (Possibly unsupported):", ex)


data_simple1=load_object("data_standard_plt_volume.pickle")

dlengths=np.array([
    len(data_simple1.iat[0,0].time),
    len(data_simple1.iat[0,1].time),
    len(data_simple1.iat[0,2].time),
    len(data_simple1.iat[0,3].time),
    len(data_simple1.iat[0,4].time),
    len(data_simple1.iat[0,5].time),
    len(data_simple1.iat[0,6].time),
    len(data_simple1.iat[0,7].time),
    len(data_simple1.iat[0,8].time),
    len(data_simple1.iat[0,9].time)])
minlength=dlengths.min(0)

temp_clst = [
data_simple1.iat[0,0].plt_clusters,
data_simple1.iat[0,1].plt_clusters,
data_simple1.iat[0,2].plt_clusters,
data_simple1.iat[0,3].plt_clusters,
data_simple1.iat[0,4].plt_clusters,
data_simple1.iat[0,5].plt_clusters,
data_simple1.iat[0,6].plt_clusters,
data_simple1.iat[0,7].plt_clusters,
data_simple1.iat[0,8].plt_clusters,
data_simple1.iat[0,9].plt_clusters]
sizeclst=[]
for iclst in range(0,minlength):
    sizecluster=[]
    for i in range(0,10):
        print("i: ", i, " iclst: ", iclst)
        clst=temp_clst[i][iclst]
        for cluster in clst:
            sizecluster.append(len(cluster))
    sizeclst.append(sizecluster)



countclst=[] 
for iclst in range(0,minlength):
    countclsttemp=[]
    for i in range(0,10):
        countclstsim=np.zeros(11)
        clst=temp_clst[i][iclst]
        for cluster in clst:
            countclstsim[len(cluster)-1]=countclstsim[len(cluster)-1]+1
        countclsttemp.append(countclstsim)
    countclst.append(countclsttemp)
     

countavg=np.mean(countclst,1)
countstd=np.std(countclst,1)
counttot=np.transpose(np.tile(np.sum(countavg,1), (11, 1)))
countavg=np.mean(countclst,1)/counttot
countstd=np.std(countclst,1)/counttot

singavg=countavg[:,0]
singstd=countstd[:,0]

clustavg=np.sum(countavg[:,1:],1)
cluststd=np.sqrt(np.sum(pow(countstd[:,1:],2),1))

        



bins = np.linspace(0.5, 11.5, 12)
arr=plot.hist([sizeclst[10],sizeclst[20],sizeclst[30],sizeclst[40]], bins)
fig = plot.figure(figsize=(20,16))


ax12 = fig.add_subplot(111) ## 2 tall 1 wide plot number 1 (tall,wide,plot number)
Capsize=0
szclst10 = list(filter(lambda x: x!= 1, sizeclst[10]))
szclst20 = list(filter(lambda x: x!= 1, sizeclst[20]))
szclst30 = list(filter(lambda x: x!= 1, sizeclst[30]))
szclst40 = list(filter(lambda x: x!= 1, sizeclst[40]))
bins = np.linspace(1.5, 6.5, 6)

binsc=np.linspace(2, 11, 10)
countclst10=np.array( countclst[10])[:,1:]
countclst20=np.array( countclst[20])[:,1:]
countclst30=np.array( countclst[30])[:,1:]
countclst40=np.array( countclst[40])[:,1:]


print(np.max(countclst10,0))
prob_number10= np.mean(countclst10,0)/np.sum(np.mean(countclst10,0))
prob_number20= np.mean(countclst20,0)/np.sum(np.mean(countclst20,0))
prob_number30= np.mean(countclst30,0)/np.sum(np.mean(countclst30,0))
prob_number40=np.mean(countclst40,0)/np.sum(np.mean(countclst40,0))


colorShade1=sns.xkcd_rgb["almost black"]
colorShade2=sns.xkcd_rgb["baby blue"]
color1=sns.xkcd_rgb["black"]
color2=sns.xkcd_rgb["blue"]
lineWidth=2

ax12.fill_between(range(0,minlength), clustavg-cluststd, clustavg+cluststd,
                 color = colorShade1, alpha = 0.15)
ax12.plot(range(0,minlength),clustavg,label='Clusters',linestyle='-',linewidth=lineWidth, color=color1)

ax12.fill_between(range(0,minlength), singavg-singstd, singavg+singstd,
                 color = colorShade2, alpha = 0.15)
ax12.plot(range(0,minlength),singavg,label='Single Platelets',linestyle='--',linewidth=lineWidth, color=color2)

ax12.legend(fontsize=20,loc='upper right')

ax12.set_ylabel('Ratios of \n Single vs. Clustered Platelets',fontsize=24)
ax12.set_xlabel('Time (minutes)',fontsize=24)
ax12.tick_params(axis='both', which='major', labelsize=20)


ax12.annotate("A", xy=(-0.05, 1.00), xycoords="axes fraction",fontsize=30)



rclot=30/pow(2,1/3)
# Generate data...
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==1:
            cols.append('red')
        elif l==2:
            cols.append('blue')
        elif l==3:
            cols.append('green')
        elif l==4:
            cols.append('purple')
        elif l==5:
            cols.append('brown')
        else:
            cols.append('black')
    return cols
x = np.array(data_simple1.iat[0,1].time)

Capsize=0
plt_dist_clusters=[]
distclst=[]
for t in range(0,minlength):

    y = np.array(np.concatenate([data_simple1.iat[0,0].plt_dist_clusters[t],
                                 data_simple1.iat[0,1].plt_dist_clusters[t],
                                 data_simple1.iat[0,2].plt_dist_clusters[t],
                                 data_simple1.iat[0,3].plt_dist_clusters[t],
                                 data_simple1.iat[0,4].plt_dist_clusters[t],
                                 data_simple1.iat[0,5].plt_dist_clusters[t],
                                 data_simple1.iat[0,6].plt_dist_clusters[t],
                                 data_simple1.iat[0,7].plt_dist_clusters[t],
                                 data_simple1.iat[0,8].plt_dist_clusters[t],
                                 data_simple1.iat[0,9].plt_dist_clusters[t]]))/rclot
    plt_dist_clusters.append(y)
    
    
    distcluster=[]
    for i in range(0,len(sizeclst[t])):
        cluster=sizeclst[t][i]
        if cluster>1:
                distcluster.append(y[i])
    distclst.append(distcluster)

times=[0,10,20,30]

Capsize=0
temp_plt_dist_clusters = [
    data_simple1.iat[0,0].plt_dist_clusters,
    data_simple1.iat[0,1].plt_dist_clusters,
    data_simple1.iat[0,2].plt_dist_clusters,
    data_simple1.iat[0,3].plt_dist_clusters,
    data_simple1.iat[0,4].plt_dist_clusters,
    data_simple1.iat[0,5].plt_dist_clusters,
    data_simple1.iat[0,6].plt_dist_clusters,
    data_simple1.iat[0,7].plt_dist_clusters,
    data_simple1.iat[0,8].plt_dist_clusters,
    data_simple1.iat[0,9].plt_dist_clusters]

distclst=[]
for iclst in range(0,minlength):
    distcluster=[]
    for i in range(0,10):
        clst=temp_clst[i][iclst]
        dclst=np.array(temp_plt_dist_clusters[i][iclst])/rclot
        for icluster in range(0,len(clst)):
            sz=len(clst[icluster])
            if sz>=2:
                distcluster.append(dclst[icluster])
    distclst.append(distcluster)

countdistclst=[] 
for iclst in range(0,minlength):
    countdistclsttemp=[]
    for i in range(0,10):
        countdistclstsim=np.zeros(10)
        clst=temp_clst[i][iclst]
        dclst=np.array(temp_plt_dist_clusters[i][iclst])/rclot
        for icluster in range(0,len(clst)):
            sz=len(clst[icluster])
            if sz>=2:
                ndist=dclst[icluster]
                idist,ddist=divmod((ndist-1/(2*10))/(1/10),1)
                if idist==-1:
                    print(idist)
                if idist<9:
                    vol=np.pi*(pow((idist+1)*0.1+0.05,2)-pow((idist)*0.1+0.05,2))
                else:
                    vol=np.pi*(pow((idist+1)*0.1,2)-pow((idist)*0.1+0.05,2))
                countdistclstsim[int(idist)]=countdistclstsim[int(idist)]+1/vol
        countdistclsttemp.append(countdistclstsim)
    countdistclst.append(countdistclsttemp)
bins=np.linspace(0.05, 1.05, 11)

binsc=np.linspace(0.1, 1.0, 10)

countdistclst10=np.array( countdistclst[10])
countdistclst20=np.array( countdistclst[20])
countdistclst30=np.array( countdistclst[30])
countdistclst40=np.array( countdistclst[40])

prob_distance10_x= np.mean(countdistclst10,0,dtype=np.float64)/np.sum(np.mean(countdistclst10,0))
prob_distance20_x = np.mean(countdistclst20,0,dtype=np.float64)/np.sum(np.mean(countdistclst20,0))
prob_distance30_x = np.mean(countdistclst30,0,dtype=np.float64)/np.sum(np.mean(countdistclst30,0))
prob_distance40_x = np.mean(countdistclst40,0,dtype=np.float64)/np.sum(np.mean(countdistclst40,0))




plot.tight_layout()

plot.savefig("D:/new_codes_April_20/figure9A_final.png")



