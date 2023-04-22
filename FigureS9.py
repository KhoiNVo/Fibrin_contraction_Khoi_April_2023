# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:48:53 2020

@author: Francesco and Khoi Vo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:57:25 2020

@author: Francesco and Khoi Vo
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
import pickle

from scipy import stats
from scipy.optimize import curve_fit
import os
import re

import numpy as np




def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

data_simple2=load_object("data_standard_with_strain_average2.pickle") #low sim data
data_simple3=load_object("data_low_with_strain_average1.pickle") #high sim data
data_simple1=load_object("data_high_with_strain_average2.pickle")

fiber_segment_vol = 0.3 * np.pi*(0.05)*(0.05)
norm_density_mean1_pcf = np.array([1,4.819716184,7.339398174,8.951472259,9.97336082,11.02843781,11.75590173,11.75043859]) #mean along cols
norm_density_std1_pcf = np.array([0,2.12444694,2.883953096,3.733808182,4.178924841,5.037553633,5.254252243,4.488141697]) #std along cols
time1 = np.array([1.198017,5.990085,11.98017,17.970255,23.96034,29.950425,35.94051,41.930595])



dlengths=np.array([
    len(data_simple1.iat[0,1].time),
    len(data_simple1.iat[0,2].time),
    len(data_simple1.iat[0,3].time),
    len(data_simple1.iat[0,4].time),
    len(data_simple1.iat[0,5].time),
    len(data_simple1.iat[0,6].time),
    len(data_simple1.iat[0,7].time),
    len(data_simple1.iat[0,8].time),
    len(data_simple1.iat[0,9].time),
    len(data_simple1.iat[0,0].time)])
minlength1=dlengths.min(0)

dlengths=np.array([
    len(data_simple2.iat[0,1].time),
    len(data_simple2.iat[0,2].time),
    len(data_simple2.iat[0,3].time),
    len(data_simple2.iat[0,4].time),
    len(data_simple2.iat[0,5].time),
    len(data_simple2.iat[0,6].time),
    len(data_simple2.iat[0,7].time),
    len(data_simple2.iat[0,8].time),
    len(data_simple2.iat[0,9].time),
    len(data_simple2.iat[0,0].time)])
minlength2=dlengths.min(0)


print(data_simple2.iat[0,2].original_edge_strain[0:minlength2])
#print(len(data_simple2.iat[0,1].normalized_fiber_density_sphere[0]))
print(type(data_simple2.iat[0,1].normalized_fiber_density_sphere[0:minlength2]))
print(data_simple2.iat[0,1].normalized_fiber_density_sphere[0:minlength2])
#print(data_simple2.iat[0,2].original_edge_strain[0:minlength2])

temp_mat1_pcf = np.matrix([
data_simple1.iat[0,1].original_edge_strain[0:minlength1],
data_simple1.iat[0,2].original_edge_strain[0:minlength1],
data_simple1.iat[0,3].original_edge_strain[0:minlength1],
data_simple1.iat[0,4].original_edge_strain[0:minlength1],
data_simple1.iat[0,5].original_edge_strain[0:minlength1],
data_simple1.iat[0,6].original_edge_strain[0:minlength1],
data_simple1.iat[0,7].original_edge_strain[0:minlength1],
data_simple1.iat[0,8].original_edge_strain[0:minlength1],
data_simple1.iat[0,9].original_edge_strain[0:minlength1],
data_simple1.iat[0,0].original_edge_strain[0:minlength1]])
norm_density_mean1_pcf = temp_mat1_pcf.mean(0) #mean along cols
print(norm_density_mean1_pcf)
norm_density_std1_pcf = temp_mat1_pcf.std(0) #std along cols
norm_density_sem1_pcf = stats.sem(temp_mat1_pcf) #sem along cols
time1 = data_simple1.iat[0,1].time[0:minlength1]


temp_mat2_pcf = np.matrix([
data_simple2.iat[0,1].original_edge_strain[0:minlength2],
data_simple2.iat[0,2].original_edge_strain[0:minlength2],
data_simple2.iat[0,3].original_edge_strain[0:minlength2],
data_simple2.iat[0,4].original_edge_strain[0:minlength2],
data_simple2.iat[0,5].original_edge_strain[0:minlength2],
data_simple2.iat[0,6].original_edge_strain[0:minlength2],
data_simple2.iat[0,7].original_edge_strain[0:minlength2],
data_simple2.iat[0,8].original_edge_strain[0:minlength2],
data_simple2.iat[0,9].original_edge_strain[0:minlength2],
data_simple2.iat[0,0].original_edge_strain[0:minlength2]])
norm_density_mean2_pcf = temp_mat2_pcf.mean(0) #mean along cols
print(norm_density_mean2_pcf)
norm_density_std2_pcf = temp_mat2_pcf.std(0) #std along cols
norm_density_sem2_pcf = stats.sem(temp_mat2_pcf) #sem along cols
time2 = data_simple2.iat[0,1].time[0:minlength2]

#print(data_simple2.iat[0,1].original_edge_strain[0:minlength2])
#print(data_simple2.iat[0,1].normalized_fiber_plt_density[0:minlength2])
#print(data_simple2.iat[0,1].originalEdgeStrain[0:minlength2])

dlengths=np.array([len(data_simple3.iat[0,0].time[1:]),len(data_simple3.iat[0,1].time[1:]),len(data_simple3.iat[0,2].time[1:]),len(data_simple3.iat[0,3].time[1:]),len(data_simple3.iat[0,4].time[1:]),len(data_simple3.iat[0,5].time[1:]),len(data_simple3.iat[0,6].time[1:]),len(data_simple3.iat[0,7].time[1:]),len(data_simple3.iat[0,8].time[1:]),len(data_simple3.iat[0,9].time[1:])])
minlength3=dlengths.min(0)
temp_mat3_pcf = np.matrix([
data_simple3.iat[0,1].original_edge_strain[0:minlength3],
data_simple3.iat[0,2].original_edge_strain[0:minlength3],
data_simple3.iat[0,3].original_edge_strain[0:minlength3],
data_simple3.iat[0,4].original_edge_strain[0:minlength3],
data_simple3.iat[0,5].original_edge_strain[0:minlength3],
data_simple3.iat[0,6].original_edge_strain[0:minlength3],
data_simple3.iat[0,7].original_edge_strain[0:minlength3],
data_simple3.iat[0,8].original_edge_strain[0:minlength3],
data_simple3.iat[0,9].original_edge_strain[0:minlength3],
data_simple3.iat[0,0].original_edge_strain[0:minlength3]])
norm_density_mean3_pcf = temp_mat3_pcf.mean(0) #mean along cols
norm_density_std3_pcf = temp_mat3_pcf.std(0) #std along cols
norm_density_sem3_pcf = stats.sem(temp_mat3_pcf) #sem along cols
time3 = data_simple3.iat[0,1].time[0:minlength3]




sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

fig = plot.figure(figsize=(16,20))


ax1 = fig.add_subplot(111)
ax1.set_xlim([0,30])
ax1.set_ylim([0,0.1])
#ax2 = fig.add_subplot(142)
#ax2.set_xlim([0,30])
#ax3 = fig.add_subplot(143)
#ax3.set_xlim([0,30])
#ax4 = fig.add_subplot(144)
#ax4.set_xlim([0,30])

color0 = sns.xkcd_rgb["black"]
color1 = sns.xkcd_rgb["blue"]
color2 = sns.xkcd_rgb["green"]
color3 = sns.xkcd_rgb["purple"]
color5 = sns.xkcd_rgb["brown"]
color6 = sns.xkcd_rgb["pink"]
color7 = sns.xkcd_rgb["red"]

colorShade0=sns.xkcd_rgb["almost black"]
colorShade1=sns.xkcd_rgb["baby blue"]
colorShade2=sns.xkcd_rgb["grass green"]
colorShade3=sns.xkcd_rgb["light purple"]
colorShade5=sns.xkcd_rgb["tan"]
colorShade6=sns.xkcd_rgb["light pink"]
colorShade7=sns.xkcd_rgb["light red"]



#bound_upper = norm_density_mean1_pcf.T+norm_density_std1_pcf.T
#bound_lower = norm_density_mean1_pcf.T-norm_density_std1_pcf.T

lineWidth = 2
#ax1.fill_between(np.squeeze(time1-time1[0]), np.squeeze(np.asarray(bound_lower)), np.squeeze(np.asarray(bound_upper)),
#                 color = colorShade1, alpha = 0.15)

#ax1.plot(time1-time1[0], norm_density_mean1_pcf.T,
#         label='Exp. Obs. Filopodia',
#         linestyle='-',
#         linewidth=lineWidth, 
#         color=color1)

bound_upper = norm_density_mean1_pcf.T+norm_density_std1_pcf.T
bound_lower = norm_density_mean1_pcf.T-norm_density_std1_pcf.T

ax1.fill_between(np.squeeze(time1), np.squeeze(np.asarray(bound_lower)), np.squeeze(np.asarray(bound_upper)),
                 color = colorShade1, alpha = 0.15)

ax1.plot(time1, norm_density_mean1_pcf.T,
         label='High Filopodia',
         linestyle='dashed',
         linewidth=lineWidth, 
         color=color1)

bound_upper = norm_density_mean2_pcf.T+norm_density_std2_pcf.T
bound_lower = norm_density_mean2_pcf.T-norm_density_std2_pcf.T

ax1.fill_between(np.squeeze(time2), np.squeeze(np.asarray(bound_lower)), np.squeeze(np.asarray(bound_upper)),
                 color = colorShade2, alpha = 0.15)

ax1.plot(time2, norm_density_mean2_pcf.T,
         label='Standard Filopodia',
         linestyle='dashed',
         linewidth=lineWidth, 
         color=color2)

bound_upper = norm_density_mean3_pcf.T+norm_density_std3_pcf.T
bound_lower = norm_density_mean3_pcf.T-norm_density_std3_pcf.T


ax1.fill_between(np.squeeze(time3), np.squeeze(np.asarray(bound_lower)), np.squeeze(np.asarray(bound_upper)),
                 color = colorShade7, alpha = 0.15)

ax1.plot(time3, norm_density_mean3_pcf.T,
         label='Low Filopodia',
         linestyle=':',
         linewidth=lineWidth, 
         color=color7)




ax1.legend(fontsize=20, loc='upper left')


ax1.set_xlabel('Time (minutes)',fontsize=24)
ax1.set_ylabel('Mean Mean Strain accross all fibrin filaments in the simulated clot',fontsize=24)
ax1.set_ylim(0,0.1)

#For ax2 experiment we need to use time7
#For ax3 experiment we need to use time7der

time7=np.array([0,1.19801700000000,2.39603300000000,3.59405000000000,4.79206700000000,5.99008300000000,7.18810000000000,8.38611700000000,9.58413300000000,10.7821500000000,11.9801700000000,13.1781800000000,14.3762000000000,15.5742200000000,16.7722300000000,17.9702500000000,19.1682700000000,20.3662800000000,21.5643000000000,22.7623200000000,23.9603300000000,25.1583500000000,26.3563700000000,27.5543800000000,28.7524000000000,29.9504200000000,31.1484300000000,32.3464500000000,33.5444700000000,34.7424800000000,35.9405000000000,37.1385200000000,38.3365300000000,39.5345500000000,40.7325700000000,41.9305800000000,43.1286000000000,44.3266200000000,45.5246300000000,46.7226500000000,47.9206700000000,49.1186800000000,50.3167000000000,51.5147200000000,52.71273])
mean7=np.array([1 ,1.025854333 ,1.050884 ,1.077305333 ,1.10938 ,1.143144 ,1.170345667 ,1.196161 ,1.218583333 ,1.234435333 ,1.246152 ,1.260432667 ,1.273055 ,1.284098667 ,1.296228333 ,1.304188 ,1.31134 ,1.320541 ,1.329271667 ,1.340100333 ,1.347512 ,1.353231333 ,1.360315333 ,1.368303333 ,1.381377 ,1.388938 ,1.395523667 ,1.405131 ,1.410820667 ,1.419555333 ,1.424568 ,1.432106 ,1.438552333 ,1.442066 ,1.44833 ,1.453848667 ,1.458839667 ,1.459928333 ,1.465229333 ,1.465916 ,1.470344 ,1.473416 ,1.474653333 ,1.477827,1.478420333])
std7=np.array([0,0.011732514,0.020026857,0.022498318,0.028798289,0.037646998,0.052345234,0.066356318,0.077326602,0.082302146,0.092155973,0.095653523,0.106607462,0.116392158,0.118616492,0.1260202,0.132435315,0.132104641,0.139460522,0.140467204,0.146854914,0.151523609,0.157800664,0.165463667,0.169391299,0.177014175,0.182852303,0.182831861,0.188406006,0.188336411,0.193571207,0.19230779,0.190418275,0.19374179,0.191301147,0.187728583,0.183492135,0.184458814,0.180469167,0.18097173,0.17582598,0.169629533,0.170472408,0.163859623,0.163538493])

time7der=np.array([0.5990085,1.797025,2.9950415,4.1930585,5.391075,6.5890915,7.7871085,8.985125,10.1831415,11.38116,12.579175,13.77719,14.97521,16.173225,17.37124,18.56926,19.767275,20.96529,22.16331,23.361325,24.55934,25.75736,26.955375,28.15339,29.35141,30.549425,31.74744,32.94546,34.143475,35.34149,36.53951,37.737525,38.93554,40.13356,41.331575,42.52959,43.72761,44.925625,46.12364,47.32166,48.519675,49.71769,50.91571,52.113725])
#der=np.array([0.013179936,0.019700715,0.024594427,0.026855134,0.025362,0.024454983,0.021305631,0.017434153,0.014306083,0.012196377,0.010769119,0.00995344,0.009371905,0.008717121,0.007487167,0.007399278,0.007425097,0.007056063,0.007071095,0.006372615,0.006530672,0.006839902,0.006665258,0.007528854,0.00747705,0.007326369,0.006476756,0.005709129,0.006185652,0.005642883,0.005026036,0.005030587,0.004829887,0.004363492,0.003991493,0.003370437,0.003155988,0.002669323,0.002286612,0.002390729,0.002520008,0.001822607,0.001638643,0.00073954])
der=np.array([0.021317515,0.022022859,0.023718817,0.024884795,0.025209715,0.024096622,0.02109923,0.017287489,0.014504943,0.012174089,0.010700239,0.01042631,0.009869303,0.008535729,0.007704008,0.007130497,0.007175157,0.007577902,0.007303775,0.006598177,0.006000241,0.00644552,0.007268988,0.007545137,0.00759738,0.006883893,0.006273553,0.00611931,0.005967439,0.005642498,0.005294769,0.004777269,0.004733878,0.004478573,0.003884137,0.003800342,0.002837951,0.002513456,0.002501282,0.002489708,0.002205759,0.002139549,0.001436844,0.001040424])
stdder=np.array([0.009777095,0.006911952,0.002059551,0.005249975,0.007373924,0.01224853,0.011675904,0.009141904,0.004146286,0.008211523,0.002914625,0.009128283,0.008153913,0.001853611,0.006169756,0.00534593,0.000275562,0.006129901,0.000838901,0.005323092,0.003890579,0.005230879,0.006385836,0.003273026,0.006352397,0.004865107,1.70354E-05,0.004645121,5.79962E-05,0.00436233,0.001052848,0.001574595,0.002769596,0.002033869,0.002977137,0.003530373,0.000805565,0.003324706,0.000418803,0.004288125,0.005163706,0.000702396,0.005510654,0.000267609])





temp_mat2 = np.matrix([
data_simple2.iat[0,1].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,2].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,3].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,4].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,5].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,6].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,7].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,8].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,9].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,0].normalized_fiber_density_sphere[0:minlength2]])
norm_density_mean2 = temp_mat2.mean(0) #mean along cols
norm_density_std2 = temp_mat2.std(0) #std along cols
norm_density_sem2 = stats.sem(temp_mat2) #sem along cols
temp_mat_vel2 = np.matrix([
data_simple2.iat[0,1].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,2].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,3].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,4].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,5].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,6].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,7].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,8].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,9].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,0].change_in_norm_fiber_density_sphere[0:minlength2]])
norm_density_vel_mean2 = temp_mat_vel2.mean(0) #mean along cols
norm_density_vel_std2 = temp_mat_vel2.std(0) #std along cols
norm_density_vel_sem2 = stats.sem(temp_mat_vel2) #sem along cols

temp_mat3 = np.matrix([
data_simple3.iat[0,1].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,2].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,3].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,4].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,5].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,6].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,7].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,8].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,9].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,0].normalized_fiber_density_sphere[0:minlength3]])
norm_density_mean3 = temp_mat3.mean(0) #mean along cols
norm_density_std3 = temp_mat3.std(0) #std along cols
norm_density_sem3 = stats.sem(temp_mat3) #sem along cols
temp_mat_vel3 = np.matrix([
data_simple3.iat[0,1].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,2].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,3].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,4].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,5].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,6].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,7].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,8].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,9].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,0].change_in_norm_fiber_density_sphere[0:minlength3]])
norm_density_vel_mean3 = temp_mat_vel3.mean(0) #mean along cols
norm_density_vel_std3 = temp_mat_vel3.std(0) #std along cols
norm_density_vel_sem3 = stats.sem(temp_mat_vel3) #sem along cols




Fontsize_Sub = 16


ax1.annotate("A", xy=(-0.20, 1.00), xycoords="axes fraction",fontsize=Fontsize_Sub)



ax1.tick_params(labelsize=20)


plot.tight_layout()
plot.savefig("D:/new_codes_April_20/figureS9.png")