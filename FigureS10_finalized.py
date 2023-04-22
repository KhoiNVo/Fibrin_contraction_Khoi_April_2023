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
#import parse_high_sim_super as ph
#import parse_low_sim_super as pl


df = pd.read_csv('contraction_data_prp_bleb.csv')
#This df will contain experimental data of standard and tenth

time7=np.array([0,1.19801700000000,2.39603300000000,3.59405000000000,4.79206700000000,5.99008300000000,7.18810000000000,8.38611700000000,9.58413300000000,10.7821500000000,11.9801700000000,13.1781800000000,14.3762000000000,15.5742200000000,16.7722300000000,17.9702500000000,19.1682700000000,20.3662800000000,21.5643000000000,22.7623200000000,23.9603300000000,25.1583500000000,26.3563700000000,27.5543800000000,28.7524000000000,29.9504200000000,31.1484300000000,32.3464500000000,33.5444700000000,34.7424800000000,35.9405000000000,37.1385200000000,38.3365300000000,39.5345500000000,40.7325700000000,41.9305800000000,43.1286000000000,44.3266200000000,45.5246300000000,46.7226500000000,47.9206700000000,49.1186800000000,50.3167000000000,51.5147200000000,52.71273])



def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

data_simple1=load_object("data_standard_plt_volume.pickle")
data_simple2=load_object("data_half_plt_volume2.pickle") #half sim data
data_simple3=load_object("data_tenth_plt_volume.pickle") #tenth sim data




fiber_segment_vol = 0.3 * np.pi*(0.05)*(0.05)
norm_density_mean1_pcf = np.array([1,4.819716184,7.339398174,8.951472259,9.97336082,11.02843781,11.75590173,11.75043859]) #mean along cols
norm_density_std1_pcf = np.array([0,2.12444694,2.883953096,3.733808182,4.178924841,5.037553633,5.254252243,4.488141697]) #std along cols


#experimental data at 30 min
ex_stand=[0.920006,0.950817,1.022251,1.069347,1.052782,0.930129,0.886546,1.073737,1.029241,1.063296]
ex_tenth=[0.013764,0.00874,0.005196,0.020826,0.013239,0.019379]#delete 4,5,9,10

#ex_tenth=[0.013764,0.00874,0.005196,0.125349,0.149618,0.020826,0.013239,0.019379,0.095075,0.088653]

ex_stand_5min=[15.49006,16.11652,29.0947,22.56376,21.43823,14.25319,6.49659,24.25423,16.75091,20.41354]
ex_tenth_5min=[1.12461,1.08191,0.66084,9.06187,4.9328,1.60175,1.4942,2.12541,6.55507,5.11861]

ex_stand_10min=[43.61954,43.56363,58.6518,51.67281,52.03874,40.6595,37.97245,59.017,45.93323,55.60685]
ex_tenth_10min=[1.18645,0.92872,0.57706,1.8335,1.2659,2.03003] #delete 4,5,9,10

#ex_tenth_10min=[1.18645,0.92872,0.57706,12.64376,10.88495,1.8335,1.2659,2.03003,9.65517,9.56287]

ex_stand_15min=[56.18664,57.95477,70.8100,66.39786,66.763,54.28475,52.86195,75.14984,60.43034,71.92895]
ex_tenth_15min=[1.12831,0.85348,0.54149,1.75083,1.17686,1.96574]

#ex_tenth_15min=[1.12831,0.85348,0.54149,12.61917,12.12348,1.75083,1.17686,1.96574,9.65847,9.84145]


ex_stand_20min=[62.78395,64.88967,76.54988,75.05381,74.58947,61.7002,60.84866,81.23964,68.56387,79.44334]
ex_tenth_20min=[1.11033,0.8096,0.51024,12.38632,12.68067,1.70498,1.14357,1.89301,9.49621,9.58443]


#ex_tenth_20min=[1.11033,0.8096,0.51024,12.38632,12.68067,1.70498,1.14357,1.89301,9.49621,9.58443]

#dlengths=np.array([len(data_noscale1.iat[0,0].time[1:]),len(data_noscale2.iat[0,0].time[1:]),len(data_noscale3.iat[0,0].time[1:]),len(data_noscale4.iat[0,0].time[1:]),len(data_noscale5.iat[0,0].time[1:]),len(data_noscale6.iat[0,0].time[1:]),len(data_noscale7.iat[0,0].time[1:]),len(data_noscale8.iat[0,0].time[1:]),len(data_noscale9.iat[0,0].time[1:]),len(data_noscale10.iat[0,0].time[1:])])

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

temp_mat1_pcf = np.matrix([
data_simple1.iat[0,1].normalized_fiber_plt_density[0:minlength1],
data_simple1.iat[0,2].normalized_fiber_plt_density[0:minlength1],
data_simple1.iat[0,3].normalized_fiber_plt_density[0:minlength1],
data_simple1.iat[0,4].normalized_fiber_plt_density[0:minlength1],
data_simple1.iat[0,5].normalized_fiber_plt_density[0:minlength1],
data_simple1.iat[0,6].normalized_fiber_plt_density[0:minlength1],
data_simple1.iat[0,7].normalized_fiber_plt_density[0:minlength1],
data_simple1.iat[0,8].normalized_fiber_plt_density[0:minlength1],
data_simple1.iat[0,9].normalized_fiber_plt_density[0:minlength1],
data_simple1.iat[0,0].normalized_fiber_plt_density[0:minlength1]])
norm_density_mean2_pcf = temp_mat1_pcf.mean(0) #mean along cols
norm_density_std2_pcf = temp_mat1_pcf.std(0) #std along cols
norm_density_sem2_pcf = stats.sem(temp_mat1_pcf) #sem along cols
time1 = data_simple1.iat[0,1].time[1:minlength1]

dlengths=np.array([
    len(data_simple2.iat[0,3].time),
    len(data_simple2.iat[0,5].time),
    len(data_simple2.iat[0,6].time)])

minlength2=dlengths.min(0)

temp_mat2_pcf = np.matrix([
data_simple2.iat[0,3].normalized_fiber_plt_density[0:minlength2],
data_simple2.iat[0,5].normalized_fiber_plt_density[0:minlength2],
data_simple2.iat[0,6].normalized_fiber_plt_density[0:minlength2]])
norm_density_mean2_pcf = temp_mat2_pcf.mean(0) #mean along cols
norm_density_std2_pcf = temp_mat2_pcf.std(0) #std along cols
norm_density_sem2_pcf = stats.sem(temp_mat2_pcf) #sem along cols
time2 = data_simple2.iat[0,3].time[0:minlength2]



dlengths=np.array([len(data_simple3.iat[0,3].time[1:]),len(data_simple3.iat[0,5].time[1:]),len(data_simple3.iat[0,6].time[1:])])
minlength3=dlengths.min(0)
temp_mat3_pcf = np.matrix([
data_simple3.iat[0,3].normalized_fiber_plt_density[0:minlength3],
data_simple3.iat[0,5].normalized_fiber_plt_density[0:minlength3],
data_simple3.iat[0,6].normalized_fiber_plt_density[0:minlength3]])
norm_density_mean3_pcf = temp_mat3_pcf.mean(0) #mean along cols
norm_density_std3_pcf = temp_mat3_pcf.std(0) #std along cols
norm_density_sem3_pcf = stats.sem(temp_mat3_pcf) #sem along cols
time3 = data_simple3.iat[0,3].time[1:minlength3]






sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

fig = plot.figure(figsize=(40,20))

#ax4 is 30 minute
#ax2 is 10 min
#ax3 is 5 min
#ax1 is 15 min
#ax5 is 20 min

#SOrry for the not-in-order number of ax


ax3 = fig.add_subplot(121)
ax3.set_xlim([0,30]) #keep this
ax3.set_ylim([0.00,1.0]) 
ax5 = fig.add_subplot(122)
ax5.set_xlim([0,5])
ax5.set_ylim([0.00,1.1]) #keep this





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






time7=np.array([0,1.19801700000000,2.39603300000000,3.59405000000000,4.79206700000000,5.99008300000000,7.18810000000000,8.38611700000000,9.58413300000000,10.7821500000000,11.9801700000000,13.1781800000000,14.3762000000000,15.5742200000000,16.7722300000000,17.9702500000000,19.1682700000000,20.3662800000000,21.5643000000000,22.7623200000000,23.9603300000000,25.1583500000000,26.3563700000000,27.5543800000000,28.7524000000000,29.9504200000000,31.1484300000000,32.3464500000000,33.5444700000000,34.7424800000000,35.9405000000000,37.1385200000000,38.3365300000000,39.5345500000000,40.7325700000000,41.9305800000000,43.1286000000000,44.3266200000000,45.5246300000000,46.7226500000000,47.9206700000000,49.1186800000000,50.3167000000000,51.5147200000000,52.71273])
mean7=np.array([1 ,1.025854333 ,1.050884 ,1.077305333 ,1.10938 ,1.143144 ,1.170345667 ,1.196161 ,1.218583333 ,1.234435333 ,1.246152 ,1.260432667 ,1.273055 ,1.284098667 ,1.296228333 ,1.304188 ,1.31134 ,1.320541 ,1.329271667 ,1.340100333 ,1.347512 ,1.353231333 ,1.360315333 ,1.368303333 ,1.381377 ,1.388938 ,1.395523667 ,1.405131 ,1.410820667 ,1.419555333 ,1.424568 ,1.432106 ,1.438552333 ,1.442066 ,1.44833 ,1.453848667 ,1.458839667 ,1.459928333 ,1.465229333 ,1.465916 ,1.470344 ,1.473416 ,1.474653333 ,1.477827,1.478420333])
std7=np.array([0,0.011732514,0.020026857,0.022498318,0.028798289,0.037646998,0.052345234,0.066356318,0.077326602,0.082302146,0.092155973,0.095653523,0.106607462,0.116392158,0.118616492,0.1260202,0.132435315,0.132104641,0.139460522,0.140467204,0.146854914,0.151523609,0.157800664,0.165463667,0.169391299,0.177014175,0.182852303,0.182831861,0.188406006,0.188336411,0.193571207,0.19230779,0.190418275,0.19374179,0.191301147,0.187728583,0.183492135,0.184458814,0.180469167,0.18097173,0.17582598,0.169629533,0.170472408,0.163859623,0.163538493])

time7der=np.array([0.5990085,1.797025,2.9950415,4.1930585,5.391075,6.5890915,7.7871085,8.985125,10.1831415,11.38116,12.579175,13.77719,14.97521,16.173225,17.37124,18.56926,19.767275,20.96529,22.16331,23.361325,24.55934,25.75736,26.955375,28.15339,29.35141,30.549425,31.74744,32.94546,34.143475,35.34149,36.53951,37.737525,38.93554,40.13356,41.331575,42.52959,43.72761,44.925625,46.12364,47.32166,48.519675,49.71769,50.91571,52.113725])
#der=np.array([0.013179936,0.019700715,0.024594427,0.026855134,0.025362,0.024454983,0.021305631,0.017434153,0.014306083,0.012196377,0.010769119,0.00995344,0.009371905,0.008717121,0.007487167,0.007399278,0.007425097,0.007056063,0.007071095,0.006372615,0.006530672,0.006839902,0.006665258,0.007528854,0.00747705,0.007326369,0.006476756,0.005709129,0.006185652,0.005642883,0.005026036,0.005030587,0.004829887,0.004363492,0.003991493,0.003370437,0.003155988,0.002669323,0.002286612,0.002390729,0.002520008,0.001822607,0.001638643,0.00073954])
der=np.array([0.021317515,0.022022859,0.023718817,0.024884795,0.025209715,0.024096622,0.02109923,0.017287489,0.014504943,0.012174089,0.010700239,0.01042631,0.009869303,0.008535729,0.007704008,0.007130497,0.007175157,0.007577902,0.007303775,0.006598177,0.006000241,0.00644552,0.007268988,0.007545137,0.00759738,0.006883893,0.006273553,0.00611931,0.005967439,0.005642498,0.005294769,0.004777269,0.004733878,0.004478573,0.003884137,0.003800342,0.002837951,0.002513456,0.002501282,0.002489708,0.002205759,0.002139549,0.001436844,0.001040424])
stdder=np.array([0.009777095,0.006911952,0.002059551,0.005249975,0.007373924,0.01224853,0.011675904,0.009141904,0.004146286,0.008211523,0.002914625,0.009128283,0.008153913,0.001853611,0.006169756,0.00534593,0.000275562,0.006129901,0.000838901,0.005323092,0.003890579,0.005230879,0.006385836,0.003273026,0.006352397,0.004865107,1.70354E-05,0.004645121,5.79962E-05,0.00436233,0.001052848,0.001574595,0.002769596,0.002033869,0.002977137,0.003530373,0.000805565,0.003324706,0.000418803,0.004288125,0.005163706,0.000702396,0.005510654,0.000267609])





temp_mat2 = np.matrix([
data_simple2.iat[0,3].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,5].normalized_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,6].normalized_fiber_density_sphere[0:minlength2]])
norm_density_mean2 = temp_mat2.mean(0) #mean along cols
norm_density_std2 = temp_mat2.std(0) #std along cols
norm_density_sem2 = stats.sem(temp_mat2) #sem along cols

temp_mat_vel2 = np.matrix([
data_simple2.iat[0,3].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,5].change_in_norm_fiber_density_sphere[0:minlength2],
data_simple2.iat[0,6].change_in_norm_fiber_density_sphere[0:minlength2]])
norm_density_vel_mean2 = temp_mat_vel2.mean(0) #mean along cols
norm_density_vel_std2 = temp_mat_vel2.std(0) #std along cols
norm_density_vel_sem2 = stats.sem(temp_mat_vel2) #sem along cols

temp_mat3 = np.matrix([
data_simple3.iat[0,3].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,5].normalized_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,6].normalized_fiber_density_sphere[0:minlength3]])
norm_density_mean3 = temp_mat3.mean(0) #mean along cols
norm_density_std3 = temp_mat3.std(0) #std along cols
norm_density_sem3 = stats.sem(temp_mat3) #sem along cols
temp_mat_vel3 = np.matrix([
data_simple3.iat[0,3].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,5].change_in_norm_fiber_density_sphere[0:minlength3],
data_simple3.iat[0,6].change_in_norm_fiber_density_sphere[0:minlength3]])
norm_density_vel_mean3 = temp_mat_vel3.mean(0) #mean along cols
norm_density_vel_std3 = temp_mat_vel3.std(0) #std along cols
norm_density_vel_sem3 = stats.sem(temp_mat_vel3) #sem along cols


#start experiemental data
temp_ex_standard=np.matrix([np.array(df.Standard_1[1:61])/np.array(df.Standard_1)[61],
                            np.array(df.Standard_2[1:61])/np.array(df.Standard_2)[61],
                            np.array(df.Standard_3[1:61])/np.array(df.Standard_3)[61],
                            np.array(df.Standard_4[1:61])/np.array(df.Standard_4)[61],
                            np.array(df.Standard_5[1:61])/np.array(df.Standard_5)[61],
                            np.array(df.Standard_6[1:61])/np.array(df.Standard_6)[61],
                            np.array(df.Standard_7[1:61])/np.array(df.Standard_7)[61],
                            np.array(df.Standard_8[1:61])/np.array(df.Standard_8)[61],
                            np.array(df.Standard_9[1:61])/np.array(df.Standard_9)[61],
                            np.array(df.Standard_10[1:61])/np.array(df.Standard_10)[61]])
norm_density_ex_standard_mean=temp_ex_standard.mean(0)
norm_density_ex_standard_std=temp_ex_standard.std(0)
time_ex=np.array(df.time[1:61])
time_ex_2=np.array(df.time[1:61])



#this time is for both standard and tenth experiments
norm_1=df.Tenth_1[61]
norm_2=df.Tenth_2[61]
norm_3=df.Tenth_3[61]
norm_4=df.Tenth_4[61]
norm_5=df.Tenth_5[61]
norm_6=df.Tenth_6[61]
norm_7=df.Tenth_7[61]
norm_8=df.Tenth_8[61]
norm_9=df.Tenth_9[61]
norm_10=df.Tenth_10[61]


temp_ex_tenth=np.matrix([np.array(df.Tenth_1[1:61]),
                         np.array(df.Tenth_2[1:61]),
                         np.array(df.Tenth_3[1:61]),
                         np.array(df.Tenth_6[1:61]),
                         np.array(df.Tenth_7[1:61]),
                         np.array(df.Tenth_8[1:61])])


norm_density_ex_tenth_mean=temp_ex_tenth.mean(0)
normalize_const=norm_density_ex_tenth_mean[0,59]
norm_density_ex_tenth_mean_new=temp_ex_tenth.mean(0)/normalize_const

#norm_density_ex_tenth_mean_new_2 is tenth (Bleb) normalized by the final control 
#norm_density_ex_tenth_std_new_2 is corresponding std

norm_density_ex_tenth_mean_new_2=temp_ex_tenth.mean(0)/norm_density_ex_standard_mean[0,59]
norm_density_ex_tenth_std_new_2=temp_ex_tenth.std(0)/norm_density_ex_standard_mean[0,59]


norm_density_ex_tenth_std=temp_ex_tenth.std(0)
std_const=norm_density_ex_tenth_std[0,59]
norm_density_ex_tenth_std_new=norm_density_ex_tenth_std





temp_mat1_chv = np.matrix([
1-np.array(data_simple1.iat[0,1].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,1].fiber_plt_volume[1],
1-np.array(data_simple1.iat[0,2].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,2].fiber_plt_volume[1],
1-np.array(data_simple1.iat[0,3].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,3].fiber_plt_volume[1],
1-np.array(data_simple1.iat[0,4].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,4].fiber_plt_volume[1],
1-np.array(data_simple1.iat[0,5].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,5].fiber_plt_volume[1],
1-np.array(data_simple1.iat[0,6].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,6].fiber_plt_volume[1],
1-np.array(data_simple1.iat[0,7].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,7].fiber_plt_volume[1],
1-np.array(data_simple1.iat[0,8].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,8].fiber_plt_volume[1],
1-np.array(data_simple1.iat[0,9].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,9].fiber_plt_volume[1],
1-np.array(data_simple1.iat[0,0].fiber_plt_volume[1:minlength1])/data_simple1.iat[0,0].fiber_plt_volume[1]])
norm_density_mean10 = temp_mat1_chv.mean(0) #mean along cols
#normalized the data with 30 minute time frame
normalized_standard_mean_const=norm_density_mean10[0,30]
norm_density_mean10_new=norm_density_mean10/normalized_standard_mean_const


norm_density_std10 = temp_mat1_chv.std(0) #std along cols
normalized_standard_std_const=norm_density_std10[0,30]
norm_density_std10_new=norm_density_std10/normalized_standard_std_const



temp_mat2_chv = np.matrix([
1-np.array(data_simple2.iat[0,3].fiber_plt_volume[0:minlength2])/data_simple2.iat[0,3].fiber_plt_volume[0],
1-np.array(data_simple2.iat[0,5].fiber_plt_volume[0:minlength2])/data_simple2.iat[0,5].fiber_plt_volume[0],
1-np.array(data_simple2.iat[0,6].fiber_plt_volume[0:minlength2])/data_simple2.iat[0,6].fiber_plt_volume[0]])
norm_density_mean20 = temp_mat2_chv.mean(0) #mean along cols
norm_density_std20 = temp_mat2_chv.std(0) #std along cols


temp_mat3_chv = np.matrix([
1-np.array(data_simple3.iat[0,3].fiber_plt_volume[1:minlength3])/data_simple3.iat[0,3].fiber_plt_volume[1],
1-np.array(data_simple3.iat[0,5].fiber_plt_volume[1:minlength3])/data_simple3.iat[0,5].fiber_plt_volume[1],
1-np.array(data_simple3.iat[0,6].fiber_plt_volume[1:minlength3])/data_simple3.iat[0,6].fiber_plt_volume[1]])
norm_density_mean30 = temp_mat3_chv.mean(0) #mean along cols
norm_density_std30 = temp_mat3_chv.std(0) #std along cols
#normalize the data now
normalized_tenth_mean_const=norm_density_mean30[0,30]
norm_density_mean30_new=norm_density_mean30/normalized_tenth_mean_const
normalized_tenth_std_const=norm_density_std30[0,30]
norm_density_std30_new=norm_density_std30/normalized_tenth_std_const

#The norm_density_mean30_new_2 is the tenth simulated normalized by the control (hence we use mean20) at final time
#corresponding norm_density_std30_new_2

norm_density_mean30_new_2=norm_density_mean30/norm_density_mean20[0,30]
norm_density_std30_new_2=norm_density_std30/norm_density_std20[0,30]



#start ax3 graph

#start ax3 graph

#this is simulated standard
lineWidth=2

bound_upper = (norm_density_mean10_new.T+norm_density_std10.T)
bound_lower = (norm_density_mean10_new.T-norm_density_std10.T)

ax3.fill_between(np.squeeze(time1), np.squeeze(np.asarray(bound_lower)), np.squeeze(np.asarray(bound_upper)),
                 color = colorShade1, alpha = 0.15)

ax3.plot(time1, norm_density_mean10_new.T,
         label='Modelling',
         linestyle='dashed',
         linewidth=lineWidth, 
         color=color1)

#Try to replace this by the experiemental data of standard
bound_upper = norm_density_ex_standard_mean.T+norm_density_ex_standard_std.T
bound_lower = norm_density_ex_standard_mean.T-norm_density_ex_standard_std.T

ax3.fill_between(np.squeeze(time_ex), np.squeeze(np.asarray(bound_lower)), np.squeeze(np.asarray(bound_upper)),
                 color = colorShade7, alpha = 0.15)

ax3.plot(time_ex, norm_density_ex_standard_mean.T,
         label='Experiment',
         linestyle='dashed',
         linewidth=lineWidth, 
         color=color7)





ax3.legend(fontsize=33, loc='lower right')


Fontsize_Sub = 16


ax3.set_xlabel('Time (minutes)',fontsize=33)
ax3.set_ylabel('Normalized Extent of Clot Contraction',fontsize=33)


#end of drawing ax3






#start to draw ax5
#ax5 is 20 minute
#start tp draw ax2
#ax2 is now 10 min


stand=norm_density_mean10.T[21]
half=norm_density_mean20.T[21]
tenth=norm_density_mean30.T[21]

# Calculate the average
stand_mean=float(stand)/float(stand)
half_mean = float(half)/float(stand)
tenth_mean = float(tenth)/float(stand)



# Calculate the standard deviation
stand_std=float(norm_density_std10.T[21])
half_std = float(norm_density_std20.T[21])
tenth_std = float(norm_density_std30.T[21])

#experimental data
ex_stand_20min_std=stats.sem(ex_stand_20min/np.max(ex_stand_20min))
ex_tenth_20min_std=stats.sem(ex_tenth_20min/np.max(ex_tenth_20min))

ex_stand_20min_mean=np.mean(ex_stand_20min/np.max(ex_stand_20min))/np.mean(ex_stand_20min/np.max(ex_stand_20min))
ex_tenth_20min_mean=np.mean(ex_tenth_20min/np.max(ex_tenth_20min))/np.mean(ex_stand_20min/np.max(ex_stand_20min))
CTEs_ex=[ex_stand_20min_mean,ex_tenth_20min_mean]
error_ex=[ex_stand_20min_std,ex_tenth_20min_std]

# Define labels, positions, bar heights and error bar heights
labels = ['           Control','Blebbistatin','              Control','Blebbistatin']

CTEs = [stand_mean,tenth_mean,0,0]
error = [stand_std,tenth_std,0,0]
x_pos = [0.5,1.2,1.7,2.2]
x_pos_2=[3.0,3.7]
x_pos_3=[0.5,1.5,3,4]

# Build the plot

ax5.bar(x_pos, CTEs,
       yerr=error,
       align='edge',
       alpha=0.3, color=['blue'],
       ecolor='black',
       capsize=10,width=[0.5,0.5,0.5,0.5],label='Modelling')

ax5.bar(x_pos_2, CTEs_ex,
       yerr=error_ex,
       align='edge',
       alpha=0.3, color=['red'],
       ecolor='black',
       capsize=10,width=[0.5,0.5],label='Experiment')

ax5.set_xticks(x_pos_3)
ax5.set_xticklabels(labels)


Fontsize_Sub = 10



ax5.set_ylabel('Normalized Extent of Clot Contraction',fontsize=33)
ax5.legend(fontsize=30,loc='upper right')

#end of ax5


ax3.annotate("A", xy=(0.05, 0.95), xycoords="axes fraction",fontsize=37)
ax5.annotate("B", xy=(0.05, 0.95), xycoords="axes fraction",fontsize=37)


ax3.tick_params(labelsize=33)
ax5.tick_params(labelsize=33)



plot.tight_layout()
plot.savefig("D:/new_codes_April_20/figureS10_finalized.png")
