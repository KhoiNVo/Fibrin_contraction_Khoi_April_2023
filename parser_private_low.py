# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 06:24:36 2021

@author: Francesco
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 07:40:46 2020

@author: samuel_britton
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:20:31 2019 modified on June 30 9:35:35 2019

@author: samuel_britton & francesco_pancaldi
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

#from scipy import stats
#from scipy.optimize import curve_fit
import os
import re

#from PIL import Image
#import matplotlib.image as mpimg

tickSize=25
sns.set_style("ticks", {"xtick.major.size": tickSize, "ytick.major.size": tickSize})

figure_norm = 12 #convert to 17.1cm
figure_len_factor=4/3

figure_factor=17.1/8.3#ratio for sm journal

Fontsize_Title=45
Fontsize_Sub = 40
Fontsize_Leg = 30

dotSize=5
lineWidth=3

colorChoice = sns.xkcd_rgb["black"]
def exp_shift(x, a, b, c, d, e):
     return a * np.exp(b*(x)**e + c) - d
 
def exp(x, a, b, c):
     return a * np.exp(b*(x)) - c

def fun_line(x1,y1,z1,x2,y2,z2,t):
    return [float(x1 + t*(x2-x1)),float(y1 + t*(y2-y1)), float(z1 + t*(z2-z1))];

def poly4(x, a, b, c, d,e):
    return (a*x*x*x*x + b*x*x*x + c * x * x + d * x + e)


def poly3(x, a, b, c, d):
    return (a*x*x*x + b * x * x + c * x + d)

def poly2(x, a, b, c):
    return a * x * x + b * x + c

def alpha_law(x, a, b, c):
    return a * (x**b) + c

      
        
class DataFile:  
    FiberArea = np.pi*(0.05**2)#assumes 75nm radius or 0.05micron radius
    def __init__(self, direct_,file_):
        self.direct = direct_
        self.file=file_
        self.time=0
        self.netWorkStrain=0
        self.totalAppliedForce=0
        self.originalNodeCount=0
        self.nodeCountDiscretize=0
        self.originalEdgeCount=0
        self.areaCoveredByFibers = 0
        self.center_fraction = 0.5
        self.center_sphere_fraction_10 = 0.1
        self.center_sphere_fraction_20 = 0.2
        self.center_sphere_fraction_30 = 0.3
        self.center_sphere_fraction_40 = 0.4
        self.center_sphere_fraction_60 = 0.6
        self.center_sphere_fraction_70 = 0.7
        self.center_sphere_fraction_80 = 0.8
        self.center_sphere_fraction_90 = 0.9
        self.center_sphere_fraction_100 = 1.0
        self.center_sphere_fraction = 0.5
        self.center_vertical_slice = 3.0
        
        self.interior_rect_center_vol = 0.0
        self.interior_sphere_center_vol = 0.0
        self.interior_area_center_vol = 0.0
        
        self.fiber_rect_density=0.0
        self.fiber_sphere_density=0.0
        self.fiber_area_density=0.0
        
        self.minX = 0.0
        self.maxX = 0.0
        self.minY = 0.0
        self.maxY = 0.0
        self.minZ = 0.0
        self.maxZ = 0.0
        
        self.meanX = 0.0
        self.meanY = 0.0
        self.meanZ = 0.0
        
        self.plt_mass=0.486
        self.plt_force=2.1
        self.plt_r=1.13
        self.plt_r_force=2.18
	     
        self.xPos=[]
        self.yPos=[]
        self.zPos=[]   
        self.xPltPos=[]
        self.yPltPos=[]
        self.zPltPos=[]          
        self.originEdgeLeft=[]
        self.originEdgeRight=[]
        self.addedEdgeLeft=[]
        self.addedEdgeRight=[]
        self.originalEdgeStrain=[] 
        self.originalEdgeMidPoint=[]
        self.originalEdgeMidPointScaledForStrain=[]
        self.originalEdgeAlignment=[]
        self.originalNodeForce=[]
        
        
        self.originalEdgeInContactWithPlatelet=[] #holds T/F wrt to edge touching plt
        self.originalNodeInContactWithPlatelet=[] #holds T/F wrt to edge touching plt
        self.originalNodeIsInRectCenter=[]
        self.originalNodeIsInSphereCenter=[]
        self.originalNodeIsInSphereCenter_10=[]
        self.originalNodeIsInSphereCenter_20=[]
        self.originalNodeIsInSphereCenter_30=[]
        self.originalNodeIsInSphereCenter_40=[]
        self.originalNodeIsInSphereCenter_60=[]
        self.originalNodeIsInSphereCenter_70=[]
        self.originalNodeIsInSphereCenter_80=[]
        self.originalNodeIsInSphereCenter_90=[]
        self.originalNodeIsInSphereCenter_100=[]
        self.originalNodeIsInAreaSlice=[]
        self.originalNodeIsInSphereEpiCenter=[]
        self.originalNodeIsInSphereEpiCenter_10=[]
        self.originalNodeIsInSphereEpiCenter_20=[]
        self.originalNodeIsInSphereEpiCenter_30=[]
        self.originalNodeIsInSphereEpiCenter_40=[]
        self.originalNodeIsInSphereEpiCenter_60=[]
        self.originalNodeIsInSphereEpiCenter_70=[]
        self.originalNodeIsInSphereEpiCenter_80=[]
        self.originalNodeIsInSphereEpiCenter_90=[]
        self.originalNodeIsInSphereEpiCenter_100=[]
        
        self.addedEdgeStrain=[]
        self.bindSitesPerNode=[]
        self.curvaturePerEdge=[]
         
        self.curvaturePerEdgeMiddle=[]
        self.curvatureXPointPerEdgeMiddle=[]
        self.curvatureYPointPerEdgeMiddle=[]
        
        self.noDivisionEdgesLeft=[] 
        self.noDivisionEdgesRight=[] 
        self.noDivisionEdgesStrain=[]
        self.noDivisionEdgesForce=[] 
        self.noDivisionEdgesAlignment=[]
        self.noDivisionEdgesBindSites=[]
        self.noDivisionEdgesUniqueBindSites=[]
        self.noDivisionEdgesAverageForce=[]#averaged over all nodes on the edge
        self.noDivisionEdgesMidPoint=[]
        self.noDivisionEdgesMidPointScaledForStrain=[]
        
        self.noDivisionEdgesInContactWithPlatelet=[]
        
        
        
        self.pltContactOtherPlt = [] #for each plt, hold an array of plt's it's touching (self not included)




def parser(DataFile):
    direct = DataFile.direct
    file = DataFile.file
    print(file)
    b1X = 0
    b2X = 0
    b1Y = 0
    b2Y = 0
    with open(os.path.join(direct, file), "r") as f:
        for line in f:
            string_split = line.split()
            for elem in string_split:
                if elem == 'time':
                    DataFile.time = float(string_split[1])
                if elem == 'network_strain':
                    DataFile.netWorkStrain = float(string_split[1])
                if elem == 'total_applied_force':
                    DataFile.totalAppliedForce = float(string_split[1])
                if elem == 'original_node_count':
                    DataFile.originalNodeCount = float(string_split[1])
                if elem == 'node_count_discretize':
                    DataFile.nodeCountDiscretize = float(string_split[1])
                if elem == 'original_edge_count':
                    DataFile.originalEdgeCount = float(string_split[1])
                   
                                
                if (elem == 'maxX ' or elem == 'maxX'):
                    b2X = (float(string_split[1]) )
                    
                if (elem == 'maxY ' or elem == 'maxY'):
                    b2Y = (float(string_split[1]) )
                    
                if (elem == 'minX ' or elem == 'minX'):
                    b1X = (float(string_split[1]) )
                     
                if (elem == 'minY' or elem == 'minY'):
                    b1Y = (float(string_split[1]) )
                    
                    
                if elem == 'node':
                    try:
                        x=float(string_split[1])
                        y=float(string_split[2])
                        z=float(string_split[3])
                        
                        DataFile.xPos.append(x)
                        DataFile.yPos.append(y)
                        DataFile.zPos.append(z)
                        
                    except ValueError:
                        pass  
                    
                if elem == 'plt':
                    try:
                        x=float(string_split[1])
                        y=float(string_split[2])
                        z=float(string_split[3])
                        
                        DataFile.xPltPos.append(x)
                        DataFile.yPltPos.append(y)
                        DataFile.zPltPos.append(z)
                    except ValueError:
                        pass  
                
                if elem == 'force_on_node':
                    force = float(string_split[1])
                    DataFile.originalNodeForce.append(force)
                    
                if elem == 'original_edge_discretized': 
                    eL = float(string_split[1])
                    eR = float(string_split[2])
                    DataFile.originEdgeLeft.append(eL)
                    DataFile.originEdgeRight.append(eR)
                    
                if elem == 'added_edge': 
                    eL = float(string_split[1])
                    eR = float(string_split[2])
                    DataFile.addedEdgeLeft.append(eL)
                    DataFile.addedEdgeRight.append(eR)
                    
                if elem == 'original_edge_strain':
                    DataFile.originalEdgeStrain.append(float(string_split[1]))
                    
                if elem == 'original_edge_alignment':
                    DataFile.originalEdgeAlignment.append(2 * float(string_split[1]) - 1)
                    
                if elem == 'added_edge_strain':
                    DataFile.addedEdgeStrain.append(float(string_split[1]))
                
                if elem == 'bind_sites_per_node':
                    DataFile.bindSitesPerNode.append(float(string_split[1]))
                
        b1 = np.maximum(b1X,b1Y)#min
        b2 = np.minimum(b2X,b2Y)#max       
        #now take original fibers and generate 
                        
        DataFile.maxX = np.max(DataFile.xPos)
        DataFile.minX = np.min(DataFile.xPos)
        DataFile.maxY = np.max(DataFile.yPos)
        DataFile.minY = np.min(DataFile.yPos)
        DataFile.maxZ = np.max(DataFile.zPos)
        DataFile.minZ = np.min(DataFile.zPos)
        
        DataFile.meanX = np.mean(DataFile.xPos)
        DataFile.meanY = np.mean(DataFile.yPos)
        DataFile.meanZ = np.mean(DataFile.zPos)
        
        midX = DataFile.maxX - DataFile.minX
        midY = DataFile.maxY - DataFile.minY
        midZ = DataFile.maxZ - DataFile.minZ
        box_edgeX_left = midX * DataFile.center_fraction
        box_edgeX_right = midX + midX * DataFile.center_fraction
        box_edgeY_left = midY * DataFile.center_fraction
        box_edgeY_right = midY + midY * DataFile.center_fraction
        box_edgeZ_left = midZ * DataFile.center_fraction
        box_edgeZ_right = midZ + midZ * DataFile.center_fraction
        
        DataFile.interior_center_vol = midX * DataFile.center_fraction * midY * DataFile.center_fraction * midZ * DataFile.center_fraction
        DataFile.interior_vol = midX * midY * midZ
        
        for node in range(0,len(DataFile.xPos)):
            x = DataFile.xPos[node]            
            y = DataFile.yPos[node]
            z = DataFile.zPos[node]
            if ((x > box_edgeX_left) and (x < box_edgeX_right) and
                (y > box_edgeY_left) and (y < box_edgeY_right) and
                (z > box_edgeZ_left) and (z < box_edgeZ_right)):
            
                DataFile.originalNodeIsInRectCenter.append(True)
            else:
                DataFile.originalNodeIsInRectCenter.append(False)
        
        trueEdgeLeft = DataFile.originalNodeCount
        trueEdgeRight = DataFile.originalNodeCount
        subEdgeCounter=0
        undividedEdgeCounter=0
        undividedEdgeStrainTemp = 0.0
        undividedEdgeForceTemp = 0.0
        bindSitesPerEdge = 0
        aveAlignment = 0
        aveMidPoint = 0
        aveMidPointScaled = 0
        distcurvetemp=0.0;
        edgeCurveWasInMiddle = False
        xMiddle = 0.0
        yMiddle = 0.0
        
        for node in range(0,len(DataFile.xPos)):
            xPos = DataFile.xPos[node]            
            yPos = DataFile.yPos[node]
            zPos = DataFile.zPos[node]
            
            node_contacted_plt = False
            for plt in range(0,len(DataFile.xPltPos)):
                temp_plt_arr = []
                xPosPlt = DataFile.xPltPos[plt]   
                yPosPlt = DataFile.yPltPos[plt]   
                zPosPlt = DataFile.zPltPos[plt]
                
                dist_to_plt = math.sqrt((xPos-xPosPlt)**2+(yPos-yPosPlt)**2+(zPos-zPosPlt)**2)
                    
                if (dist_to_plt < DataFile.plt_r):
                        edge_contacted_plt = True
                
            if (node_contacted_plt == True):
                DataFile.originalNodeInContactWithPlatelet.append(True);
            else:
                DataFile.originalNodeInContactWithPlatelet.append(False);
                
            
            
        for edge in range(0,len(DataFile.originEdgeLeft)):
            
            edge_contacted_plt = False #each edge, assume not touching a plt
            
            eL = int(DataFile.originEdgeLeft[edge])
            eR = int(DataFile.originEdgeRight[edge])
            if ((eL < DataFile.nodeCountDiscretize) and ( eR < DataFile.nodeCountDiscretize)):
                zPosL = DataFile.zPos[eL]
                zPosR = DataFile.zPos[eR]
                yPosL = DataFile.yPos[eL]
                yPosR = DataFile.yPos[eR]
                xPosL = DataFile.xPos[eL]
                xPosR = DataFile.xPos[eR]
                DataFile.originalEdgeMidPoint.append(( zPosL+ zPosR)/2.0 )  
                
                
                for platelet in range(0,len(DataFile.xPltPos)):   
                    xPosPlt = DataFile.xPltPos[platelet]   
                    yPosPlt = DataFile.yPltPos[platelet]   
                    zPosPlt = DataFile.zPltPos[platelet]
                    distL_to_plt = math.sqrt((xPosL-xPosPlt)**2+(yPosL-yPosPlt)**2+(zPosL-zPosPlt)**2)
                    distR_to_plt = math.sqrt((xPosR-xPosPlt)**2+(yPosR-yPosPlt)**2+(zPosR-zPosPlt)**2)
                    
                    if ((distL_to_plt < DataFile.plt_r) or (distR_to_plt < DataFile.plt_r)):
                        edge_contacted_plt = True
                
                if (edge_contacted_plt == True):
                    DataFile.originalEdgeInContactWithPlatelet.append(True);
                else:
                    DataFile.originalEdgeInContactWithPlatelet.append(False);
                
                        
                        
                    
                #now scale midpoint to strain
                edgeScaled=DataFile.netWorkStrain*(DataFile.originalEdgeMidPoint[edge])/((DataFile.maxZ-DataFile.minZ))
                DataFile.originalEdgeMidPointScaledForStrain.append(edgeScaled)
                subEdgeCounter+=1

                #midpoint slice fiber count
#                midpoint = DataFile.netWorkStrain/2
                
                #add bind sites for all left sides
                bindSitesPerEdge += DataFile.bindSitesPerNode[eR]
                
                #count alignment
                aveAlignment += DataFile.originalEdgeAlignment[edge]

                #add strain for subEdges
                undividedEdgeStrainTemp += DataFile.originalEdgeStrain[edge]
                
                #add force for subEdges
                undividedEdgeForceTemp += (DataFile.originalNodeForce[eR] + 
                                            DataFile.originalNodeForce[eL])
                

                distcurvetemp += math.sqrt((xPosL-xPosR)**2+(yPosL-yPosR)**2+(zPosL-zPosR)**2)
                #average midoint
                aveMidPoint += DataFile.originalEdgeMidPoint[edge]
                aveMidPointScaled += edgeScaled
                #trial left and right
                if (eL < DataFile.originalNodeCount):
                    trueEdgeLeft = eL
                    bindSitesPerEdge += DataFile.bindSitesPerNode[eL]#last get bind sites for edge right

                if (eR < DataFile.originalNodeCount):
                    trueEdgeRight = eR
                    

                if ( (trueEdgeLeft < DataFile.originalNodeCount) & (trueEdgeRight < DataFile.originalNodeCount) ) :
                    
                           
                    zPosL = DataFile.zPos[trueEdgeLeft]
                    zPosR = DataFile.zPos[trueEdgeRight]
                    yPosL = DataFile.yPos[trueEdgeLeft]
                    yPosR = DataFile.yPos[trueEdgeRight]
                    xPosL = DataFile.xPos[trueEdgeLeft]
                    xPosR = DataFile.xPos[trueEdgeRight]
                    length_0 = math.sqrt((xPosL-xPosR)**2+(yPosL-yPosR)**2+(zPosL-zPosR)**2)
                    
                    DataFile.curvaturePerEdge.append(float(length_0/distcurvetemp));
                    
                    if (edgeCurveWasInMiddle==True):           
                        DataFile.curvaturePerEdgeMiddle.append(float(length_0/distcurvetemp))
                        DataFile.curvatureXPointPerEdgeMiddle.append(float(xMiddle/(b2-b1)))
                        DataFile.curvatureYPointPerEdgeMiddle.append(float(yMiddle/(b2-b1)))
                    
                    DataFile.noDivisionEdgesLeft.append(int(trueEdgeLeft))
                    DataFile.noDivisionEdgesRight.append(int(trueEdgeRight))
                    
                    
                    #average alignment
                    DataFile.noDivisionEdgesMidPoint.append(aveMidPoint/subEdgeCounter)
                    DataFile.noDivisionEdgesMidPointScaledForStrain.append(aveMidPointScaled/subEdgeCounter)

                    

                    #now that we have the correct points, get strain
                    DataFile.noDivisionEdgesStrain.append(undividedEdgeStrainTemp/(subEdgeCounter))
                    
                    #average force. Note: edges count 2 node forces, so double edge count
                    DataFile.noDivisionEdgesForce.append(undividedEdgeForceTemp/( 2 * subEdgeCounter))
                                                 
                    #count bind sites for original undivided edge
                    DataFile.noDivisionEdgesBindSites.append(bindSitesPerEdge)
                    
                    #If a single sub edge touched a plt, count the edge
                    if (edge_contacted_plt == True):
                        DataFile.noDivisionEdgesInContactWithPlatelet.append(True);
                    else:
                        DataFile.noDivisionEdgesInContactWithPlatelet.append(False);
                    
                    #average alignment 
                    DataFile.noDivisionEdgesAlignment.append(aveAlignment / subEdgeCounter)
                    #then reset for next edge iteration
                    undividedEdgeCounter += 1
                    trueEdgeLeft = 2*(DataFile.originalNodeCount)
                    trueEdgeRight = 2*(DataFile.originalNodeCount)
                    undividedEdgeStrainTemp = 0.0
                    undividedEdgeForceTemp = 0.0
                    bindSitesPerEdge = 0
                    aveAlignment = 0
                    aveMidPoint = 0
                    aveMidPointScaled = 0
                    subEdgeCounter = 0
                    distcurvetemp=0.0
                    edgeCurveWasInMiddle=False
        #now everything is done, so we rescale the area
#        area=(((b2-b1)*(b2-b1)))
        
        #area = ((DataFile.maxX-DataFile.minX) * (DataFile.maxY-DataFile.minY))
        #DataFile.areaCoveredByFibers = 100*DataFile.numFibersInMiddleSlice * DataFile.FiberArea / area
        minNumOriginal = np.abs(np.min(DataFile.originalEdgeMidPointScaledForStrain))
        minNumNoDivision = np.abs(np.min(DataFile.noDivisionEdgesMidPointScaledForStrain))
        for i in range (len(DataFile.originalEdgeMidPointScaledForStrain)):    
            DataFile.originalEdgeMidPointScaledForStrain[i] += minNumOriginal
        for i in range (len(DataFile.noDivisionEdgesMidPointScaledForStrain)):    
            DataFile.noDivisionEdgesMidPointScaledForStrain[i] += minNumNoDivision
        
        #find unique binding sites
        #for i in range(len(DataFile.noDivisionEdgesLeft)):
        #think of a way to track unique edges that are bound to a given edge.     


#Store files from different simulations in T1-T10 etc
#rows represent different sims, 
#columns represent time difference
def data_generate(data_,rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        colChoice = 0
        rowChoice = 0
        
        files.sort(key=lambda s: int(re.split("\_|\.",s)[1]))
        print((files))
        for file in files:
            #simRow should be t1-t10
            simrow = subdir.split('\\')
            if (len(simrow) > 1):
                simRow=subdir.split('\\')[1]

                #print(simRow)
                if (simRow=='L1'):
                    rowChoice=0
                if (simRow=='L2'):
                    rowChoice=1
                if (simRow=='L3'):
                    rowChoice=2
                if (simRow=='L4'):
                    rowChoice=3
                if (simRow=='L5'):
                    rowChoice=4
                if (simRow=='L6'):
                    rowChoice=5
                if (simRow=='L7'):
                    rowChoice=6
                if (simRow=='L8'):
                    rowChoice=7
                if (simRow=='L9'):
                    rowChoice=8
                if (simRow=='L10'):
                    rowChoice=9
                

                    #now the row,col and file have been chosen. 
                if (rowChoice>=0 and colChoice >=0 and rowChoice < 11):
                       # if ((data_.iat[rowChoice,colChoice]) == DataFile):

                    temp_datafile = DataFile(subdir,file)
                    parser(temp_datafile)
                    data_.iat[rowChoice,colChoice] = temp_datafile
                    print(file)
                    print(rowChoice,colChoice)
                    colChoice+=1;
                    

def parser_short(DataFile):
    direct = DataFile.direct
    file = DataFile.file
    print(file)
#    b1X = 0
#    b2X = 0
#    b1Y = 0
#    b2Y = 0
    with open(os.path.join(direct, file), "r") as f:
        for line in f:
            string_split = line.split()
            for elem in string_split:
                if elem == 'time':
                    DataFile.time = float(string_split[1])
                if elem == 'network_strain':
                    DataFile.netWorkStrain = float(string_split[1])
                if elem == 'total_applied_force':
                    DataFile.totalAppliedForce = float(string_split[1])
                if elem == 'original_node_count':
                    DataFile.originalNodeCount = float(string_split[1])
                if elem == 'node_count_discretize':
                    DataFile.nodeCountDiscretize = float(string_split[1])
                if elem == 'original_edge_count':
                    DataFile.originalEdgeCount = float(string_split[1])
                   
                                
#                if (elem == 'maxX ' or elem == 'maxX'):
#                    b2X = (float(string_split[1]) )
#                    
#                if (elem == 'maxY ' or elem == 'maxY'):
#                    b2Y = (float(string_split[1]) )
#                    
#                if (elem == 'minX ' or elem == 'minX'):
#                    b1X = (float(string_split[1]) )
#                     
#                if (elem == 'minY' or elem == 'minY'):
#                    b1Y = (float(string_split[1]) )
                    
                    
                if elem == 'node':
                    try:
                        x=float(string_split[1])
                        y=float(string_split[2])
                        z=float(string_split[3])
                        
                        DataFile.xPos.append(x)
                        DataFile.yPos.append(y)
                        DataFile.zPos.append(z)
                        
                    except ValueError:
                        pass  
                    
                if elem == 'plt':
                    try:
                        x=float(string_split[1])
                        y=float(string_split[2])
                        z=float(string_split[3])
                        
                        DataFile.xPltPos.append(x)
                        DataFile.yPltPos.append(y)
                        DataFile.zPltPos.append(z)
                    except ValueError:
                        pass
                    
               # if elem == 'force_on_node':
               #     force = float(string_split[1])
               #     DataFile.originalNodeForce.append(force)
                    
                if elem == 'original_edge_discretized': 
                    eL = float(string_split[1])
                    eR = float(string_split[2])
                    DataFile.originEdgeLeft.append(eL)
                    DataFile.originEdgeRight.append(eR)
                    
                #if elem == 'added_edge': 
                #    eL = float(string_split[1])
                #   eR = float(string_split[2])
                #   DataFile.addedEdgeLeft.append(eL)
                #    DataFile.addedEdgeRight.append(eR)
                
        #Calculate box length for internal volume
        DataFile.maxX = np.max(DataFile.xPos)
        DataFile.minX = np.min(DataFile.xPos)
        DataFile.maxY = np.max(DataFile.yPos)
        DataFile.minY = np.min(DataFile.yPos)
        DataFile.maxZ = np.max(DataFile.zPos)
        DataFile.minZ = np.min(DataFile.zPos)
        
        DataFile.meanX = np.mean(DataFile.xPos)
        DataFile.meanY = np.mean(DataFile.yPos)
        DataFile.meanZ = np.mean(DataFile.zPos)
        
        lenX = DataFile.maxX - DataFile.minX
        lenY = DataFile.maxY - DataFile.minY
        lenZ = DataFile.maxZ - DataFile.minZ
        
        midX = (DataFile.maxX + DataFile.minX)/2.0
        midY = (DataFile.maxY + DataFile.minY)/2.0
        midZ = (DataFile.maxZ + DataFile.minZ)/2.0
        box_edgeX_left = midX - lenX * DataFile.center_fraction/2
        box_edgeX_right = midX + lenX * DataFile.center_fraction/2
        box_edgeY_left = midY - lenY * DataFile.center_fraction/2
        box_edgeY_right = midY + lenY * DataFile.center_fraction/2
        box_edgeZ_left = midZ - lenZ * DataFile.center_fraction/2
        box_edgeZ_right = midZ + lenZ * DataFile.center_fraction/2
        
        #Calculate sphere length for internal volume
        radius = lenX * DataFile.center_sphere_fraction
        volume = (4.0/3.0) * (np.pi) * (radius**3.0)
        radius_10 = lenX * DataFile.center_sphere_fraction_10
        volume_10 = (4.0/3.0) * (np.pi) * (radius_10**3.0)
        radius_20 = lenX * DataFile.center_sphere_fraction_20
        volume_20 = (4.0/3.0) * (np.pi) * (radius_20**3.0)
        radius_30 = lenX * DataFile.center_sphere_fraction_30
        volume_30 = (4.0/3.0) * (np.pi) * (radius_30**3.0)
        radius_40 = lenX * DataFile.center_sphere_fraction_40
        volume_40 = (4.0/3.0) * (np.pi) * (radius_40**3.0)
        radius_60 = lenX * DataFile.center_sphere_fraction_60
        volume_60 = (4.0/3.0) * (np.pi) * (radius_60**3.0)
        radius_70 = lenX * DataFile.center_sphere_fraction_70
        volume_70 = (4.0/3.0) * (np.pi) * (radius_70**3.0)
        radius_80 = lenX * DataFile.center_sphere_fraction_80
        volume_80 = (4.0/3.0) * (np.pi) * (radius_80**3.0)
        radius_90 = lenX * DataFile.center_sphere_fraction_90
        volume_90 = (4.0/3.0) * (np.pi) * (radius_90**3.0)
        radius_100 = lenX * DataFile.center_sphere_fraction_100
        volume_100 = (4.0/3.0) * (np.pi) * (radius_100**3.0)
        
        
        print("volume:")
        print(volume)
        DataFile.interior_rect_center_vol = 2*lenX + 2*lenY + 2*lenZ
        DataFile.interior_sphere_center_vol = volume
        DataFile.interior_sphere_center_vol_10 = volume_10
        DataFile.interior_sphere_center_vol_20 = volume_20
        DataFile.interior_sphere_center_vol_30 = volume_30
        DataFile.interior_sphere_center_vol_40 = volume_40
        DataFile.interior_sphere_center_vol_60 = volume_60
        DataFile.interior_sphere_center_vol_70 = volume_70
        DataFile.interior_sphere_center_vol_80 = volume_80
        DataFile.interior_sphere_center_vol_90 = volume_90
        DataFile.interior_sphere_center_vol_100 = volume_100
        DataFile.interior_area_center_vol = 2*lenX + 2*lenY + 2 * DataFile.center_vertical_slice
        
                
        DataFile.interior_vol = midX * midY * midZ
        for node in range(0,len(DataFile.xPos)):
            x = DataFile.xPos[node]            
            y = DataFile.yPos[node]
            z = DataFile.zPos[node]
            
            rad_i = np.sqrt(x**2 + y**2 + z**2)
            radepi_i = np.sqrt((x-DataFile.meanX)**2 + (y-DataFile.meanY)**2 + (z-DataFile.meanZ)**2)
                    
            if ((x > box_edgeX_left) and (x < box_edgeX_right) and
                (y > box_edgeY_left) and (y < box_edgeY_right) and
                (z > box_edgeZ_left) and (z < box_edgeZ_right)):
                    
                DataFile.originalNodeIsInRectCenter.append(True)
            else:
                DataFile.originalNodeIsInRectCenter.append(False)
                
            if (rad_i < radius):
                DataFile.originalNodeIsInSphereCenter.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter.append(False)
                
            if (rad_i < radius_10):
                DataFile.originalNodeIsInSphereCenter_10.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter_10.append(False)
                
            if (rad_i < radius_20):
                DataFile.originalNodeIsInSphereCenter_20.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter_20.append(False)
                
            if (rad_i < radius_30):
                DataFile.originalNodeIsInSphereCenter_30.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter_30.append(False)
                
            if (rad_i < radius_40):
                DataFile.originalNodeIsInSphereCenter_40.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter_40.append(False)
                
            if (rad_i < radius_60):
                DataFile.originalNodeIsInSphereCenter_60.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter_60.append(False)
                  
            if (rad_i < radius_70):
                DataFile.originalNodeIsInSphereCenter_70.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter_70.append(False)
                
            if (rad_i < radius_80):
                DataFile.originalNodeIsInSphereCenter_80.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter_80.append(False)
                
            if (rad_i < radius_90):
                DataFile.originalNodeIsInSphereCenter_90.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter_90.append(False)
                
            if (rad_i < radius_100):
                DataFile.originalNodeIsInSphereCenter_100.append(True)
            else:
                DataFile.originalNodeIsInSphereCenter_100.append(False)
                
                
            if (radepi_i < radius):
                DataFile.originalNodeIsInSphereEpiCenter.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter.append(False)
                
            if (radepi_i < radius_10):
                DataFile.originalNodeIsInSphereEpiCenter_10.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter_10.append(False)
                
            if (radepi_i < radius_20):
                DataFile.originalNodeIsInSphereEpiCenter_20.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter_20.append(False)
                
            if (radepi_i < radius_30):
                DataFile.originalNodeIsInSphereEpiCenter_30.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter_30.append(False)
                
            if (radepi_i < radius_40):
                DataFile.originalNodeIsInSphereEpiCenter_40.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter_40.append(False)
                
            if (radepi_i < radius_60):
                DataFile.originalNodeIsInSphereEpiCenter_60.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter_60.append(False)
                  
            if (radepi_i < radius_70):
                DataFile.originalNodeIsInSphereEpiCenter_70.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter_70.append(False)
                
            if (radepi_i < radius_80):
                DataFile.originalNodeIsInSphereEpiCenter_80.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter_80.append(False)
                
            if (radepi_i < radius_90):
                DataFile.originalNodeIsInSphereEpiCenter_90.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter_90.append(False)
                
            if (radepi_i < radius_100):
                DataFile.originalNodeIsInSphereEpiCenter_100.append(True)
            else:
                DataFile.originalNodeIsInSphereEpiCenter_100.append(False)
                  
                
            if ((x > box_edgeX_left) and (x < box_edgeX_right) and
                (y > box_edgeY_left) and (y < box_edgeY_right) and
                (z < midZ + DataFile.center_vertical_slice) and 
                (z > midZ - DataFile.center_vertical_slice)):
                
                DataFile.originalNodeIsInAreaSlice.append(True)
            else:
                DataFile.originalNodeIsInAreaSlice.append(False)
                
        DataFile.fiber_rect_density=sum(DataFile.originalNodeIsInRectCenter)/DataFile.interior_rect_center_vol
        DataFile.fiber_sphere_density=sum(DataFile.originalNodeIsInSphereCenter)/DataFile.interior_sphere_center_vol
        DataFile.fiber_sphere_density_10=sum(DataFile.originalNodeIsInSphereCenter_10)/DataFile.interior_sphere_center_vol_10
        DataFile.fiber_sphere_density_20=sum(DataFile.originalNodeIsInSphereCenter_20)/DataFile.interior_sphere_center_vol_20
        DataFile.fiber_sphere_density_30=sum(DataFile.originalNodeIsInSphereCenter_30)/DataFile.interior_sphere_center_vol_30
        DataFile.fiber_sphere_density_40=sum(DataFile.originalNodeIsInSphereCenter_40)/DataFile.interior_sphere_center_vol_40
        DataFile.fiber_sphere_density_60=sum(DataFile.originalNodeIsInSphereCenter_60)/DataFile.interior_sphere_center_vol_60
        DataFile.fiber_sphere_density_70=sum(DataFile.originalNodeIsInSphereCenter_70)/DataFile.interior_sphere_center_vol_70
        DataFile.fiber_sphere_density_80=sum(DataFile.originalNodeIsInSphereCenter_80)/DataFile.interior_sphere_center_vol_80
        DataFile.fiber_sphere_density_90=sum(DataFile.originalNodeIsInSphereCenter_90)/DataFile.interior_sphere_center_vol_90
        DataFile.fiber_sphere_density_100=sum(DataFile.originalNodeIsInSphereCenter_100)/DataFile.interior_sphere_center_vol_100
        DataFile.fiber_area_density=sum(DataFile.originalNodeIsInAreaSlice)/DataFile.interior_area_center_vol
        
        DataFile.fiber_episphere_density=sum(DataFile.originalNodeIsInSphereEpiCenter)/DataFile.interior_sphere_center_vol
        DataFile.fiber_episphere_density_10=sum(DataFile.originalNodeIsInSphereEpiCenter_10)/DataFile.interior_sphere_center_vol_10
        DataFile.fiber_episphere_density_20=sum(DataFile.originalNodeIsInSphereEpiCenter_20)/DataFile.interior_sphere_center_vol_20
        DataFile.fiber_episphere_density_30=sum(DataFile.originalNodeIsInSphereEpiCenter_30)/DataFile.interior_sphere_center_vol_30
        DataFile.fiber_episphere_density_40=sum(DataFile.originalNodeIsInSphereEpiCenter_40)/DataFile.interior_sphere_center_vol_40
        DataFile.fiber_episphere_density_60=sum(DataFile.originalNodeIsInSphereEpiCenter_60)/DataFile.interior_sphere_center_vol_60
        DataFile.fiber_episphere_density_70=sum(DataFile.originalNodeIsInSphereEpiCenter_70)/DataFile.interior_sphere_center_vol_70
        DataFile.fiber_episphere_density_80=sum(DataFile.originalNodeIsInSphereEpiCenter_80)/DataFile.interior_sphere_center_vol_80
        DataFile.fiber_episphere_density_90=sum(DataFile.originalNodeIsInSphereEpiCenter_90)/DataFile.interior_sphere_center_vol_90
        DataFile.fiber_episphere_density_100=sum(DataFile.originalNodeIsInSphereEpiCenter_100)/DataFile.interior_sphere_center_vol_100
       
        
        
        for platelet in range(0,len(DataFile.xPltPos)):
            temp_plt_arr = []
            xPosPlt = DataFile.xPltPos[platelet]   
            yPosPlt = DataFile.yPltPos[platelet]   
            zPosPlt = DataFile.zPltPos[platelet]
            
            for plt_other in range(0, len(DataFile.xPltPos)):
                if (plt_other != platelet):
                    xPosPltAlt = DataFile.xPltPos[plt_other]   
                    yPosPltAlt = DataFile.yPltPos[plt_other]   
                    zPosPltAlt = DataFile.zPltPos[plt_other]
                    dist = math.sqrt((xPosPltAlt-xPosPlt)**2+(yPosPltAlt-yPosPlt)**2+(zPosPltAlt-zPosPlt)**2)
                    
                    if (dist < 3.0): 
                        
                        temp_plt_arr.append(plt_other);
            
            
            DataFile.pltContactOtherPlt.append(temp_plt_arr);
            
        trueEdgeLeft = 2*(DataFile.originalNodeCount)
        trueEdgeRight = 2*(DataFile.originalNodeCount)
		
        for node in range(0,len(DataFile.xPos)):
            xPos = DataFile.xPos[node]            
            yPos = DataFile.yPos[node]
            zPos = DataFile.zPos[node]
		    
            node_contacted_plt=False
            for platelet in range(0,len(DataFile.xPltPos)):   
                xPosPlt = DataFile.xPltPos[platelet]   
                yPosPlt = DataFile.yPltPos[platelet]   
                zPosPlt = DataFile.zPltPos[platelet]
                dist_to_plt = math.sqrt((xPos-xPosPlt)**2+(yPos-yPosPlt)**2+(zPos-zPosPlt)**2)
                if (dist_to_plt < DataFile.plt_r):
                    node_contacted_plt = True
                
            if (node_contacted_plt == True):
                DataFile.originalNodeInContactWithPlatelet.append(True);
            else:
                DataFile.originalNodeInContactWithPlatelet.append(False);
                
        for edge in range(0,len(DataFile.originEdgeLeft)):
            
            edge_contacted_plt = False #each edge, assume not touching a plt
            
            eL = int(DataFile.originEdgeLeft[edge])
            eR = int(DataFile.originEdgeRight[edge])
            if ((eL < DataFile.nodeCountDiscretize) and ( eR < DataFile.nodeCountDiscretize)):
                zPosL = DataFile.zPos[eL]
                zPosR = DataFile.zPos[eR]
                yPosL = DataFile.yPos[eL]
                yPosR = DataFile.yPos[eR]
                xPosL = DataFile.xPos[eL]
                xPosR = DataFile.xPos[eR]
                DataFile.originalEdgeMidPoint.append(( zPosL+ zPosR)/2.0 )  
                
                
                for platelet in range(0,len(DataFile.xPltPos)):   
                    xPosPlt = DataFile.xPltPos[platelet]   
                    yPosPlt = DataFile.yPltPos[platelet]   
                    zPosPlt = DataFile.zPltPos[platelet]
                    distL_to_plt = math.sqrt((xPosL-xPosPlt)**2+(yPosL-yPosPlt)**2+(zPosL-zPosPlt)**2)
                    distR_to_plt = math.sqrt((xPosR-xPosPlt)**2+(yPosR-yPosPlt)**2+(zPosR-zPosPlt)**2)
                    
                    if ((distL_to_plt < DataFile.plt_r) or (distR_to_plt < DataFile.plt_r)):
                        edge_contacted_plt = True
                
                if (edge_contacted_plt == True):
                    DataFile.originalEdgeInContactWithPlatelet.append(True);
                else:
                    DataFile.originalEdgeInContactWithPlatelet.append(False);
                
                #trial left and right
                if (eL < DataFile.originalNodeCount):
                    trueEdgeLeft = eL

                if (eR < DataFile.originalNodeCount):
                    trueEdgeRight = eR
                    

                if ( (trueEdgeLeft < DataFile.originalNodeCount) and (trueEdgeRight < DataFile.originalNodeCount) ) :
                    
                    #If a single sub edge touched a plt, count the edge
                    if (edge_contacted_plt == True):
                        DataFile.noDivisionEdgesInContactWithPlatelet.append(True);
                    else:
                        DataFile.noDivisionEdgesInContactWithPlatelet.append(False);
                    
                    trueEdgeLeft = 2*(DataFile.originalNodeCount)
                    trueEdgeRight = 2*(DataFile.originalNodeCount)

            
#In this file, I only want to calculate basic averages
class SimpleDataFile:  
    FiberArea = np.pi*(0.05**2)#assumes 75nm radius or 0.05micron radius
    fiber_segment_vol = 0.3 * np.pi*(0.05)*(0.05)
    def __init__(self):
        self.time=[]
        self.average_fiber_density_rect=[]
        self.average_fiber_density_sphere=[]
        self.average_fiber_density_sphere_10=[]
        self.average_fiber_density_sphere_20=[]
        self.average_fiber_density_sphere_30=[]
        self.average_fiber_density_sphere_40=[]
        self.average_fiber_density_sphere_60=[]
        self.average_fiber_density_sphere_70=[]
        self.average_fiber_density_sphere_80=[]
        self.average_fiber_density_sphere_90=[]
        self.average_fiber_density_sphere_100=[]
        self.average_fiber_density_episphere=[]
        self.average_fiber_density_episphere_10=[]
        self.average_fiber_density_episphere_20=[]
        self.average_fiber_density_episphere_30=[]
        self.average_fiber_density_episphere_40=[]
        self.average_fiber_density_episphere_60=[]
        self.average_fiber_density_episphere_70=[]
        self.average_fiber_density_episphere_80=[]
        self.average_fiber_density_episphere_90=[]
        self.average_fiber_density_episphere_100=[]
        self.average_fiber_density_area=[]
        self.average_fiber_conn_plt=[]
        self.average_plt_speed=[]
        
        self.change_in_fiber_density_rect=[]
        self.change_in_fiber_density_sphere=[]
        self.change_in_fiber_density_area=[]
        
        self.normalized_fiber_density_rect=[]
        self.normalized_fiber_density_sphere=[]
        self.normalized_fiber_density_sphere_10=[]
        self.normalized_fiber_density_sphere_20=[]
        self.normalized_fiber_density_sphere_30=[]
        self.normalized_fiber_density_sphere_40=[]
        self.normalized_fiber_density_sphere_60=[]
        self.normalized_fiber_density_sphere_70=[]
        self.normalized_fiber_density_sphere_80=[]
        self.normalized_fiber_density_sphere_90=[]
        self.normalized_fiber_density_sphere_100=[]
        self.normalized_fiber_density_episphere=[]
        self.normalized_fiber_density_episphere_10=[]
        self.normalized_fiber_density_episphere_20=[]
        self.normalized_fiber_density_episphere_30=[]
        self.normalized_fiber_density_episphere_40=[]
        self.normalized_fiber_density_episphere_60=[]
        self.normalized_fiber_density_episphere_70=[]
        self.normalized_fiber_density_episphere_80=[]
        self.normalized_fiber_density_episphere_90=[]
        self.normalized_fiber_density_episphere_100=[]
        self.normalized_fiber_density_area=[]
        self.change_in_norm_fiber_density_rect=[]
        self.change_in_norm_fiber_density_sphere=[]
        self.change_in_norm_fiber_density_area=[]
        
        self.normalized_fiber_plt_density=[]
        
        self.individual_plt_dist_from_center=[]
        self.individual_plt_dist_from_epicenter=[]
        self.individual_plt_speed=[]
        self.individual_plt_dist=[]
        self.plt_clusters=[]
        self.xplt_clusters=[]
        self.yplt_clusters=[]
        self.zplt_clusters=[]
        self.plt_dist_clusters=[]
        self.fiber_convex_hull=[]
        self.fiber_convex_hull_volume=[]
        

#This generates a short version of mutiple parameter files
def data_generate_short(data_,rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        rowChoice=0
        
        files.sort(key=lambda s: int(re.split("\_|\.",s)[1]))
        print((files))
        #simRow should be t1-t10
        simrow = subdir.split('\\')
        if (len(simrow) > 1):
            simRow=subdir.split('\\')[1]

            #print(simRow)
            if (simRow=='L1'):
                rowChoice=0
            if (simRow=='L2'):
                rowChoice=1
            if (simRow=='L3'):
                rowChoice=2
            if (simRow=='L4'):
                rowChoice=3
            if (simRow=='L5'):
                rowChoice=4
            if (simRow=='L6'):
                rowChoice=5
            if (simRow=='L7'):
                rowChoice=6
            if (simRow=='L8'):
                rowChoice=7
            if (simRow=='L9'):
                rowChoice=8
            if (simRow=='L10'):
                rowChoice=9
                
            #for each file calculate averages and store them .
            #fill temp_datafile with only some values. 
            simpleData = SimpleDataFile();
            file_counter=0
 

                 
            first_midX = 0
            first_midY = 0
            first_midZ = 0
            first_box_edgeX_left = 0
            first_box_edgeX_right = 0
            first_box_edgeY_left = 0 
            first_box_edgeY_right = 0
            first_box_edgeZ_left = 0
            first_box_edgeZ_right = 0
            first_interior_rect_center_vol = 0
            first_interior_sphere_center_vol = 0
            first_interior_sphere_center_vol_10 = 0
            first_interior_sphere_center_vol_20 = 0
            first_interior_sphere_center_vol_30 = 0
            first_interior_sphere_center_vol_40 = 0
            first_interior_sphere_center_vol_60 = 0
            first_interior_sphere_center_vol_70 = 0
            first_interior_sphere_center_vol_80 = 0
            first_interior_sphere_center_vol_90 = 0
            first_interior_sphere_center_vol_100 = 0
            first_interior_area_center_vol = 0
            
            first_radius = 0
            
            for file in files:
            #This is a loop for every time step
                
                #print(file)
                temp_datafile = DataFile(subdir,file)
                
                parser_short(temp_datafile)#simple parser
                
                if (file_counter == 0):
                            
                    first_lenX = temp_datafile.maxX - temp_datafile.minX
                    first_lenY = temp_datafile.maxY - temp_datafile.minY
                    first_lenZ = temp_datafile.maxZ - temp_datafile.minZ
                    
                    first_midX = (temp_datafile.maxX + temp_datafile.minX)/2.0
                    first_midY = (temp_datafile.maxY + temp_datafile.minY)/2.0
                    first_midZ = (temp_datafile.maxZ + temp_datafile.minZ)/2.0
                    first_box_edgeX_left = first_midX - first_lenX * temp_datafile.center_fraction/2
                    first_box_edgeX_right = first_midX + first_lenX * temp_datafile.center_fraction/2
                    first_box_edgeY_left = first_midY - first_lenY * temp_datafile.center_fraction/2
                    first_box_edgeY_right = first_midY + first_lenY * temp_datafile.center_fraction/2
                    first_box_edgeZ_left = first_midZ - first_lenZ * temp_datafile.center_fraction/2
                    first_box_edgeZ_right = first_midZ + first_lenZ * temp_datafile.center_fraction/2
                
                    first_radius = first_lenX * temp_datafile.center_sphere_fraction
                    first_volume = (4.0/3.0) * (np.pi) * (first_radius**3.0)
                    first_radius_10 = first_lenX * temp_datafile.center_sphere_fraction_10
                    first_volume_10 = (4.0/3.0) * (np.pi) * (first_radius_10**3.0)
                    first_radius_20 = first_lenX * temp_datafile.center_sphere_fraction_20
                    first_volume_20 = (4.0/3.0) * (np.pi) * (first_radius_20**3.0)
                    first_radius_30 = first_lenX * temp_datafile.center_sphere_fraction_30
                    first_volume_30 = (4.0/3.0) * (np.pi) * (first_radius_30**3.0)
                    first_radius_40 = first_lenX * temp_datafile.center_sphere_fraction_40
                    first_volume_40 = (4.0/3.0) * (np.pi) * (first_radius_40**3.0)
                    first_radius_60 = first_lenX * temp_datafile.center_sphere_fraction_60
                    first_volume_60 = (4.0/3.0) * (np.pi) * (first_radius_60**3.0)
                    first_radius_70 = first_lenX * temp_datafile.center_sphere_fraction_70
                    first_volume_70 = (4.0/3.0) * (np.pi) * (first_radius_70**3.0)
                    first_radius_80 = first_lenX * temp_datafile.center_sphere_fraction_80
                    first_volume_80 = (4.0/3.0) * (np.pi) * (first_radius_80**3.0)
                    first_radius_90 = first_lenX * temp_datafile.center_sphere_fraction_90
                    first_volume_90 = (4.0/3.0) * (np.pi) * (first_radius_90**3.0)
                    first_radius_100 = first_lenX * temp_datafile.center_sphere_fraction_100
                    first_volume_100 = (4.0/3.0) * (np.pi) * (first_radius_100**3.0)
                         
                    first_interior_rect_center_vol = 2 * first_lenX + 2 * first_lenY + 2 * first_lenZ
                    first_interior_sphere_center_vol = first_volume
                    first_interior_sphere_center_vol_10 = first_volume_10
                    first_interior_sphere_center_vol_20 = first_volume_20
                    first_interior_sphere_center_vol_30 = first_volume_30
                    first_interior_sphere_center_vol_40 = first_volume_40
                    first_interior_sphere_center_vol_60 = first_volume_60
                    first_interior_sphere_center_vol_70 = first_volume_70
                    first_interior_sphere_center_vol_80 = first_volume_80
                    first_interior_sphere_center_vol_90 = first_volume_90
                    first_interior_sphere_center_vol_100 = first_volume_100
                    first_interior_area_center_vol = 2*first_lenX + 2*first_lenY + 2 * temp_datafile.center_vertical_slice
                    
                #TIME
                simpleData.time.append(temp_datafile.time/60)
                
                #CLUSTER
                plt_r=1.13
                xPltPos=np.array(temp_datafile.xPltPos)
                yPltPos=np.array(temp_datafile.yPltPos)
                zPltPos=np.array(temp_datafile.zPltPos)
                toll=plt_r*5/100
                
                def get_nbrs(index,nbrs):
                    
                    xPosPlt1 = xPltPos[index]   
                    yPosPlt1 = yPltPos[index]   
                    zPosPlt1 = zPltPos[index]
                    for plt in range(0,len(xPltPos)):
                        
                        xPosPlt2 = xPltPos[plt]   
                        yPosPlt2 = yPltPos[plt]   
                        zPosPlt2 = zPltPos[plt]
                        dist_plt = math.sqrt((xPosPlt1-xPosPlt2)**2+(yPosPlt1-yPosPlt2)**2+(zPosPlt1-zPosPlt2)**2)
                                    
                        if (dist_plt < 2*(plt_r+toll)):
                            #print(plt)
                            nbrs.append(plt)
                    
                
                def generate_clusters(cluster, true_clusters_input):
                    
                    nbr_lists=[]
                    for item in cluster:
                        nbr_list=[]
                        get_nbrs(item,nbr_list)
                        nbr_lists=list(set(nbr_lists + nbr_list))
                
                    elt_was_added=False
                    for elt in nbr_lists:
                        if (elt not in true_clusters_input):
                            true_clusters_input.append(elt)
                            elt_was_added=True
                    
                    #if new nbrs were added, generate new clusters around them. 
                    if (elt_was_added):
                        #note, old nbrs from previous clusters are removed. 
                        for nbr in cluster:
                            nbr_lists.remove(nbr)    
                        generate_clusters(nbr_lists,true_clusters_input)
                            
                #find initial neighbors 
                immediate_clusters=[]
                for plt1 in range(0,len(xPltPos)):  
                    temp_list=[]
                    get_nbrs(plt1, temp_list)
                    immediate_clusters.append(list(set(temp_list)))
                    
                
                #iterate through each initial cluster and recursively construct nbrs. 
                true_clusters=[]    
                for cluster in immediate_clusters:
                    true_cluster=[]
                    true_cluster=cluster.copy() #initialize as immediate nbrs
                    generate_clusters(cluster,true_cluster)
                    #print(true_cluster.sort())
                    true_cluster.sort()
                    true_clusters.append(true_cluster)
                
                unique_data = [list(x) for x in set(tuple(x) for x in true_clusters)]
                simpleData.plt_clusters.append(unique_data)
                Xtemp=[]
                Ytemp=[]
                Ztemp=[]
                RC=[]
                for cluster in unique_data:
                    xtemp=0;
                    ytemp=0;
                    ztemp=0;
                    for index in cluster:
                        xtemp=xtemp+xPltPos[index]
                        ytemp=ytemp+yPltPos[index]
                        ztemp=ztemp+zPltPos[index]
                    xtemp=xtemp/len(cluster)
                    ytemp=ytemp/len(cluster)
                    ztemp=ztemp/len(cluster)
                    radius_clst = np.sqrt(xtemp**2 + 
                                         ytemp**2 + 
                                         ztemp**2)
                    Xtemp.append(xtemp)
                    Ytemp.append(ytemp)
                    Ztemp.append(ztemp)
                    RC.append(radius_clst)
                
                simpleData.xplt_clusters.append(Xtemp)
                simpleData.yplt_clusters.append(Ytemp)
                simpleData.zplt_clusters.append(Ztemp)
                simpleData.plt_dist_clusters.append(RC)
                
                #PLT AREA
                
                num_plts = len(temp_datafile.yPltPos)
                edges_contacting_plt=sum(temp_datafile.originalEdgeInContactWithPlatelet)

                average_plt_area = simpleData.fiber_segment_vol * (edges_contacting_plt)/num_plts
                
                simpleData.average_fiber_conn_plt.append(average_plt_area)
                
                n_plt_fib_dense = average_plt_area / simpleData.average_fiber_conn_plt[0]
                simpleData.normalized_fiber_plt_density.append(n_plt_fib_dense)
                
                #track individual platelets over time
                plt_radius_time=[]
                plt_radius_epi_time=[]
                for platelet in range(0,num_plts):
                    radius_plt = np.sqrt(temp_datafile.xPltPos[platelet]**2 + 
                                         temp_datafile.yPltPos[platelet]**2 + 
                                         temp_datafile.zPltPos[platelet]**2)
                    plt_radius_time.append(radius_plt)
                    radius_epi_plt = np.sqrt((temp_datafile.xPltPos[platelet]-temp_datafile.meanX)**2 + 
                                         (temp_datafile.yPltPos[platelet]-temp_datafile.meanY)**2 + 
                                         (temp_datafile.zPltPos[platelet]-temp_datafile.meanZ)**2)
                    plt_radius_epi_time.append(radius_epi_plt)
                
                print("plt_radius_time")
                print(plt_radius_time)
                
                #after saving radii, save in vector format
                simpleData.individual_plt_dist_from_center.append(plt_radius_time)
                simpleData.individual_plt_dist_from_epicenter.append(plt_radius_epi_time)
                    
                    
                #average number of fibers must be compared to the first box 
                #otherwise the box shrinks during simulations
                originalNodeIsInRectCenter = []
                originalNodeIsInSphereCenter = []
                originalNodeIsInSphereCenter_10=[]
                originalNodeIsInSphereCenter_20=[]
                originalNodeIsInSphereCenter_30=[]
                originalNodeIsInSphereCenter_40=[]
                originalNodeIsInSphereCenter_60=[]
                originalNodeIsInSphereCenter_70=[]
                originalNodeIsInSphereCenter_80=[]
                originalNodeIsInSphereCenter_90=[]
                originalNodeIsInSphereCenter_100=[]
                originalNodeIsInAreaSlice = []
                originalNodeIsInSphereEpiCenter = []
                originalNodeIsInSphereEpiCenter_10=[]
                originalNodeIsInSphereEpiCenter_20=[]
                originalNodeIsInSphereEpiCenter_30=[]
                originalNodeIsInSphereEpiCenter_40=[]
                originalNodeIsInSphereEpiCenter_60=[]
                originalNodeIsInSphereEpiCenter_70=[]
                originalNodeIsInSphereEpiCenter_80=[]
                originalNodeIsInSphereEpiCenter_90=[]
                originalNodeIsInSphereEpiCenter_100=[]
                tempcoord=[]
                
                for node in range(0,len(temp_datafile.xPos)):
                    x = temp_datafile.xPos[node]            
                    y = temp_datafile.yPos[node]
                    z = temp_datafile.zPos[node]
                    tempcoord.append([x,y,z])
                    
                    rad_i = np.sqrt(x**2 + y**2 + z**2)
                    radepi_i = np.sqrt((x-temp_datafile.meanX)**2 + (y-temp_datafile.meanY)**2 + (z-temp_datafile.meanZ)**2)
                    if ((x > first_box_edgeX_left) and (x < first_box_edgeX_right) and
                        (y > first_box_edgeY_left) and (y < first_box_edgeY_right) and
                        (z > first_box_edgeZ_left) and (z < first_box_edgeZ_right)):
                            
                        originalNodeIsInRectCenter.append(True)
                    else:
                        originalNodeIsInRectCenter.append(False)
                        
                
                    if (rad_i < first_radius):
                        originalNodeIsInSphereCenter.append(True)
                    else:
                        originalNodeIsInSphereCenter.append(False)
                        
                    if (rad_i < first_radius_10):
                        originalNodeIsInSphereCenter_10.append(True)
                    else:
                        originalNodeIsInSphereCenter_10.append(False)
                        
                    if (rad_i < first_radius_20):
                        originalNodeIsInSphereCenter_20.append(True)
                    else:
                        originalNodeIsInSphereCenter_20.append(False)
                        
                    if (rad_i < first_radius_30):
                        originalNodeIsInSphereCenter_30.append(True)
                    else:
                        originalNodeIsInSphereCenter_30.append(False)
                        
                    if (rad_i < first_radius_40):
                        originalNodeIsInSphereCenter_40.append(True)
                    else:
                        originalNodeIsInSphereCenter_40.append(False)
                        
                    if (rad_i < first_radius_60):
                        originalNodeIsInSphereCenter_60.append(True)
                    else:
                        originalNodeIsInSphereCenter_60.append(False)
                          
                    if (rad_i < first_radius_70):
                        originalNodeIsInSphereCenter_70.append(True)
                    else:
                        originalNodeIsInSphereCenter_70.append(False)
                        
                    if (rad_i < first_radius_80):
                        originalNodeIsInSphereCenter_80.append(True)
                    else:
                        originalNodeIsInSphereCenter_80.append(False)
                        
                    if (rad_i < first_radius_90):
                        originalNodeIsInSphereCenter_90.append(True)
                    else:
                        originalNodeIsInSphereCenter_90.append(False)
                        
                    if (rad_i < first_radius_100):
                        originalNodeIsInSphereCenter_100.append(True)
                    else:
                        originalNodeIsInSphereCenter_100.append(False)
                        
                        
                        
                    if (radepi_i < first_radius):
                        originalNodeIsInSphereEpiCenter.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter.append(False)
                        
                    if (radepi_i < first_radius_10):
                        originalNodeIsInSphereEpiCenter_10.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter_10.append(False)
                        
                    if (radepi_i < first_radius_20):
                        originalNodeIsInSphereEpiCenter_20.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter_20.append(False)
                        
                    if (radepi_i < first_radius_30):
                        originalNodeIsInSphereEpiCenter_30.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter_30.append(False)
                        
                    if (radepi_i < first_radius_40):
                        originalNodeIsInSphereEpiCenter_40.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter_40.append(False)
                        
                    if (radepi_i < first_radius_60):
                        originalNodeIsInSphereEpiCenter_60.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter_60.append(False)
                          
                    if (radepi_i < first_radius_70):
                        originalNodeIsInSphereEpiCenter_70.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter_70.append(False)
                        
                    if (radepi_i < first_radius_80):
                        originalNodeIsInSphereEpiCenter_80.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter_80.append(False)
                        
                    if (radepi_i < first_radius_90):
                        originalNodeIsInSphereEpiCenter_90.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter_90.append(False)
                        
                    if (radepi_i < first_radius_100):
                        originalNodeIsInSphereEpiCenter_100.append(True)
                    else:
                        originalNodeIsInSphereEpiCenter_100.append(False)
                
                   
                        
                        
                    if ((x > first_box_edgeX_left) and (x < first_box_edgeX_right) and
                        (y > first_box_edgeY_left) and (y < first_box_edgeY_right) and
                        (z < first_midZ + temp_datafile.center_vertical_slice) and 
                        (z > first_midZ - temp_datafile.center_vertical_slice)):
                        
                        originalNodeIsInAreaSlice.append(True)
                    else:
                        originalNodeIsInAreaSlice.append(False)
                
                fiber_density_rect=sum(originalNodeIsInRectCenter)/first_interior_rect_center_vol
                fiber_density_sphere=sum(originalNodeIsInSphereCenter)/first_interior_sphere_center_vol
                fiber_density_sphere_10=sum(originalNodeIsInSphereCenter_10)/first_interior_sphere_center_vol_10
                fiber_density_sphere_20=sum(originalNodeIsInSphereCenter_20)/first_interior_sphere_center_vol_20
                fiber_density_sphere_30=sum(originalNodeIsInSphereCenter_30)/first_interior_sphere_center_vol_30
                fiber_density_sphere_40=sum(originalNodeIsInSphereCenter_40)/first_interior_sphere_center_vol_40
                fiber_density_sphere_60=sum(originalNodeIsInSphereCenter_60)/first_interior_sphere_center_vol_60
                fiber_density_sphere_70=sum(originalNodeIsInSphereCenter_70)/first_interior_sphere_center_vol_70
                fiber_density_sphere_80=sum(originalNodeIsInSphereCenter_80)/first_interior_sphere_center_vol_80
                fiber_density_sphere_90=sum(originalNodeIsInSphereCenter_90)/first_interior_sphere_center_vol_90
                fiber_density_sphere_100=sum(originalNodeIsInSphereCenter_100)/first_interior_sphere_center_vol_100
                
                fiber_density_episphere=sum(originalNodeIsInSphereEpiCenter)/first_interior_sphere_center_vol
                fiber_density_episphere_10=sum(originalNodeIsInSphereEpiCenter_10)/first_interior_sphere_center_vol_10
                fiber_density_episphere_20=sum(originalNodeIsInSphereEpiCenter_20)/first_interior_sphere_center_vol_20
                fiber_density_episphere_30=sum(originalNodeIsInSphereEpiCenter_30)/first_interior_sphere_center_vol_30
                fiber_density_episphere_40=sum(originalNodeIsInSphereEpiCenter_40)/first_interior_sphere_center_vol_40
                fiber_density_episphere_60=sum(originalNodeIsInSphereEpiCenter_60)/first_interior_sphere_center_vol_60
                fiber_density_episphere_70=sum(originalNodeIsInSphereEpiCenter_70)/first_interior_sphere_center_vol_70
                fiber_density_episphere_80=sum(originalNodeIsInSphereEpiCenter_80)/first_interior_sphere_center_vol_80
                fiber_density_episphere_90=sum(originalNodeIsInSphereEpiCenter_90)/first_interior_sphere_center_vol_90
                fiber_density_episphere_100=sum(originalNodeIsInSphereEpiCenter_100)/first_interior_sphere_center_vol_100
                
                fiber_density_area=sum(originalNodeIsInAreaSlice)/first_interior_area_center_vol
               
                simpleData.average_fiber_density_rect.append(fiber_density_rect)
                simpleData.average_fiber_density_sphere.append(fiber_density_sphere)
                simpleData.average_fiber_density_sphere_10.append(fiber_density_sphere_10)
                simpleData.average_fiber_density_sphere_20.append(fiber_density_sphere_20)
                simpleData.average_fiber_density_sphere_30.append(fiber_density_sphere_30)
                simpleData.average_fiber_density_sphere_40.append(fiber_density_sphere_40)
                simpleData.average_fiber_density_sphere_60.append(fiber_density_sphere_60)
                simpleData.average_fiber_density_sphere_70.append(fiber_density_sphere_70)
                simpleData.average_fiber_density_sphere_80.append(fiber_density_sphere_80)
                simpleData.average_fiber_density_sphere_90.append(fiber_density_sphere_90)
                simpleData.average_fiber_density_sphere_100.append(fiber_density_sphere_100)
                simpleData.average_fiber_density_area.append(fiber_density_area)
                
                simpleData.average_fiber_density_episphere.append(fiber_density_episphere)
                simpleData.average_fiber_density_episphere_10.append(fiber_density_episphere_10)
                simpleData.average_fiber_density_episphere_20.append(fiber_density_episphere_20)
                simpleData.average_fiber_density_episphere_30.append(fiber_density_episphere_30)
                simpleData.average_fiber_density_episphere_40.append(fiber_density_episphere_40)
                simpleData.average_fiber_density_episphere_60.append(fiber_density_episphere_60)
                simpleData.average_fiber_density_episphere_70.append(fiber_density_episphere_70)
                simpleData.average_fiber_density_episphere_80.append(fiber_density_episphere_80)
                simpleData.average_fiber_density_episphere_90.append(fiber_density_episphere_90)
                simpleData.average_fiber_density_episphere_100.append(fiber_density_episphere_100)
                hull=ConvexHull(tempcoord)
                simpleData.fiber_convex_hull.append(hull);
                simpleData.fiber_convex_hull_volume.append(hull.area);
                print("convex volume:")
                print(hull.area)
                
                #normalize density
                n_fib_rect_dense = temp_datafile.fiber_rect_density / simpleData.average_fiber_density_rect[0]
                n_fib_sphere_dense = temp_datafile.fiber_sphere_density / simpleData.average_fiber_density_sphere[0]
                n_fib_sphere_dense_10 = temp_datafile.fiber_sphere_density_10 / simpleData.average_fiber_density_sphere_10[0]
                n_fib_sphere_dense_20 = temp_datafile.fiber_sphere_density_20 / simpleData.average_fiber_density_sphere_20[0]
                n_fib_sphere_dense_30 = temp_datafile.fiber_sphere_density_30 / simpleData.average_fiber_density_sphere_30[0]
                n_fib_sphere_dense_40 = temp_datafile.fiber_sphere_density_40 / simpleData.average_fiber_density_sphere_40[0]
                n_fib_sphere_dense_60 = temp_datafile.fiber_sphere_density_60 / simpleData.average_fiber_density_sphere_60[0]
                n_fib_sphere_dense_70 = temp_datafile.fiber_sphere_density_70 / simpleData.average_fiber_density_sphere_70[0]
                n_fib_sphere_dense_80 = temp_datafile.fiber_sphere_density_80 / simpleData.average_fiber_density_sphere_80[0]
                n_fib_sphere_dense_90 = temp_datafile.fiber_sphere_density_90 / simpleData.average_fiber_density_sphere_90[0]
                n_fib_sphere_dense_100 = temp_datafile.fiber_sphere_density_100 / simpleData.average_fiber_density_sphere_100[0]
                n_fib_area_dense = temp_datafile.fiber_area_density / simpleData.average_fiber_density_area[0]
                
                n_fib_episphere_dense = temp_datafile.fiber_episphere_density / simpleData.average_fiber_density_episphere[0]
                n_fib_episphere_dense_10 = temp_datafile.fiber_episphere_density_10 / simpleData.average_fiber_density_episphere_10[0]
                n_fib_episphere_dense_20 = temp_datafile.fiber_episphere_density_20 / simpleData.average_fiber_density_episphere_20[0]
                n_fib_episphere_dense_30 = temp_datafile.fiber_episphere_density_30 / simpleData.average_fiber_density_episphere_30[0]
                n_fib_episphere_dense_40 = temp_datafile.fiber_episphere_density_40 / simpleData.average_fiber_density_episphere_40[0]
                n_fib_episphere_dense_60 = temp_datafile.fiber_episphere_density_60 / simpleData.average_fiber_density_episphere_60[0]
                n_fib_episphere_dense_70 = temp_datafile.fiber_episphere_density_70 / simpleData.average_fiber_density_episphere_70[0]
                n_fib_episphere_dense_80 = temp_datafile.fiber_episphere_density_80 / simpleData.average_fiber_density_episphere_80[0]
                n_fib_episphere_dense_90 = temp_datafile.fiber_episphere_density_90 / simpleData.average_fiber_density_episphere_90[0]
                n_fib_episphere_dense_100 = temp_datafile.fiber_episphere_density_100 / simpleData.average_fiber_density_episphere_100[0]
                
               
                simpleData.normalized_fiber_density_rect.append(n_fib_rect_dense)
                simpleData.normalized_fiber_density_sphere.append(n_fib_sphere_dense)
                simpleData.normalized_fiber_density_sphere_10.append(n_fib_sphere_dense_10)
                simpleData.normalized_fiber_density_sphere_20.append(n_fib_sphere_dense_20)
                simpleData.normalized_fiber_density_sphere_30.append(n_fib_sphere_dense_30)
                simpleData.normalized_fiber_density_sphere_40.append(n_fib_sphere_dense_40)
                simpleData.normalized_fiber_density_sphere_60.append(n_fib_sphere_dense_60)
                simpleData.normalized_fiber_density_sphere_70.append(n_fib_sphere_dense_70)
                simpleData.normalized_fiber_density_sphere_80.append(n_fib_sphere_dense_80)
                simpleData.normalized_fiber_density_sphere_90.append(n_fib_sphere_dense_90)
                simpleData.normalized_fiber_density_sphere_100.append(n_fib_sphere_dense_100)
                simpleData.normalized_fiber_density_area.append(n_fib_area_dense)
                
                simpleData.normalized_fiber_density_episphere.append(n_fib_episphere_dense)
                simpleData.normalized_fiber_density_episphere_10.append(n_fib_episphere_dense_10)
                simpleData.normalized_fiber_density_episphere_20.append(n_fib_episphere_dense_20)
                simpleData.normalized_fiber_density_episphere_30.append(n_fib_episphere_dense_30)
                simpleData.normalized_fiber_density_episphere_40.append(n_fib_episphere_dense_40)
                simpleData.normalized_fiber_density_episphere_60.append(n_fib_episphere_dense_60)
                simpleData.normalized_fiber_density_episphere_70.append(n_fib_episphere_dense_70)
                simpleData.normalized_fiber_density_episphere_80.append(n_fib_episphere_dense_80)
                simpleData.normalized_fiber_density_episphere_90.append(n_fib_episphere_dense_90)
                simpleData.normalized_fiber_density_episphere_100.append(n_fib_episphere_dense_100)
                

                if ( (len(simpleData.time) > 1) ):
                    #current fiber density over last
                    T2 = simpleData.time[-1]
                    T1 = simpleData.time[-2]
                    
                    #plt radii and speed
                    
                    
                    #######################################################
                    #######################################################
                    #######################################################
                    ##calculate positions over time of platelets. 
                    plt_pos_2 = simpleData.individual_plt_dist_from_center[-1]
                    plt_pos_1 = simpleData.individual_plt_dist_from_center[-2]
                    
                    individual_plt_speed_vec=[]
                    individual_plt_dist_vec=[]
                    for platelet in range (num_plts):
                        plt_p2 = plt_pos_2[platelet]
                        plt_p1 = plt_pos_1[platelet]
                        
                        plt_vel = np.abs(plt_p2 - plt_p1)/(T2-T1)
                        plt_dist = (plt_p2 + plt_p1)/(2)
                        individual_plt_speed_vec.append(plt_vel)
                        individual_plt_dist_vec.append(plt_dist)
                        
                    print("individual_plt_speed_vec" )
                    print(individual_plt_speed_vec)
                    print("individual_plt_dist_vec" )
                    print(individual_plt_dist_vec)
                    
                    simpleData.individual_plt_speed.append(individual_plt_speed_vec)
                    simpleData.individual_plt_dist.append(individual_plt_dist_vec)
                    
                    #after tracking individuals, take mean
                    simpleData.average_plt_speed.append(np.mean(individual_plt_speed_vec))
                    
                    
                    #######################################################
                    #######################################################
                    #######################################################
                    #rectangle
                    D2 = simpleData.average_fiber_density_rect[-1]
                    D1 = simpleData.average_fiber_density_rect[-2]
                    val_vel_rect = (D2-D1)/(T2-T1)
                    simpleData.change_in_fiber_density_rect.append(val_vel_rect)
                    
                    ND2 = simpleData.normalized_fiber_density_rect[-1]
                    ND1 = simpleData.normalized_fiber_density_rect[-2]
                    Nval_vel_rect = (ND2-ND1)/(T2-T1)
                    simpleData.change_in_norm_fiber_density_rect.append(Nval_vel_rect)
                    
                    #sphere
                    D2 = simpleData.average_fiber_density_sphere[-1]
                    D1 = simpleData.average_fiber_density_sphere[-2]
                    val_vel_sphere = (D2-D1)/(T2-T1)
                    simpleData.change_in_fiber_density_sphere.append(val_vel_sphere)
                    
                    ND2 = simpleData.normalized_fiber_density_sphere[-1]
                    ND1 = simpleData.normalized_fiber_density_sphere[-2]
                    Nval_vel_sphere = (ND2-ND1)/(T2-T1)
                    simpleData.change_in_norm_fiber_density_sphere.append(Nval_vel_sphere)
                    
                    #area
                    D2 = simpleData.average_fiber_density_area[-1]
                    D1 = simpleData.average_fiber_density_area[-2]
                    val_vel_area = (D2-D1)/(T2-T1)
                    simpleData.change_in_fiber_density_area.append(val_vel_area)
                    
                    ND2 = simpleData.normalized_fiber_density_area[-1]
                    ND1 = simpleData.normalized_fiber_density_area[-2]
                    Nval_vel_area = (ND2-ND1)/(T2-T1)
                    simpleData.change_in_norm_fiber_density_area.append(Nval_vel_area)
                    

                    
                
                #Set the last change equal to last value
                if (file == files[-1]):
                    simpleData.change_in_fiber_density_rect.append( simpleData.change_in_fiber_density_rect[-1] )
                    simpleData.change_in_norm_fiber_density_rect.append( simpleData.change_in_norm_fiber_density_rect[-1] )
                    
                    
                    simpleData.change_in_fiber_density_sphere.append( simpleData.change_in_fiber_density_sphere[-1] )
                    simpleData.change_in_norm_fiber_density_sphere.append( simpleData.change_in_norm_fiber_density_sphere[-1] )
                    
                    simpleData.change_in_fiber_density_area.append( simpleData.change_in_fiber_density_area[-1] )
                    simpleData.change_in_norm_fiber_density_area.append( simpleData.change_in_norm_fiber_density_area[-1] )
                
                    simpleData.average_plt_speed.append( simpleData.average_plt_speed[-1])
                    
            data_[rowChoice] = simpleData

#This generates a short version of mutiple parameter files
def data_generate_short2(data_,rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        rowChoice=0
        
        files.sort(key=lambda s: int(re.split("\_|\.",s)[1]))
        print((files))
        #simRow should be t1-t10
        simrow = subdir.split('\\')
        if (len(simrow) > 1):
            simRow=subdir.split('\\')[1]

            #print(simRow)
            if (simRow=='F1'):
                rowChoice=0
            if (simRow=='F2'):
                rowChoice=1
            if (simRow=='F3'):
                rowChoice=2
            if (simRow=='F4'):
                rowChoice=3
            if (simRow=='F5'):
                rowChoice=4
            if (simRow=='F6'):
                rowChoice=5
            if (simRow=='F7'):
                rowChoice=6
            if (simRow=='F8'):
                rowChoice=7
            if (simRow=='F9'):
                rowChoice=8
            if (simRow=='F10'):
                rowChoice=9
                
            #for each file calculate averages and store them .
            #fill temp_datafile with only some values. 
            simpleData = DataFile();
            file_counter=0
 

                 
            first_midX = 0
            first_midY = 0
            first_midZ = 0
            first_box_edgeX_left = 0
            first_box_edgeX_right = 0
            first_box_edgeY_left = 0 
            first_box_edgeY_right = 0
            first_box_edgeZ_left = 0
            first_box_edgeZ_right = 0
            first_interior_rect_center_vol = 0
            first_interior_sphere_center_vol = 0
            first_interior_area_center_vol = 0
            
            first_radius = 0
            
            for file in files:
                
                #print(file)
                temp_datafile = DataFile(subdir,file)
                
                parser(temp_datafile)#simple parser
                
                if (file_counter == 0):
                            
                    first_lenX = temp_datafile.maxX - temp_datafile.minX
                    first_lenY = temp_datafile.maxY - temp_datafile.minY
                    first_lenZ = temp_datafile.maxZ - temp_datafile.minZ
                    
                    first_midX = (temp_datafile.maxX + temp_datafile.minX)/2.0
                    first_midY = (temp_datafile.maxY + temp_datafile.minY)/2.0
                    first_midZ = (temp_datafile.maxZ + temp_datafile.minZ)/2.0
                    first_box_edgeX_left = first_midX - first_lenX * temp_datafile.center_fraction/2
                    first_box_edgeX_right = first_midX + first_lenX * temp_datafile.center_fraction/2
                    first_box_edgeY_left = first_midY - first_lenY * temp_datafile.center_fraction/2
                    first_box_edgeY_right = first_midY + first_lenY * temp_datafile.center_fraction/2
                    first_box_edgeZ_left = first_midZ - first_lenZ * temp_datafile.center_fraction/2
                    first_box_edgeZ_right = first_midZ + first_lenZ * temp_datafile.center_fraction/2
                
                    first_radius = first_lenX * temp_datafile.center_sphere_fraction
                    first_volume = (4.0/3.0) * (np.pi) * (first_radius**3.0)
                         
                    first_interior_rect_center_vol = 2 * first_lenX + 2 * first_lenY + 2 * first_lenZ
                    first_interior_sphere_center_vol = first_volume
                    first_interior_area_center_vol = 2*first_lenX + 2*first_lenY + 2 * temp_datafile.center_vertical_slice
                    
                #TIME
                simpleData.time.append(temp_datafile.time/60)
                
                #PLT AREA
                
                num_plts = len(temp_datafile.yPltPos)
                edges_contacting_plt=sum(temp_datafile.originalEdgeInContactWithPlatelet)

                average_plt_area = simpleData.fiber_segment_vol * (edges_contacting_plt)/num_plts
                
                simpleData.average_fiber_conn_plt.append(average_plt_area)
                
                n_plt_fib_dense = average_plt_area / simpleData.average_fiber_conn_plt[0]
                simpleData.normalized_fiber_plt_density.append(n_plt_fib_dense)
                
                #track individual platelets over time
                plt_radius_time=[]
                plt_radius_epi_time=[]
                for platelet in range(0,num_plts):
                    radius_plt = np.sqrt(temp_datafile.xPltPos[platelet]**2 + 
                                         temp_datafile.yPltPos[platelet]**2 + 
                                         temp_datafile.zPltPos[platelet]**2)
                    plt_radius_time.append(radius_plt)
                    radius_epi_plt = np.sqrt((temp_datafile.xPltPos[platelet]-temp_datafile.meanX)**2 + 
                                         (temp_datafile.yPltPos[platelet]-temp_datafile.meanY)**2 + 
                                         (temp_datafile.zPltPos[platelet]-temp_datafile.meanZ)**2)
                    plt_radius_epi_time.append(radius_epi_plt)
                
                print("plt_radius_time")
                print(plt_radius_time)
                
                #after saving radii, save in vector format
                simpleData.individual_plt_dist_from_center.append(plt_radius_time)
                simpleData.individual_plt_dist_from_epicenter.append(plt_radius_epi_time)
                    
                    
                #average number of fibers must be compared to the first box 
                #otherwise the box shrinks during simulations
                originalNodeIsInRectCenter = []
                originalNodeIsInSphereCenter = []
                originalNodeIsInAreaSlice = []
                
                for node in range(0,len(temp_datafile.xPos)):
                    x = temp_datafile.xPos[node]            
                    y = temp_datafile.yPos[node]
                    z = temp_datafile.zPos[node]
                    
                    rad_i = np.sqrt(x**2 + y**2 + z**2)
                    if ((x > first_box_edgeX_left) and (x < first_box_edgeX_right) and
                        (y > first_box_edgeY_left) and (y < first_box_edgeY_right) and
                        (z > first_box_edgeZ_left) and (z < first_box_edgeZ_right)):
                            
                        originalNodeIsInRectCenter.append(True)
                    else:
                        originalNodeIsInRectCenter.append(False)
                        
                
                    if (rad_i < first_radius):
                        originalNodeIsInSphereCenter.append(True)
                    else:
                        originalNodeIsInSphereCenter.append(False)
                        
                        
                    if ((x > first_box_edgeX_left) and (x < first_box_edgeX_right) and
                        (y > first_box_edgeY_left) and (y < first_box_edgeY_right) and
                        (z < first_midZ + temp_datafile.center_vertical_slice) and 
                        (z > first_midZ - temp_datafile.center_vertical_slice)):
                        
                        originalNodeIsInAreaSlice.append(True)
                    else:
                        originalNodeIsInAreaSlice.append(False)
                
                fiber_density_rect=sum(originalNodeIsInRectCenter)/first_interior_rect_center_vol
                fiber_density_sphere=sum(originalNodeIsInSphereCenter)/first_interior_sphere_center_vol
                fiber_density_area=sum(originalNodeIsInAreaSlice)/first_interior_area_center_vol
               
                simpleData.average_fiber_density_rect.append(fiber_density_rect)
                simpleData.average_fiber_density_sphere.append(fiber_density_sphere)
                simpleData.average_fiber_density_area.append(fiber_density_area)
                
                #normalize density
                n_fib_rect_dense = temp_datafile.fiber_rect_density / simpleData.average_fiber_density_rect[0]
                n_fib_sphere_dense = temp_datafile.fiber_sphere_density / simpleData.average_fiber_density_sphere[0]
                n_fib_area_dense = temp_datafile.fiber_area_density / simpleData.average_fiber_density_area[0]
               
                simpleData.normalized_fiber_density_rect.append(n_fib_rect_dense)
                simpleData.normalized_fiber_density_sphere.append(n_fib_sphere_dense)
                simpleData.normalized_fiber_density_area.append(n_fib_area_dense)
                

                if ( (len(simpleData.time) > 1) ):
                    #current fiber density over last
                    T2 = simpleData.time[-1]
                    T1 = simpleData.time[-2]
                    
                    #plt radii and speed
                    
                    
                    #######################################################
                    #######################################################
                    #######################################################
                    ##calculate positions over time of platelets. 
                    plt_pos_2 = simpleData.individual_plt_dist_from_center[-1]
                    plt_pos_1 = simpleData.individual_plt_dist_from_center[-2]
                    
                    individual_plt_speed_vec=[]
                    individual_plt_dist_vec=[]
                    for platelet in range (num_plts):
                        plt_p2 = plt_pos_2[platelet]
                        plt_p1 = plt_pos_1[platelet]
                        
                        plt_vel = np.abs(plt_p2 - plt_p1)/(T2-T1)
                        plt_dist = (plt_p2 + plt_p1)/(2)
                        individual_plt_speed_vec.append(plt_vel)
                        individual_plt_dist_vec.append(plt_dist)
                        
                    print("individual_plt_speed_vec" )
                    print(individual_plt_speed_vec)
                    print("individual_plt_dist_vec" )
                    print(individual_plt_dist_vec)
                    
                    simpleData.individual_plt_speed.append(individual_plt_speed_vec)
                    simpleData.individual_plt_dist.append(individual_plt_dist_vec)
                    
                    #after tracking individuals, take mean
                    simpleData.average_plt_speed.append(np.mean(individual_plt_speed_vec))
                    
                    
                    #######################################################
                    #######################################################
                    #######################################################
                    #rectangle
                    D2 = simpleData.average_fiber_density_rect[-1]
                    D1 = simpleData.average_fiber_density_rect[-2]
                    val_vel_rect = (D2-D1)/(T2-T1)
                    simpleData.change_in_fiber_density_rect.append(val_vel_rect)
                    
                    ND2 = simpleData.normalized_fiber_density_rect[-1]
                    ND1 = simpleData.normalized_fiber_density_rect[-2]
                    Nval_vel_rect = (ND2-ND1)/(T2-T1)
                    simpleData.change_in_norm_fiber_density_rect.append(Nval_vel_rect)
                    
                    #sphere
                    D2 = simpleData.average_fiber_density_sphere[-1]
                    D1 = simpleData.average_fiber_density_sphere[-2]
                    val_vel_sphere = (D2-D1)/(T2-T1)
                    simpleData.change_in_fiber_density_sphere.append(val_vel_sphere)
                    
                    ND2 = simpleData.normalized_fiber_density_sphere[-1]
                    ND1 = simpleData.normalized_fiber_density_sphere[-2]
                    Nval_vel_sphere = (ND2-ND1)/(T2-T1)
                    simpleData.change_in_norm_fiber_density_sphere.append(Nval_vel_sphere)
                    
                    #area
                    D2 = simpleData.average_fiber_density_area[-1]
                    D1 = simpleData.average_fiber_density_area[-2]
                    val_vel_area = (D2-D1)/(T2-T1)
                    simpleData.change_in_fiber_density_area.append(val_vel_area)
                    
                    ND2 = simpleData.normalized_fiber_density_area[-1]
                    ND1 = simpleData.normalized_fiber_density_area[-2]
                    Nval_vel_area = (ND2-ND1)/(T2-T1)
                    simpleData.change_in_norm_fiber_density_area.append(Nval_vel_area)
                    

                    
                
                #Set the last change equal to last value
                if (file == files[-1]):
                    simpleData.change_in_fiber_density_rect.append( simpleData.change_in_fiber_density_rect[-1] )
                    simpleData.change_in_norm_fiber_density_rect.append( simpleData.change_in_norm_fiber_density_rect[-1] )
                    
                    
                    simpleData.change_in_fiber_density_sphere.append( simpleData.change_in_fiber_density_sphere[-1] )
                    simpleData.change_in_norm_fiber_density_sphere.append( simpleData.change_in_norm_fiber_density_sphere[-1] )
                    
                    simpleData.change_in_fiber_density_area.append( simpleData.change_in_fiber_density_area[-1] )
                    simpleData.change_in_norm_fiber_density_area.append( simpleData.change_in_norm_fiber_density_area[-1] )
                
                    simpleData.average_plt_speed.append( simpleData.average_plt_speed[-1])
                    
            data_[rowChoice] = simpleData