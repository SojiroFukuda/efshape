""" Python for elliptic-Fourier and Principal Component Analysis.
    efa.py provides various methods to analyse closed contours in your images.
"""

import cv2
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import os
from sklearn.decomposition import PCA
import random
import scipy.stats as st


def convShape2func(cnt:np.ndarray):
    """_summary_

    Args:
        cnt (np.ndarray): a contour data detected by findContours() from openCV.

    Returns:
        dt (np.ndarray): length along the countour between each coordinate.
        cum (np.ndarray): cumlative length of dt
    """
    dt = np.zeros( len( cnt ) - 1 ) # length along the countour between each coordinate.
    cum = np.zeros( len( cnt ) ) # cumlative length of dt.
    cum[0] = 0
    for i in range(len( cnt ) - 1):
        dt[i] = np.sqrt( (cnt[i+1][0][0]-cnt[i][0][0])*(cnt[i+1][0][0]-cnt[i][0][0]) + (cnt[i+1][0][1]-cnt[i][0][1])*(cnt[i+1][0][1]-cnt[i][0][1]) )
        cum[i+1] = cum[i] + dt[i]
    return dt, cum

def getXYCoord(cnt):
    length = len(cnt)
    x_t = np.zeros(length)
    y_t = np.zeros(length)
    for i in range(length):
        x_t[i] = cnt[i][0][0]
        y_t[i] = cnt[i][0][1]
    return x_t,y_t

def adjustXYCoord(x,y):
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    x_center = x_min+np.abs(x_max-x_min)/2
    y_center = y_min+np.abs(y_max-y_min)/2
    x_a = (x-x_center)
    y_a = -(y-y_center)
    return x_a,y_a

def adjustCoordwithTime(cnt,t):
    dt, cum = convShape2func(cnt)
    x,y = getXYCoord(cnt)
    inter_func_X = interp1d(cum/cum[-1],x,kind='linear')
    inter_func_Y = interp1d(cum/cum[-1],y,kind='linear')
    x_p = inter_func_X(t)
    y_p = inter_func_Y(t)
    return x_p,y_p

def fourierApproximation(cnt,N):
    T = 1.0
    delt = 0.001
    efd_list = [] # original elliptic Fourier descriptors (EFDs)
    t = np.arange(0,T,delt)
    x_r = np.zeros(len(t)) #X-coordinate of reconstructed shape from EFDs
    y_r = np.zeros(len(t)) #Y-coordinate of reconstructed shape from EFDs
    xf_r = np.zeros(len(t)) #X-coordinate of reconstructed shape from EFDs
    yf_r = np.zeros(len(t)) #Y-coordinate of reconstructed shape from EFDs
    # cnt = np.append(cn)
    # cnt = np.concatenate( (cnt,[cnt[0]]) )
    # dt, cum = convShape2func(cnt)

    x,y = adjustXYCoord(getXYCoord(cnt)[0],getXYCoord(cnt)[1])
    xf,yf = adjustXYCoord(getXYCoord(cnt)[0]*(-1),getXYCoord(cnt)[1])
    # x,y = getXYCoord(cnt)
    dt = np.zeros( len( x ) - 1 ) # length along the countour between each coordinate.
    cum = np.zeros( len( x ) ) # cumlative length of dt.
    dtf = np.zeros( len( xf ) - 1 ) # length along the countour between each coordinate.
    cumf = np.zeros( len( xf ) ) # cumlative length of dt.
    cum[0] = 0
    cumf[0] = 0
    for i in range(len( x ) - 1):
        dt[i] = np.sqrt( (x[i+1]-x[i])*(x[i+1]-x[i]) + (y[i+1]-y[i])*(y[i+1]-y[i]) )
        cum[i+1] = cum[i] + dt[i]
        dtf[i] = np.sqrt( (xf[i+1]-xf[i])*(xf[i+1]-xf[i]) + (yf[i+1]-yf[i])*(yf[i+1]-yf[i]) )
        cumf[i+1] = cumf[i] + dtf[i]
    #create func 
    inter_func_X = interp1d(cum/cum[-1],x,kind='linear')
    inter_func_Y = interp1d(cum/cum[-1],y,kind='linear')
    inter_func_Xf = interp1d(cumf/cumf[-1],xf,kind='linear')
    inter_func_Yf = interp1d(cumf/cumf[-1],yf,kind='linear')
    # align the coordinates evenly along the contour
    x_p = inter_func_X(t)
    y_p = inter_func_Y(t)
    xf_p = inter_func_Xf(t)
    yf_p = inter_func_Yf(t)
    N_list = []
    harmonics = []
    harmonicsf = []
    for i in range(N):
        #calculate EFDs
        an, bn, cn, dn = efd(x_p,y_p,T,t,delt,i+1)
        efd_list.append(np.array([an,bn,cn,dn]))
        #Reconstruction
        x_r += an*np.cos(2*(i+1)*np.pi*t/T) + bn*np.sin(2*(i+1)*np.pi*t/T)
        y_r += cn*np.cos(2*(i+1)*np.pi*t/T) + dn*np.sin(2*(i+1)*np.pi*t/T)
        harmonics.append( [an*np.cos(2*(i+1)*np.pi*t/T) + bn*np.sin(2*(i+1)*np.pi*t/T),cn*np.cos(2*(i+1)*np.pi*t/T) + dn*np.sin(2*(i+1)*np.pi*t/T) ] )
        N_list.append([np.copy(x_r),np.copy(y_r)])

        #calculate EFDs
        anf, bnf, cnf, dnf = efd(xf_p,yf_p,T,t,delt,i+1)
        # efd_list.append(np.array([anf,bnf,cnf,dnf]))
        #Reconstruction
        xf_r += anf*np.cos(2*(i+1)*np.pi*t/T) + bnf*np.sin(2*(i+1)*np.pi*t/T)
        yf_r += cnf*np.cos(2*(i+1)*np.pi*t/T) + dnf*np.sin(2*(i+1)*np.pi*t/T)
        harmonicsf.append( [anf*np.cos(2*(i+1)*np.pi*t/T) + bnf*np.sin(2*(i+1)*np.pi*t/T),cnf*np.cos(2*(i+1)*np.pi*t/T) + dnf*np.sin(2*(i+1)*np.pi*t/T) ] )
        # N_list.append([np.copy(x_r),np.copy(y_r)])

    return N_list,harmonics,harmonicsf,x_p,y_p,t

#function that returns the traditional EFDs
def efd(x_p,y_p,T,t_p,dt,n):
    an = 0
    bn = 0
    cn = 0
    dn = 0
    for i in range( 1, len(x_p) ):
        del_xp = x_p[i]-x_p[i-1]
        del_yp = y_p[i]-y_p[i-1]
        del_t_test = np.sqrt(del_xp*del_xp+del_yp*del_yp)
        pi = np.pi
        an +=  ( del_xp / dt ) * ( np.cos(2*n*pi*t_p[i]/T) - np.cos(2*n*pi*t_p[i-1]/T) )
        bn +=  ( del_xp / dt ) * ( np.sin(2*n*pi*t_p[i]/T) - np.sin(2*n*pi*t_p[i-1]/T) )
        cn +=  ( del_yp / dt ) * ( np.cos(2*n*pi*t_p[i]/T) - np.cos(2*n*pi*t_p[i-1]/T) )
        dn +=  ( del_yp / dt ) * ( np.sin(2*n*pi*t_p[i]/T) - np.sin(2*n*pi*t_p[i-1]/T) )
    an = an* (T/(2*n*n*pi*pi))
    bn = bn* (T/(2*n*n*pi*pi))
    cn = cn* (T/(2*n*n*pi*pi))
    dn = dn* (T/(2*n*n*pi*pi))
    return an,bn,cn,dn

def efd_norm(efd_list,N,t,T):
    efd_star_list = []
    x_r, y_r = reconstContourCoord(efd_list,N,t,T)
    a1, b1, c1, d1 = efd_list[0]
    x1 = x_r[0]
    y1 = y_r[0]
    atan = np.arctan2( (2 * ( a1*b1 + c1*d1 )) , ( a1*a1 + c1*c1 - b1*b1 - d1*d1 ) )
    if atan < 0:
        atan += 2*np.pi
    theta = 0.5 * atan

    a1_star = a1 * np.cos(theta) + b1 * np.sin(theta)
    c1_star = c1 * np.cos(theta) + d1 * np.sin(theta)
    b1_star = -1 * a1 * np.sin(theta) + b1 * np.cos(theta)
    d1_star = -1 * c1 * np.sin(theta) + d1 * np.cos(theta)

    psi_1 = np.arctan2( c1_star , a1_star )
    if psi_1 < 0:
        psi_1 += 2*np.pi

    E = np.sqrt( a1_star*a1_star + c1_star*c1_star )
    psi_mat = np.array([[np.cos(psi_1),np.sin(psi_1)],[-1*np.sin(psi_1),np.cos(psi_1)]])
    x_star = np.zeros(len(t)) #X-coordinate of reconstructed shape from normalized EFDs
    y_star = np.zeros(len(t)) #Y-coordinate of reconstructed shape from normalized EFDs
    harmonics = []
    for j in range(N):
        aj = efd_list[j][0]
        bj = efd_list[j][1]
        cj = efd_list[j][2]
        dj = efd_list[j][3]
        efd_n = np.array([[aj,bj],[cj,dj]])
        theta_mat = np.array([[np.cos((j+1)*theta),-1*np.sin((j+1)*theta)],[np.sin((j+1)*theta),np.cos((j+1)*theta)]])
        efd_star = np.dot( np.dot(psi_mat,efd_n), theta_mat)
        efd_star_array = np.array([efd_star[0][0],efd_star[0][1],efd_star[1][0],efd_star[1][1]])
        efd_star_array = efd_star_array / E
        efd_star_list.append(efd_star_array) # acquire normalized EFDs
    return efd_star_list

def draw_contours(img,contours,file_name,save_dir):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img)
    ax.set_aspect('equal', 'datalim')
    ax.axis('off')
    for i, cnt in enumerate(contours):
        cnt = np.squeeze(cnt, axis=1)  # (NumPoints, 1, 2) -> (NumPoints, 2)
        # draw line between landmarks on the countour
        ax.add_patch(Polygon(cnt, color='b', fill=None, lw=2))
        # emphasize the landmark
        ax.plot(cnt[:, 0], cnt[:, 1], 'ro', mew=0, ms=2)
        # give each sample it's number
        ax.text(cnt[0][0], cnt[0][1], i+1, color='orange', size='8')
    os.makedirs(save_dir+os.sep+"contdir",exist_ok=True)
    file_name =file_name.split(os.sep)[-1]
    plt.savefig(save_dir+os.sep+"contdir"+os.sep+file_name+"_c.pdf")
    plt.close()

def getValidContours(im_path,MIN,BGC):
    img = cv2.imread(im_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if BGC == 0: # 0 = white 1 = black
        gray = cv2.bitwise_not(gray)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validcnt =[]
    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= MIN:
            validcnt.append(contours[i])
    return gray,validcnt

def setScale(contours,POSITION):
    index = 0
    area = 1
    x,y,w,h = cv2.boundingRect(contours[0])
    scale_cand = np.zeros(4)
    Min_coord = np.array([y,y+h,x+w,x])
    # [up,bottom,right,left]
    for i,cnt in enumerate(contours):
        x,y,width,height= cv2.boundingRect(cnt)
        up = y
        bottom = y + height
        right = x + width
        left = x
        if Min_coord[0] > up:
            Min_coord[0] = up
            scale_cand[0] = i

        if Min_coord[1] < bottom:
            Min_coord[1] = bottom
            scale_cand[1] = i

        if Min_coord[2] < right:
            Min_coord[2] = right
            scale_cand[2] = i

        if Min_coord[3] > left:
            Min_coord[3] = left
            scale_cand[3] = i
    
    index = scale_cand[POSITION]
    # print("index: "+str(index))
    area = cv2.contourArea(contours[int(index)])
    x,y,widht,height = cv2.boundingRect(contours[int(index)])
    return int(index),area,width,height

def contours_check(img,contours):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_contours(ax, img, contours)
    plt.show()

def group_list(string,separator):
    grouplist = string.split(separator)
    return grouplist

def FPS_calc(im_path,isSaveAll,save_dir,file,grouplist,HEADER,BGC,delt,N,MIN,SCALE,UNIT,SCALE_VALUE,POSITION):
    T = 1
    t = np.arange(0,T,delt)
    
    gray,contours = getValidContours(im_path,MIN,BGC)
    scale_index, scale_area, scale_width, scale_height = setScale(contours,POSITION)

    # unit/px
    unit_conv = 1
    if SCALE == 1:
        unit_conv = SCALE_VALUE/scale_area
        contours.pop(scale_index)
    elif SCALE == 2:
        unit_conv = (SCALE_VALUE/scale_width)*(SCALE_VALUE/scale_width)
        contours.pop(scale_index)
    elif SCALE == 3:
        unit_conv = (SCALE_VALUE/scale_height)*(SCALE_VALUE/scale_height)
        contours.pop(scale_index)
    # draw_contours(gray,contours,file,save_dir)  ## test
    Photo = im_path.split(os.sep)[-1].split(os.extsep)[0]
    len_contours = len(contours)
    INDEX_PIXEL = 0
    INDEX_AREA = 1
    INDEX_PHI = 2
    INDEX_PSI = 3
    INDEX_PERIM = 4
    INDEX_ELONG = 5
    INDEX_CONVH = 6
    INDEX_CONVA = 7
    INDEX_CP = 8
    INDEX_CA = 9
    INDEX_COX = 10
    INDEX_BLOTTPYE = 11

    if SCALE == 0:
        FPS_HEADER = ["pixel","Area(px)","phi","PSI","perim","elongation","convex hull","convex area","Cp(Glasby&Horgan)","Ca(Sochan et al)","Circ.(Cox, 1927)","Circ.(Blott&Pye)"]
    elif SCALE == 1:
        FPS_HEADER = ["pixel","Area("+UNIT+")","phi","PSI","perim","elongation","convex hull","convex area","Cp(Glasby&Horgan)","Ca(Sochan et al)","Circ.(Cox, 1927)","Circ.(Blott&Pye)"]
    else:
        FPS_HEADER = ["pixel","Area("+UNIT+"^2)","phi","PSI","perim","elongation","convex hull","convex area","Cp(Glasby&Horgan)","Ca(Sochan et al)","Circ.(Cox, 1927)","Circ.(Blott&Pye)"]

    PRE_HEADER = len(FPS_HEADER)

    for i in range(N):
        FPS_HEADER.append("FPS"+str(i+1))
    for i in range(N):
        FPS_HEADER.append("a"+str(i+1))
        FPS_HEADER.append("b"+str(i+1))
        FPS_HEADER.append("c"+str(i+1))
        FPS_HEADER.append("d"+str(i+1))
    for i in range(N):
        FPS_HEADER.append("Xam"+str(i+1))
        FPS_HEADER.append("Yam"+str(i+1))
    


    FPS_matrix = np.zeros((len(contours),7*N+PRE_HEADER))
    area_array = np.zeros(len(contours))
    perim_array = np.zeros(len(contours))
    # Fourier analysis
    if isSaveAll:
        os.makedirs(save_dir+os.sep+"NormalizedDATASET",exist_ok=True)
        os.makedirs(save_dir+os.sep+"NormalizedDATASET_X",exist_ok=True)
        os.makedirs(save_dir+os.sep+"NormalizedDATASET_Y",exist_ok=True)
        os.makedirs(save_dir+os.sep+"NormalizedDATASET_EH",exist_ok=True)
    for i,cnt in enumerate(contours):
        #Area, Perimeter
        FPS_matrix[i][INDEX_PIXEL] = cv2.contourArea(cnt) #Pixel Area
        FPS_matrix[i][INDEX_AREA] = cv2.contourArea(cnt) * unit_conv # scaled Area
        FPS_matrix[i][INDEX_PHI] = -1*np.log2(cv2.contourArea(cnt) * unit_conv) # phi
        FPS_matrix[i][INDEX_PSI] = np.log2(cv2.contourArea(cnt) * unit_conv) # Psi
        FPS_matrix[i][INDEX_PERIM] = cv2.arcLength(cnt, True) * np.sqrt(unit_conv) # Perim
        if len(cnt) < 5:
            FPS_matrix[i][INDEX_ELONG] = -1
        else:
            (x_,y_),(ma,MA),angle_ = cv2.fitEllipse(cnt)
            FPS_matrix[i][INDEX_ELONG] = ma/MA
        
        FPS_matrix[i][INDEX_CONVH] = cv2.arcLength(cv2.convexHull(cnt,False),True) *  np.sqrt(unit_conv)
        FPS_matrix[i][INDEX_CONVA] = cv2.contourArea(cv2.convexHull(cnt,False))  * unit_conv 
        FPS_matrix[i][INDEX_CP] = cv2.arcLength(cv2.convexHull(cnt,False),True) *  np.sqrt(unit_conv) / (cv2.arcLength(cnt, True) * np.sqrt(unit_conv))
        FPS_matrix[i][INDEX_CA] = (cv2.contourArea(cv2.convexHull(cnt,False)) * unit_conv  - cv2.contourArea(cnt) * unit_conv)/(cv2.contourArea(cnt)* unit_conv)
        FPS_matrix[i][INDEX_COX] = 4*np.pi*cv2.contourArea(cnt) * unit_conv / ( cv2.arcLength(cnt, True) * np.sqrt(unit_conv) * cv2.arcLength(cnt, True) * np.sqrt(unit_conv) )
        FPS_matrix[i][INDEX_BLOTTPYE] = cv2.arcLength(cnt, True) * np.sqrt(unit_conv) * cv2.arcLength(cnt, True) * np.sqrt(unit_conv) / ( cv2.contourArea(cnt) * unit_conv )
        #EFA
        efd_list = [] # original elliptic Fourier descriptors (EFDs)
        efd_star_list = [] # normalized Fourier descriptors
        x_r = np.zeros(len(t)) #X-coordinate of reconstructed shape from EFDs
        y_r = np.zeros(len(t)) #Y-coordinate of reconstructed shape from EFDs
        # cnt = np.append(cn)
        cnt = np.concatenate( (cnt,[cnt[0]]) )
        dt, cum = convShape2func(cnt)
        x,y = getXYCoord(cnt)
        #create func 
        inter_func_X = interp1d(cum/cum[-1],x,kind='linear')
        inter_func_Y = interp1d(cum/cum[-1],y,kind='linear')
        # align the coordinates evenly along the contour
        x_p = inter_func_X(t)
        y_p = inter_func_Y(t)
        # plt.plot(x_p,y_p)
        # plt.axes().set_aspect('equal', 'datalim')
        # plt.show()
        for j in range(N):
            #calculate EFDs
            an, bn, cn, dn = efd(x_p,y_p,T,t,delt,j+1)
            efd_list.append(np.array([an,bn,cn,dn]))
            #Reconstruction
            x_r += an*np.cos(2*(j+1)*np.pi*t/T) + bn*np.sin(2*(j+1)*np.pi*t/T)
            y_r += cn*np.cos(2*(j+1)*np.pi*t/T) + dn*np.sin(2*(j+1)*np.pi*t/T)
        if isSaveAll:
            fig,ax = plt.subplots()
            ax.plot(x_r,y_r)
            ax.fill_between(x_r,y_r)
            ax.set_aspect('equal','datalim')
            # ax.set_xlim([-1.5,1.5])
            # ax.set_ylim([-1.0,1.0])
            plt.savefig(save_dir+os.sep+"NormalizedDATASET"+os.sep+Photo+'_'+str(i+1)+"_oc.pdf")
            plt.close()

            fig,ax = plt.subplots()
            ax.plot(t,x_r)
            # ax.fill_between(x_r,t)
            # ax.set_aspect('equal','datalim')
            plt.xlim([0,1])
            plt.savefig(save_dir+os.sep+"NormalizedDATASET_X"+os.sep+Photo+'_'+str(i+1)+"_oc_x.pdf")
            plt.close()

            fig,ax = plt.subplots()
            ax.plot(t,y_r)
            # ax.fill_between(y_r,t)
            # ax.set_aspect('equal','datalim')
            plt.xlim([0,1])
            plt.savefig(save_dir+os.sep+"NormalizedDATASET_Y"+os.sep+Photo+'_'+str(i+1)+"_oc_y.pdf")
            plt.close()

        # confirm the reconstructed shape
        # plt.plot(x_r,y_r)
        # plt.axes().set_aspect('equal', 'datalim')
        # plt.show()

        ## Normalization (size, axial rotation, starting point)
        a1, b1, c1, d1 = efd_list[0]
        x1 = x_r[0]
        y1 = y_r[0]
        atan = np.arctan2( (2 * ( a1*b1 + c1*d1 )) , ( a1*a1 + c1*c1 - b1*b1 - d1*d1 ) )
        if atan < 0:
            atan += 2*np.pi
        theta = 0.5 * atan

        a1_star = a1 * np.cos(theta) + b1 * np.sin(theta)
        c1_star = c1 * np.cos(theta) + d1 * np.sin(theta)
        b1_star = -1 * a1 * np.sin(theta) + b1 * np.cos(theta)
        d1_star = -1 * c1 * np.sin(theta) + d1 * np.cos(theta)

        psi_1 = np.arctan2( c1_star , a1_star )
        if psi_1 < 0:
            psi_1 += 2*np.pi

        E = np.sqrt( a1_star*a1_star + c1_star*c1_star )
        psi_mat = np.array([[np.cos(psi_1),np.sin(psi_1)],[-1*np.sin(psi_1),np.cos(psi_1)]])
        x_star = np.zeros(len(t)) #X-coordinate of reconstructed shape from normalized EFDs
        y_star = np.zeros(len(t)) #Y-coordinate of reconstructed shape from normalized EFDs
        harmonics = []
        for j in range(N):
            aj = efd_list[j][0]
            bj = efd_list[j][1]
            cj = efd_list[j][2]
            dj = efd_list[j][3]
            efd_n = np.array([[aj,bj],[cj,dj]])
            theta_mat = np.array([[np.cos((j+1)*theta),-1*np.sin((j+1)*theta)],[np.sin((j+1)*theta),np.cos((j+1)*theta)]])
            efd_star = np.dot( np.dot(psi_mat,efd_n), theta_mat)
            efd_star_array = np.array([efd_star[0][0],efd_star[0][1],efd_star[1][0],efd_star[1][1]])
            efd_star_array = efd_star_array / E
            efd_star_list.append(efd_star_array) # acquire normalized EFDs
            fps_value = (efd_star_array[0]*efd_star_array[0]+efd_star_array[1]*efd_star_array[1]+efd_star_array[2]*efd_star_array[2]+efd_star_array[3]*efd_star_array[3])/2
            # Fourier Power Spectra
            FPS_matrix[i][j+PRE_HEADER] = fps_value
            # traditional EFDs
            FPS_matrix[i][4*j+PRE_HEADER+N] = efd_star_array[0]
            FPS_matrix[i][4*j+PRE_HEADER+N+1] = efd_star_array[1]
            FPS_matrix[i][4*j+PRE_HEADER+N+2] = efd_star_array[2]
            FPS_matrix[i][4*j+PRE_HEADER+N+3] = efd_star_array[3]
            # amplitude 
            FPS_matrix[i][2*j+PRE_HEADER+N*5] = (efd_star_array[0]*efd_star_array[0]+efd_star_array[1]*efd_star_array[1])/2
            FPS_matrix[i][2*j+PRE_HEADER+N*5+1] = (efd_star_array[2]*efd_star_array[2]+efd_star_array[3]*efd_star_array[3])/2
            x_star += efd_star_array[0]*np.cos(2*(j+1)*np.pi*t/T) + efd_star_array[1]*np.sin(2*(j+1)*np.pi*t/T)
            y_star += efd_star_array[2]*np.cos(2*(j+1)*np.pi*t/T) + efd_star_array[3]*np.sin(2*(j+1)*np.pi*t/T)
            harmonics.append( [efd_star_array[0]*np.cos(2*(j+1)*np.pi*t/T) + efd_star_array[1]*np.sin(2*(j+1)*np.pi*t/T), efd_star_array[2]*np.cos(2*(j+1)*np.pi*t/T) + efd_star_array[3]*np.sin(2*(j+1)*np.pi*t/T) ] )
        # Save all normalized clasts
        if isSaveAll:
            fig,ax = plt.subplots()
            ax.plot(x_star,y_star)
            ax.fill_between(x_star,y_star)
            ax.set_aspect('equal','datalim')
            ax.set_xlim([-1.5,1.5])
            ax.set_ylim([-1.0,1.0])
            plt.savefig(save_dir+os.sep+"NormalizedDATASET"+os.sep+Photo+'_'+str(i+1)+"_nc.pdf")
            plt.close()
            # os.makedirs(save_dir+os.sep+"NormalizedDATASET_EH",exist_ok=True)

            for j,harmonic in enumerate(harmonics):
                plt.close()
                # matplotlib.pyplot.plot(self.tsample,harmonic[0]-harmonicsf[i][0])
                plt.plot(t,harmonic[0])
                # matplotlib.pyplot.axes().set_aspect(aspect=0.5)
                # matplotlib.pyplot.axes().tick_params(labelbottom=False,bottom=False);
                # matplotlib.pyplot.axes().tick_params(labelleft=False,left=False);
                plt.axes().set( xlim=(min(t)-0.1,max(t)+0.1), ylim=( min(harmonic[0])-0.1, max(harmonic[0])+0.1 ) )
                plt.savefig(save_dir+os.sep+"NormalizedDATASET_EH"+os.sep+Photo+'_'+str(i+1)+'_'+str(j+1)+'_X.pdf')
                plt.close()
                # matplotlib.pyplot.plot(self.tsample,harmonic[1]-harmonicsf[i][1])
                plt.plot(t,harmonic[1])
                # matplotlib.pyplot.axes().set_aspect(aspect=0.5)
                # matplotlib.pyplot.axes().tick_params(labelbottom=False,bottom=False);
                # matplotlib.pyplot.axes().tick_params(labelleft=False,left=False);
                plt.axes().set( xlim=(min(t)-0.1,max(t)+0.1), ylim=( min(harmonic[1])-0.1, max(harmonic[1])+0.1 ) )
                plt.savefig(save_dir+os.sep+"NormalizedDATASET_EH"+os.sep+Photo+'_'+str(i+1)+'_'+str(j+1)+'_Y.pdf')

            margin = max( x_star*0.4 )
            for k in range(0,int(len(t)/5)):
                plt.plot(x_star,y_star)
                x_r = 0; y_r = 0; 
                for l,harmonic in enumerate(harmonics):
                    plt.plot([x_r,x_r + harmonic[0][k*5]],[y_r , y_r + harmonic[1][k*5] ],color='red')
                    x_hn = harmonic[0] + x_r
                    y_hn = harmonic[1] + y_r
                    x_r += harmonic[0][k*5]
                    y_r += harmonic[1][k*5]
                    plt.plot(x_hn,y_hn,color='black',alpha=0.5)
                
                plt.axes().set_aspect('equal')
                plt.axes().tick_params(labelbottom=False,bottom=False);
                plt.axes().tick_params(labelleft=False,left=False);
                plt.axes().set(xlim=(min(x_star)-margin,max(x_star)+margin),ylim=(min(y_star)-margin,max(y_star)+margin))
                # matplotlib.pyplot.gca().set_ylim(min(self.N_list[-1][1])*1.4,max(self.N_list[-1][1])*1.4)
                plt.savefig(save_dir+os.sep+"NormalizedDATASET_EH"+os.sep+Photo+'_'+str(i+1)+'_'+str(k)+"_harm.pdf")
                plt.close()

    df_pre = pd.DataFrame({ 'FileName_' :Photo,
                            'DataCount_'   :np.arange(1,len(contours)+1)
                            })
    if len(HEADER) != 0:
        for i,header in enumerate(HEADER):
            df_pre[header] = grouplist[i]
    df_fps = pd.DataFrame(FPS_matrix,columns=FPS_HEADER)
    df_concat = pd.concat([df_pre,df_fps],axis=1)
    return df_concat,len_contours

########### SHAPE RECONSTRUCTOR #############

def fps(isFPS,pcscore,inv_rot,scale_mat,center_mat):
    fps_log = np.zeros(len(pcscore))
    fps = np.zeros(len(pcscore))
    fps_log = (pcscore.dot(inv_rot).dot(scale_mat)+center_mat)
    # fps_log = (pcscore.dot(inv_rot)+center_mat)
    if isFPS == 0: # FPS mode
        for i in range(0,len(pcscore)):
            if i == 0:
                fps[i] = fps_log[0,i]
            else:
                fps[i]=np.exp(-1*fps_log[0,i])
                # fps[i]=np.exp(fps_log[0,i])   ####koko
    elif isFPS == 2: # EFD mode
        for i in range(0,len(pcscore)):
            if i == 0:
                fps[i] = fps_log[0,i]
            else:
                fps[i] = fps_log[0,i]
    else:
         for i in range(0,len(pcscore)):
            if i == 0:
                fps[i] = fps_log[0,i]
            else:
                fps[i]=np.exp(-1*fps_log[0,i])

    return fps

def efdgenerator(fps):
    abcd=2.0*fps
    # determine the ratio between a^2 + b^2 and c^2 + d^2
    ab = random.random() * abcd
    cd = abcd - ab
    aa = random.random() * ab
    bb = ab - aa
    cc = random.random() * cd
    dd = cd - cc
    a = np.sqrt(aa)*(-1)**random.randint(1,2)
    b = np.sqrt(bb)*(-1)**random.randint(1,2)
    c = np.sqrt(cc)*(-1)**random.randint(1,2)
    d = np.sqrt(dd)*(-1)**random.randint(1,2)
    return (a,b,c,d)

def efdgenerator_amp(xam,yam):
    ab=2.0*xam
    cd=2.0*yam
    # determine the ratio between a^2 + b^2 and c^2 + d^2
    aa = random.random() * ab
    bb = ab - aa
    cc = random.random() * cd
    dd = cd - cc
    a = np.sqrt(aa)*(-1)**random.randint(1,2)
    b = np.sqrt(bb)*(-1)**random.randint(1,2)
    c = np.sqrt(cc)*(-1)**random.randint(1,2)
    d = np.sqrt(dd)*(-1)**random.randint(1,2)
    return (a,b,c,d)

def reconstContourCoord(N,fps,isFPS):
    T = 1.0
    t = np.arange(0,T,0.01)
    x_t = np.zeros(len(t)) #X-coordinate
    y_t = np.zeros(len(t)) #Y-coordinate
    efd_list=[]
    if isFPS == 0:
        for n in range(1,N+1):
            if n==1:
                efd = (1.00, 0.00, 0.00, np.sqrt(np.absolute(fps[n-1]*2-1.00)))
            else:
                efd = efdgenerator(fps[n-1])
            an = efd[0]
            bn = efd[1]
            cn = efd[2]
            dn = efd[3]
            efd_list.append(efd)
            x_t += an*np.cos(2*n*np.pi*t/T) + bn*np.sin(2*n*np.pi*t/T)
            y_t += cn*np.cos(2*n*np.pi*t/T) + dn*np.sin(2*n*np.pi*t/T)
        np.append(x_t,x_t[0])
        np.append(y_t,y_t[0])
    elif isFPS == 2:
        for n in range(1,N+1):
            if n==1:
                efd = (1.00, 0.00, 0.00, fps[3])
            else:
                efd = ( fps[4*(n-1)], fps[4*(n-1)+1], fps[4*(n-1)+2], fps[4*(n-1)+3] )
            an = efd[0]
            bn = efd[1]
            cn = efd[2]
            dn = efd[3]
            efd_list.append(efd)
            x_t += an*np.cos(2*n*np.pi*t/T) + bn*np.sin(2*n*np.pi*t/T)
            y_t += cn*np.cos(2*n*np.pi*t/T) + dn*np.sin(2*n*np.pi*t/T)
        np.append(x_t,x_t[0])
        np.append(y_t,y_t[0])
    else:
        for n in range(1,N+1):
            if n==1:
                efd = (1.00, 0.00, 0.00, np.sqrt(np.absolute(fps[1]*2)))
            else:
                efd = efdgenerator_amp(fps[2*(n-1)],fps[2*(n-1)+1])
            an = efd[0]
            bn = efd[1]
            cn = efd[2]
            dn = efd[3]
            efd_list.append(efd)
            x_t += an*np.cos(2*n*np.pi*t/T) + bn*np.sin(2*n*np.pi*t/T)
            y_t += cn*np.cos(2*n*np.pi*t/T) + dn*np.sin(2*n*np.pi*t/T)
        np.append(x_t,x_t[0])
        np.append(y_t,y_t[0])
    return x_t, y_t, efd_list


def find_index(lst, value):
    for index, element in enumerate(lst):
        if element == value:
            return index
    return -1  

def create_random_contour(N: int, indexes: list,factors: list,base_factor: float=100,base_amp_ratio: float=100, isFPS:int = 0):
    
    fps = np.random.rand(N)
    for i, value in enumerate(fps):
        if i in indexes:
            ind_factor = find_index(indexes,i)
            if i == 0:
                fps[i] = 1 + factors[ind_factor]*np.random.rand()
            else:
                fps[i] = factors[ind_factor]*np.random.rand()
        else:
            fps[i] = fps[i]/(base_amp_ratio*np.power(10,base_factor*np.random.rand()))
    x, y, t = reconstContourCoord(N,fps,isFPS)
    return x, y, t

def grid(row: int,col: int, x_interval: float,y_interval: float) -> tuple:
    grid_x = (np.arange(row*col) % col) * x_interval
    grid_x_2d = np.reshape(grid_x,(row,col))
    
    grid_y = np.ones((row,col))
    yrow = np.arange(row) * y_interval
    grid_y_2d = grid_y.T * yrow
    grid_y_2d = grid_y_2d.T
    
    return (grid_x_2d, grid_y_2d)

def conductPCA_correlation(csv_path,isFPS,isCor):
    df = pd.read_csv(csv_path)
    # Normalization: case for Correlation matrix
    FPS_loc = df.columns.get_loc('FPS1')
    A1_loc = df.columns.get_loc('a1')
    N = int((len(df.columns) - FPS_loc)/7.0)
    # take log FPS2–N 
    df.iloc[:,( FPS_loc + 1 ): (FPS_loc + N )] = -1*np.log(df.iloc[:,( FPS_loc + 1 ):(FPS_loc + N )])
    df.iloc[:,( FPS_loc + 5*N + 1 ):] = -1*np.log(df.iloc[:,( FPS_loc + 5*N + 1 ):])
    # df.iloc[:,( FPS_loc + 1 ): (FPS_loc + N )] = np.log(df.iloc[:,( FPS_loc + 1 ):(FPS_loc + N )])  #### koko
    # df.iloc[:,( FPS_loc + N + 4 ):] = -1*np.log(1+df.iloc[:,( FPS_loc + N + 4 ):])
    if isFPS == 0: # FPS
        scale_array = np.std(df.iloc[:,(FPS_loc):(FPS_loc + N )])
        center_array= np.mean(df.iloc[:,(FPS_loc):(FPS_loc + N )])
    elif isFPS == 2: # EFD
        scale_array = np.std(df.iloc[:,(FPS_loc + N):(FPS_loc + 5*N)])
        center_array= np.mean(df.iloc[:,(FPS_loc + N):(FPS_loc + 5*N)])
    else: # AMp
        scale_array = np.std(df.iloc[:,(FPS_loc + 5*N):])
        center_array= np.mean(df.iloc[:,(FPS_loc + 5*N):])
    # Correlation Matrix
    # print(isFPS)
    if isFPS == 0:
        if isCor:
            dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean())/x.std(), axis=0)
        else:
            dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean()), axis=0)
    elif isFPS == 2: #EFD
        if isCor:
            dfs = df.iloc[:,FPS_loc + N:FPS_loc + 5*N].apply(lambda x: (x-x.mean())/x.std(), axis=0)
        else:
            dfs = df.iloc[:,FPS_loc + N:FPS_loc + 5*N].apply(lambda x: (x-x.mean()), axis=0)
    else:
        if isCor:
            dfs = df.iloc[:,FPS_loc + 5*N:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
        else:
            dfs = df.iloc[:,FPS_loc + 5*N:].apply(lambda x: (x-x.mean()), axis=0)
    #PCA
    pca = PCA()
    feature = pca.fit(dfs)
    feature = pca.transform(dfs)
    dfs.to_csv("test.csv")
    PC_SCORE = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    result_df = pd.concat([df,PC_SCORE],axis=1)

    #Contribute Rate
    contribution = pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    #EigenValue
    eigenvalues = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    #EigenVector Rotation
    if isFPS == 0: # FPS
        csv_rot = pd.DataFrame(pca.components_, columns=df.columns[FPS_loc:(FPS_loc + N )], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    elif isFPS == 2: # EFD
        csv_rot = pd.DataFrame(pca.components_, columns=df.columns[(FPS_loc + N):(FPS_loc + 5*N)], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    else:
        csv_rot = pd.DataFrame(pca.components_, columns=df.columns[(FPS_loc + 5*N):], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    
    # Standard Dev
    stdv_array = np.std(PC_SCORE)
    #Rotation Mat
    csv_rot = csv_rot.T
    rot_data = np.array(csv_rot.values.flatten())
    rot_array = np.reshape(rot_data,(csv_rot.shape[0],csv_rot.shape[1]))
    rot_mat = np.matrix(rot_array).astype(np.float64)
    inv_rot = np.linalg.inv(rot_mat) #inverse matrix of the rotation.
    #Scale Mat
    scale_data = np.array(scale_array.values.flatten())
    #Center Mat
    center_data = np.array(center_array.values.flatten())

    if isFPS == 0: # FPS    
        scale_mat = np.diag(np.reshape(scale_data.astype(np.float64),N))
        center_mat = np.reshape(center_data.astype(np.float64),N)
    elif isFPS == 2: # EFD
        scale_mat = np.diag(np.reshape(scale_data.astype(np.float64),4*N))
        center_mat = np.reshape(center_data.astype(np.float64),4*N)
    else:
        scale_mat = np.diag(np.reshape(scale_data.astype(np.float64),2*N))
        center_mat = np.reshape(center_data.astype(np.float64),2*N)
    #Stdv array
    stdv_array = np.array(stdv_array.values.flatten())
    cont = contribution
    cont['Cum.'] = np.cumsum(pca.explained_variance_ratio_)
    cont['Dev.'] = stdv_array
    cont.columns = ["Cont.","Cum.","Dev."]

    return result_df,cont,eigenvalues,csv_rot,scale_mat,center_mat,stdv_array,inv_rot,N

def generate_cmap(colors):
    """Return original color maps"""
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def kde2dgraphfill(ax,x,y,xmin,xmax,ymin,ymax):
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    noise_param = 1.0
    try:
        kernel = st.gaussian_kde(values)
    except np.linalg.linalg.LinAlgError:
        row = len(x)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        dev_x = np.std(x)
        dev_y = np.std(y)
        noise_param = mean_y + mean_x + dev_y + dev_x
        if noise_param == 0:
            noise_param = 1
        elif noise_param > 1:
            # noise_param = 1.0 / noise_param
            noise_param = noise_param / 2
        elif noise_param < 0:
            noise_param = -1*noise_param
        else:
            noise_param = noise_param
        ####################
        # CHECK HERE !!!!!!!
        ####################
        rd_array = np.random.rand(row) * noise_param * 1/1000
        x_dash = x + rd_array
        values = np.vstack([x_dash, y])
        kernel = st.gaussian_kde(values)
    # kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # cm = generate_cmap(['aqua', 'lawngreen', 'yellow', 'coral'])
    # cm = generate_cmap(['ghostwhite', 'deepskyblue', 'mediumblue', 'darkblue'])
    cfset = ax.contourf(xx, yy, f, cmap=plt.cm.jet)
    cfset = ax.contourf(xx, yy, f, cmap="jet")
    # cfset = ax.contourf(xx, yy, f, cmap=cm)
    cset = ax.contour(xx, yy, f, colors='k')
    return cfset
    
def kde2dgraph(ax,x,y,xmin,xmax,ymin,ymax,cmap):
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    noise_param = 1.0
    try:
        kernel = st.gaussian_kde(values)
    except np.linalg.linalg.LinAlgError:
        row = len(x)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        dev_x = np.std(x)
        dev_y = np.std(y)
        noise_param = mean_y + mean_x + dev_y + dev_x
        if noise_param == 0:
            noise_param = 1
        elif noise_param > 1:
            # noise_param = 1.0 / noise_param
            noise_param = noise_param / 2
        elif noise_param < 0:
            noise_param = -1*noise_param
        else:
            noise_param = noise_param
        rd_array = np.random.rand(row) * noise_param * 1/1000
        x_dash = x + rd_array
        values = np.vstack([x_dash, y])
        kernel = st.gaussian_kde(values) 
    # except ValueError:
        # return "valueError"

    # kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # cfset = ax.contourf(xx, yy, f, cmap=cmap)
    cset = ax.contour(xx, yy, f, cmap=cmap)
    return cset
    
