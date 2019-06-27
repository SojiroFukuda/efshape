import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap

import os
import sklearn
from sklearn.decomposition import PCA
import random
import scipy.stats as st
# print("x=",contours[0][0][0][0])
# print("y=",contours[0][0][0][1])

print("OpenCV: " + cv2.__version__)

#cnt = [ (x0,y0),(x1,y1),......]
def convShape2func(cnt):
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

#function that returns the traditional EFDs
def efd(T,x_p,y_p,t_p,dt,n):
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
    plt.savefig(save_dir+os.sep+"contdir"+os.sep+file_name+"_c.png")
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
    draw_contours(gray,contours,file,save_dir)
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
    os.makedirs(save_dir+os.sep+"NormalizedDATASET",exist_ok=True)
    for i,cnt in enumerate(contours):
        #Area, Perimeter
        FPS_matrix[i][INDEX_PIXEL] = cv2.contourArea(cnt) #Pixel Area
        FPS_matrix[i][INDEX_AREA] = cv2.contourArea(cnt) * unit_conv # scaled Area
        FPS_matrix[i][INDEX_PHI] = -1*np.log2(cv2.contourArea(cnt) * unit_conv) # phi
        FPS_matrix[i][INDEX_PSI] = np.log2(cv2.contourArea(cnt) * unit_conv) # Psi
        FPS_matrix[i][INDEX_PERIM] = cv2.arcLength(cnt, True) * np.sqrt(unit_conv) # Perim
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
            an, bn, cn, dn = efd(T,x_p,y_p,t,delt,j+1)
            efd_list.append(np.array([an,bn,cn,dn]))
            #Reconstruction
            x_r += an*np.cos(2*(j+1)*np.pi*t/T) + bn*np.sin(2*(j+1)*np.pi*t/T)
            y_r += cn*np.cos(2*(j+1)*np.pi*t/T) + dn*np.sin(2*(j+1)*np.pi*t/T)
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

def conductPCA_correlation(csv_path,isFPS,isCor):
    df = pd.read_csv(csv_path)
    # Normalization: case for Correlation matrix
    FPS_loc = df.columns.get_loc('FPS1')
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
    print(isFPS)
    if isFPS == 0:
        if isCor:
            dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean())/x.std(), axis=0)
        else:
            dfs = df.iloc[:,FPS_loc:(FPS_loc + N )].apply(lambda x: (x-x.mean()), axis=0)
    elif isFPS == 2:
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
    