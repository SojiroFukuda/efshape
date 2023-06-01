import sys
from PyQt6 import QtCore as Qc, QtGui as Qg, QtWidgets as Qw   
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib
matplotlib.use("agg")
import os
import re
import glob
import numpy as np
from . import efa as ef
# import efa as ef
import pandas as pd
from . import fgui as fpsGui 
# import fgui as fpsGui                           
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure
from pathlib import Path
# from PyQt6.Qt import PYQT_VERSION_STR
from datetime import datetime
import matplotlib.animation as animation


# Display pandas dataframe at QTableWidget
class PandasModel(Qc.QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None): 
        Qc.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return Qc.QVariant()

        if orientation == Qt.Orientation.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return Qc.QVariant()
        elif orientation == Qt.Orientation.Vertical:
            try:
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return Qc.QVariant()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return Qc.QVariant()

        if not index.isValid():
            return Qc.QVariant()

        return Qc.QVariant(str(self._df.iloc[index.row(), index.column()])) 

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=Qc.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=Qc.QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

# Conduct FPS with an independent thread
class FourierAnalyzer(QThread):
    countChanged = pyqtSignal(int)
    stringChanged = pyqtSignal(str)
    stringChangedPCA = pyqtSignal(str)
    FourierDone = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.w = Qw.QDialog(parent)
        self.parent = parent

    def run(self):
        #Progress Bar----------
        flag = 0
        PROGRESS_LIMIT = len(self.parent.FILE_LIST)
        PROGRESS_VALUE = 0
        #----------------------
        TOTAL_CLAST = 0
        for file in self.parent.FILE_LIST:
            # Get FilePath of Target Image
            separator_list = ['_',os.sep,'_']
            separator = separator_list[ self.parent.categorization ]
            grouplist = ef.group_list(file,separator)
            im_path = self.parent.FOLDA_DIR+os.sep+file+self.parent.FILE_FORMAT
            FILE_NAME = file + self.parent.FILE_FORMAT
            # Log for File Load -----------------
            self.stringChanged.emit(file+" is loaded"+"\n")
            # self.stringChanged.emit(" ")
            # -----------------------------------
            if flag == 0: # Generate CSV FILE at first
                fps,total_num = ef.FPS_calc(im_path,self.parent.ui.cb_saveAll.isChecked(),self.parent.SAVE_DIR,file,grouplist,self.parent.HEADER_LIST,self.parent.BGC,self.parent.delt,self.parent.N,self.parent.MIN_PIXEL_SIZE,self.parent.SCALE_TYPE,self.parent.SCALE_UNIT,self.parent.SCALE_VALUE,self.parent.SCALE_POSITION)
                fps.to_csv(self.parent.SAVE_PATH, header=True, index=False)
                flag = flag + 1
                # Log #--------------
                self.stringChanged.emit("done ("+str(total_num)+" contours)"+"\n"+str( len(self.parent.FILE_LIST) - flag )+" files left"+"\n")
                # -------------------
                PROGRESS_VALUE = PROGRESS_VALUE + 1
                self.countChanged.emit(PROGRESS_VALUE)
                TOTAL_CLAST = TOTAL_CLAST + total_num
            else: # Add data to 
                flag = flag + 1
                fps,total_num = ef.FPS_calc(im_path,self.parent.ui.cb_saveAll.isChecked(),self.parent.SAVE_DIR,file,grouplist,self.parent.HEADER_LIST,self.parent.BGC,self.parent.delt,self.parent.N,self.parent.MIN_PIXEL_SIZE,self.parent.SCALE_TYPE,self.parent.SCALE_UNIT,self.parent.SCALE_VALUE,self.parent.SCALE_POSITION)
                fps.to_csv(self.parent.SAVE_PATH, mode='a', header=False, index=False)  
                # Log #--------------
                self.stringChanged.emit("Completed ("+str(total_num)+" contours)"+"\n"+str( len(self.parent.FILE_LIST) - flag )+" files left"+"\n")
                # -------------------
                PROGRESS_VALUE = PROGRESS_VALUE + 1
                self.countChanged.emit(PROGRESS_VALUE)
                TOTAL_CLAST = TOTAL_CLAST + total_num
        # Log #--------------
        self.stringChanged.emit("\n"+"------------------ end."+"\n"+"In total "+str(TOTAL_CLAST)+" grains were detected"+"\n")
        # -------------------
        #######################################################
        ###########   CHECK THIS ##############################
        ######## Causative of segmentation error ##############
        self.FourierDone.emit(self.parent.SAVE_PATH)
        # self.parent.ui.textbox_fpsPath.setText(self.parent.SAVE_PATH) 
        # self.parent.FPS_PATH = self.parent.SAVE_PATH
        #######################################################
        ######################################################
        
        # self.parent.SAVE_PCA = self.parent.SAVE_PATH
        # self.parent.ui.textbox_pcaSave.setText(self.parent.SAVE_PCA)
# Conduct FPS with an independent thread
class GraphAnimation(QThread):
    graphRenew = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.w = Qw.QDialog(parent)
        self.parent = parent

    def run(self):
        self.parent.ui.button_startsample.setEnabled(False)
        # plot for y-projection
        im_shape, = self.parent.ax_py.plot([],[],color='blue') # original sample
        im_px, = self.parent.ax_py.plot([],[],color='orange') # projected function of x coordinate
        im_dline, = self.parent.ax_py.plot([],[],linestyle='dashed',color="#E8846D",alpha=0.5) # a line 
        im_circ_left, = self.parent.ax_py.plot([],[],color='red') # moving point along the contour
        im_circ_right, = self.parent.ax_py.plot([],[],color='red') # moving point along the function curve
        
        # plot for x-projection
        im_shape_x, = self.parent.ax_px.plot([],[],color='blue')
        im_px_x, = self.parent.ax_px.plot([],[],color='orange')
        im_dline_x, = self.parent.ax_px.plot([],[],linestyle='dashed',color="#E8846D",alpha=0.5)
        im_circ_left_x, = self.parent.ax_px.plot([],[],color='red')
        im_circ_right_x, = self.parent.ax_px.plot([],[],color='red')

        self.parent.ax_py.set_aspect('equal')
        self.parent.ax_px.set_aspect('equal')
        
        width = max( [max(self.parent.x_pap),max(self.parent.y_pap) ] ) # half length of long axis
        self.parent.ax_py.set_xlim(min([min(self.parent.x_pap),min(self.parent.y_pap)])*1.1,width*1.6+max(self.parent.tsample*4))
        self.parent.ax_py.set_ylim(min([min(self.parent.x_pap),min(self.parent.y_pap)])*1.1,width*1.1)
        self.parent.ax_px.set_xlim(min([min(self.parent.x_pap),min(self.parent.y_pap)])*1.1,width*1.6+max(self.parent.tsample*4))
        self.parent.ax_px.set_ylim(min([min(self.parent.x_pap),min(self.parent.y_pap)])*1.1,width*1.1)
        self.parent.ax_py.plot(np.linspace(0,max(self.parent.tsample)*4,50)+width*1.5,np.zeros(50),color='black')
        self.parent.ax_px.plot(np.linspace(0,max(self.parent.tsample)*4,50)+width*1.5,np.zeros(50),color='black')

        self.parent.ax_py.plot(np.zeros(50)+width*1.5,np.linspace(-1*width,width,50),color='black')
        self.parent.ax_px.plot(np.zeros(50)+width*1.5,np.linspace(-1*width,width,50),color='black')

        self.parent.ax_py.fill(self.parent.x_pap,self.parent.y_pap,color='#6DBBE8')
        self.parent.ax_px.fill(self.parent.y_pap*-1,self.parent.x_pap,color='#6DBBE8')

        self.parent.ax_py.text(.41,.9,r"$Y$",horizontalalignment='center',transform=self.parent.ax_py.transAxes)
        self.parent.ax_px.text(.41,.9,r"$X$",horizontalalignment='center',transform=self.parent.ax_py.transAxes)
        for i in range(len(self.parent.tsample)):
            # x
            t_long = self.parent.tsample*4
            t_it = i%len(self.parent.tsample)
            x1 = self.parent.y_pap*-1
            y1 = self.parent.x_pap
            x2 = t_long[0:t_it]+width*1.5
            y2 = np.copy(self.parent.x_pap[0:t_it])
            x3 = np.linspace(self.parent.y_pap[t_it]*-1,t_long[t_it]+width*1.5,100)
            y3 = np.ones(100)*self.parent.x_pap[t_it]
            x4 = self.parent.y_pap[t_it]*(-1)+np.cos(np.linspace(0,2*np.pi,100))/20
            y4 = self.parent.x_pap[t_it]+np.sin(np.linspace(0,2*np.pi,100))/20
            x5 = t_long[t_it]+width*1.5+np.cos(np.linspace(0,2*np.pi,100))/20
            y5 = self.parent.x_pap[t_it]+np.sin(np.linspace(0,2*np.pi,100))/20
            # im_shape_x.set_data(x1,y1)
            im_px_x.set_data(x2,y2)
            im_dline_x.set_data(x3,y3)
            im_circ_left_x.set_data(x4,y4)
            im_circ_right_x.set_data(x5,y5)

            t_long = self.parent.tsample*4
            t_it = i%len(self.parent.tsample)
            x12 = self.parent.x_pap
            y12 = self.parent.y_pap
            x22 = t_long[0:t_it]+width*1.5
            y22 = np.copy(self.parent.y_pap[0:t_it])
            x32 = np.linspace(self.parent.x_pap[t_it],t_long[t_it]+width*1.5,100)
            y32 = np.ones(100)*self.parent.y_pap[t_it]
            x42 = self.parent.x_pap[t_it]+np.cos(np.linspace(0,2*np.pi,100))/20
            y42 = self.parent.y_pap[t_it]+np.sin(np.linspace(0,2*np.pi,100))/20
            x52 = t_long[t_it]+width*1.5+np.cos(np.linspace(0,2*np.pi,100))/20
            y52 = self.parent.y_pap[t_it]+np.sin(np.linspace(0,2*np.pi,100))/20
            # im_shape.set_data(x12,y12)
            im_px.set_data(x22,y22)
            im_dline.set_data(x32,y32)
            im_circ_left.set_data(x42,y42)
            im_circ_right.set_data(x52,y52)
            
            self.graphRenew.emit(i)
            matplotlib.pyplot.pause(0.02)
        self.parent.ui.button_startsample.setEnabled(True)

class GraphAnimation_Harmonics(QThread):
    graphRenew = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.w = Qw.QDialog(parent)
        self.parent = parent

    def run(self):
        speed = 1
        self.parent.ui.button_startsample.setEnabled(False)
        N = self.parent.ui.sb_mh_sample.value() # Maximum Harmonics
        # plot for y-projection
        harmonics_list = []
        radius_list = []
        point_list = []
        for i in range(N):
            harmonics_list.append( self.parent.ax_rre.plot([],[],color='blue',alpha=0.5)[0] )
            radius_list.append( self.parent.ax_rre.plot([],[],color='red')[0] )
            point_list.append( self.parent.ax_rre.plot([],[],color='red')[0] )
        im_shape, = self.parent.ax_rre.plot([],[],color='black') # original sample
        self.parent.ax_rre.set_aspect('equal')
        margin = max( self.parent.x_Na *0.4 )
        self.parent.ax_rre.set(xlim=(min(self.parent.x_Na)-margin,max(self.parent.x_Na)+margin),ylim=(min(self.parent.y_Na)-margin,max(self.parent.y_Na)+margin))
        self.parent.ax_rre.plot(self.parent.N_list[-1][0],self.parent.N_list[-1][1],color='black')
        
        margin = max( self.parent.N_list[-1][0]*0.4 )
        for i in range(0,int(len(self.parent.tsample)/speed)):
            x_r = 0; y_r = 0;
            for j,harmonic in enumerate(self.parent.harmonics):
                radius_list[j].set_data([x_r,x_r + harmonic[0][i*speed]],[y_r , y_r + harmonic[1][i*speed] ] )
                x_hn = harmonic[0] + x_r
                y_hn = harmonic[1] + y_r
                x_r += harmonic[0][i*speed]
                y_r += harmonic[1][i*speed]
                harmonics_list[j].set_data(x_hn,y_hn)
                point_list[j].set_data(x_r,y_r)
                
            self.graphRenew.emit(i)
            matplotlib.pyplot.pause(0.02)
        self.parent.ui.button_startsample.setEnabled(True)


## Main Window #####
class MyForm(Qw.QMainWindow):               
    def __init__(self, parent=None):        
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.ui = fpsGui.Ui_MainWindow()  
        self.ui.setupUi(self)               
        self.FORMAT_LIST = [".png",".BMP",".jpeg",".jpg"]
        self.FILE_FORMAT = ".png"
        self.categorization_list = ["None","SubFolda","FileName"]
        self.categorization = 1 # 
        self.FOLDA_DIR = ""
        self.HEADER_LIST = []
        self.SAVE_DIR = ""
        self.SAVE_FILE_NAME = "FPS.csv"
        self.SAVE_PATH = ""
        self.BGC = 0 # 0->white 1-> Brack
        self.FILE_LIST = []
        self.delt = 0.001 # The inverse number of point along the contour
        self.N = 30 # Maximum number of harmonics
        self.MIN_PIXEL_SIZE = 200
        self.SCALE_TYPE = 0 # 0: None 1:Area 2: X-Length 3: Y-Length
        self.SCALE_UNIT = ""
        self.SCALE_VALUE = 1
        self.SCALE_POSITION = 0  #0: Top 1: Bottom 2: right 3: left
        
        # EFA threading
        self.fa = FourierAnalyzer(self)
        self.fa.countChanged.connect(self.onCountChanged)
        self.fa.stringChanged.connect(self.onStringChanged)
        self.fa.stringChangedPCA.connect(self.onStringChangedPCA)
        self.fa.FourierDone.connect(self.onFourierDone)
        
        self.ui.cmb_format.currentIndexChanged.connect(self.setFormat) # type: ignore
        self.ui.button_folda.clicked.connect(self.setFoldaPath) # type: ignore
        self.ui.button_save.clicked.connect(self.setSavePath) # type: ignore
        self.ui.cmb_sampleType.currentIndexChanged.connect(self.setBGC) # type: ignore
        self.ui.spb_N.valueChanged.connect(self.setN) # type: ignore
        self.ui.spb_mpxa.valueChanged.connect(self.setMPA) # type: ignore
        self.ui.button_FPS.clicked.connect(self.startEFA) # type: ignore
        self.ui.cmb_scaleType.currentIndexChanged.connect(self.setScaleType) # type: ignore
        self.ui.textBox_scaleU.textEdited.connect(self.setScaleUnit) # type: ignore
        self.ui.spb_scaleSize.valueChanged.connect(self.setScaleValue) # type: ignore
        self.ui.cmb_scalePos.currentIndexChanged.connect(self.setScalePosition) # type: ignore
        self.ui.button_header.clicked.connect(self.setHeaderName) # type: ignore
        self.ui.button_FPSpath.clicked.connect(self.setFPSPath) # type: ignore
        self.ui.button_savePCApath.clicked.connect(self.setSavePCAPath) # type: ignore
        self.ui.cmb_matrixPCA.currentIndexChanged.connect(self.setPCAmatrix) # type: ignore
        self.ui.button_PCA.clicked.connect(self.startPCA) # type: ignore
        self.ui.cmb_x.currentIndexChanged.connect(self.setXaxis) # type: ignore
        self.ui.cmb_y.currentIndexChanged.connect(self.setYaxis) # type: ignore
        self.ui.cmb_color.currentIndexChanged.connect(self.setColor) # type: ignore
        self.ui.pushButton.clicked.connect(self.graphDraw) # type: ignore
        self.ui.pushButton_2.clicked.connect(self.savePCFig) # type: ignore
        self.ui.cmb_sort.currentIndexChanged.connect(self.setSort) # type: ignore
        self.ui.cmb_plot.currentIndexChanged.connect(self.setPlotType) # type: ignore
        self.ui.cmb_recType.currentIndexChanged.connect(self.setReconstMode) # type: ignore
        self.ui.cmb_sumDev.currentIndexChanged.connect(self.setSumDev) # type: ignore
        self.ui.cmb_IP.currentIndexChanged.connect(self.setIPCaxis) # type: ignore
        self.ui.cmb_IIP.currentIndexChanged.connect(self.setIIPCaxis) # type: ignore
        self.ui.cmb_IIIP.currentIndexChanged.connect(self.setIIIPCaxis) # type: ignore
        self.ui.dsb_Id.valueChanged.connect(self.setIdev) # type: ignore
        self.ui.dsb_IId.valueChanged.connect(self.setIIdev) # type: ignore
        self.ui.dsb_IIId.valueChanged.connect(self.setIIIdev) # type: ignore
        self.ui.button_generate.clicked.connect(self.reconstGraph) # type: ignore
        self.ui.button_saveReconst.clicked.connect(self.saveReconstGraph) # type: ignore
        self.ui.cmb_FPS.currentIndexChanged.connect(self.setFPSorEFD) # type: ignore
        self.ui.cmb_subcolor.currentIndexChanged.connect(self.setSubColor) # type: ignore
        self.ui.cmb_subsort.currentIndexChanged.connect(self.setSubSort) # type: ignore
        self.ui.cmb_subsubcolor.currentIndexChanged.connect(self.setSubsubColor) # type: ignore
        self.ui.cmb_subsubsort.currentIndexChanged.connect(self.setSubsubSort) # type: ignore
        self.ui.rb_none.toggled.connect(self.setCategorize) # type: ignore
        self.ui.rb_subfolda.toggled.connect(self.setCategorize) # type: ignore
        self.ui.rb_filename.toggled.connect(self.setCategorize) # type: ignore
        self.ui.button_sample.clicked.connect(self.setSampleImage) # type: ignore
        self.ui.button_startsample.clicked.connect(self.startPlayground) # type: ignore
        self.ui.button_save_sample.clicked.connect(self.saveSample) # type: ignore
        self.setWindowTitle('Grain Shape analyzer')
        #PCA PAGE
        self.PCA_METHOD = ["Fourier Power Spectra","Amplitudes of X and Y ellipses","Elliptic Fourier Descriptors"]
        self.isFPS = 0 # 0:FPS 1:amplitude 2:EFD
        self.FPS_PATH = ""
        self.SAVE_PCA = ""
        self.isCorrelationMatrix = True
        #PC Graph
        # self.isAlreadyGraph = False
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canv = FigureCanvas(self.fig)
        self.canv.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding)
        self.canv.updateGeometry()
        self.layout = Qw.QGridLayout(self.ui.canvas)
        self.layout.addWidget(self.canv)
        #Reconst shape
        self.isPCAdone = False
        self.isSummaryMODE = True
        self.sumdev = 1.0
        self.inv_rot = np.ndarray([])
        self.scale_mat = np.ndarray([])
        self.stdv_array = np.ndarray([])
        self.PLOT_TYPE = 0

        # The gridview for reconstructed shape figures
        self.layout11 = Qw.QGridLayout(self.ui.fg_1_1);
        # Setting 
        self.fig11 = Figure(); self.fig12 = Figure(); self.fig13 = Figure();
        self.fig21 = Figure(); self.fig22 = Figure(); self.fig23 = Figure();
        self.fig31 = Figure(); self.fig32 = Figure(); self.fig33 = Figure();
        self.ax11 = self.fig11.add_subplot(111); self.ax12 = self.fig12.add_subplot(111); self.ax13 = self.fig13.add_subplot(111);
        self.ax21 = self.fig21.add_subplot(111); self.ax22 = self.fig22.add_subplot(111); self.ax23 = self.fig23.add_subplot(111);
        self.ax31 = self.fig31.add_subplot(111); self.ax32 = self.fig32.add_subplot(111); self.ax33 = self.fig33.add_subplot(111);
        self.ax11.tick_params(labelbottom=False,bottom=False); self.ax12.tick_params(labelbottom=False,bottom=False); self.ax13.tick_params(labelbottom=False,bottom=False);
        self.ax21.tick_params(labelbottom=False,bottom=False); self.ax22.tick_params(labelbottom=False,bottom=False); self.ax23.tick_params(labelbottom=False,bottom=False);
        self.ax31.tick_params(labelbottom=False,bottom=False); self.ax32.tick_params(labelbottom=False,bottom=False); self.ax33.tick_params(labelbottom=False,bottom=False);
        self.ax11.tick_params(labelleft=False,left=False); self.ax12.tick_params(labelleft=False,left=False); self.ax13.tick_params(labelleft=False,left=False);
        self.ax21.tick_params(labelleft=False,left=False); self.ax22.tick_params(labelleft=False,left=False); self.ax23.tick_params(labelleft=False,left=False);
        self.ax31.tick_params(labelleft=False,left=False); self.ax32.tick_params(labelleft=False,left=False); self.ax33.tick_params(labelleft=False,left=False);
        self.ax11.set_xticklabels([]); self.ax12.set_xticklabels([]); self.ax13.set_xticklabels([]); 
        self.ax21.set_xticklabels([]); self.ax22.set_xticklabels([]); self.ax23.set_xticklabels([]); 
        self.ax31.set_xticklabels([]); self.ax32.set_xticklabels([]); self.ax33.set_xticklabels([]); 
        self.canv11 = FigureCanvas(self.fig11); self.canv12 = FigureCanvas(self.fig12); self.canv13 = FigureCanvas(self.fig13);
        self.canv21 = FigureCanvas(self.fig21); self.canv22 = FigureCanvas(self.fig22); self.canv23 = FigureCanvas(self.fig23);
        self.canv31 = FigureCanvas(self.fig31); self.canv32 = FigureCanvas(self.fig32); self.canv33 = FigureCanvas(self.fig33);
        self.canv11.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding); self.canv12.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding); self.canv13.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding); 
        self.canv21.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding); self.canv22.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding); self.canv23.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding); 
        self.canv31.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding); self.canv32.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding); self.canv33.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding); 
        self.canv11.updateGeometry(); self.canv12.updateGeometry(); self.canv13.updateGeometry();
        self.canv21.updateGeometry(); self.canv22.updateGeometry(); self.canv23.updateGeometry();
        self.canv31.updateGeometry(); self.canv32.updateGeometry(); self.canv33.updateGeometry();
        # Add figures to parent view
        self.layout11.addWidget(self.canv11,0,0); self.layout11.addWidget(self.canv12,0,1); self.layout11.addWidget(self.canv13,0,2); 
        self.layout11.addWidget(self.canv21,1,0); self.layout11.addWidget(self.canv22,1,1); self.layout11.addWidget(self.canv23,1,2); 
        self.layout11.addWidget(self.canv31,2,0); self.layout11.addWidget(self.canv32,2,1); self.layout11.addWidget(self.canv33,2,2); 
        

        #-- Playground --------
        self.ga = GraphAnimation(self)
        self.ga.graphRenew.connect(self.projection)
        self.gah = GraphAnimation_Harmonics(self)
        self.gah.graphRenew.connect(self.projection_h)

        self.isSampleMODE = False
        self.sample_path = ''
        self.layout_ori = Qw.QGridLayout(self.ui.gv_original);
        # selected sample
        self.layout_ss = Qw.QGridLayout(self.ui.f_selected);
        self.figss = Figure();
        self.figss.subplots_adjust(left=0,right=1,bottom=0,top=1)
        self.axss = self.figss.add_subplot(111);
        self.axss.tick_params(labelbottom=False,bottom=False);
        self.axss.tick_params(labelleft=False,left=False);
        self.axss.set_xticklabels([]);self.axss.set_yticklabels([]);
        self.canvss = FigureCanvas(self.figss);
        self.canvss.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding);
        self.canvss.updateGeometry();
        self.layout_ss.addWidget(self.canvss,0,0);

        #-- Projection gif -----------------
        self.layout_px = Qw.QGridLayout(self.ui.f_proj_x);
        self.figpx = Figure();
        self.figpx.subplots_adjust(left=0,right=1,bottom=0,top=1)
        self.ax_px = self.figpx.add_subplot(111);
        self.ax_px.tick_params(labelbottom=False,bottom=False);
        self.ax_px.tick_params(labelleft=False,left=False);
        self.ax_px.set_xticklabels([]);self.ax_px.set_yticklabels([]);
        self.canvpx = FigureCanvas(self.figpx);
        self.canvpx.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding);
        self.canvpx.updateGeometry();
        self.layout_px.addWidget(self.canvpx,0,0);

        self.layout_py = Qw.QGridLayout(self.ui.f_proj_y);
        self.figpy = Figure();
        self.figpy.subplots_adjust(left=0,right=1,bottom=0,top=1)
        self.ax_py = self.figpy.add_subplot(111);
        self.ax_py.tick_params(labelbottom=False,bottom=False);
        self.ax_py.tick_params(labelleft=False,left=False);
        self.ax_py.set_xticklabels([]);self.ax_py.set_yticklabels([]);
        self.canvpy = FigureCanvas(self.figpy);
        self.canvpy.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding);
        self.canvpy.updateGeometry();
        self.layout_py.addWidget(self.canvpy,0,0);

        #-- X, Y, approximation view -------
        self.layout_x = Qw.QGridLayout(self.ui.f_x_sample);
        self.figx = Figure();
        self.ax_x = self.figx.add_subplot(111);
        self.canvx = FigureCanvas(self.figx);
        self.canvx.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding);
        self.canvx.updateGeometry();
        self.layout_x.addWidget(self.canvx,0,0);

        self.layout_y = Qw.QGridLayout(self.ui.f_y_sample);
        self.figy = Figure();
        self.ax_y = self.figy.add_subplot(111);
        self.canvy = FigureCanvas(self.figy);
        self.canvy.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding);
        self.canvy.updateGeometry();
        self.layout_y.addWidget(self.canvy,0,0);

        #-- Reconst -----
        self.layout_rre = Qw.QGridLayout(self.ui.f_r_recon);
        self.fig_rre = Figure();
        self.fig_rre.subplots_adjust(left=0,right=1,bottom=0)
        self.fig_rre.patch.set_visible(False)
        self.ax_rre = self.fig_rre.add_subplot(111);
        self.ax_rre.tick_params(labelbottom=False,bottom=False);
        self.ax_rre.tick_params(labelleft=False,left=False);
        # self.ax_rre.axis('off')
        self.ax_rre.set_xticklabels([]);self.ax_rre.set_yticklabels([]);
        self.canv_rre = FigureCanvas(self.fig_rre);
        self.canv_rre.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding);
        self.canv_rre.updateGeometry();
        self.layout_rre.addWidget(self.canv_rre,0,0);

        self.layout_rcom = Qw.QGridLayout(self.ui.f_r_comp);
        self.fig_rcom = Figure();
        self.fig_rcom.subplots_adjust(left=0,right=1,bottom=0)
        self.fig_rcom.patch.set_visible(False)
        self.ax_rcom = self.fig_rcom.add_subplot(111);
        self.ax_rcom.tick_params(labelbottom=False,bottom=False);
        self.ax_rcom.tick_params(labelleft=False,left=False);
        # self.ax_rcom.axis('off')
        self.ax_rcom.set_xticklabels([]);self.ax_rcom.set_yticklabels([]);
        self.canv_rcom = FigureCanvas(self.fig_rcom);
        self.canv_rcom.setSizePolicy(Qw.QSizePolicy.Policy.Expanding, Qw.QSizePolicy.Policy.Expanding);
        self.canv_rcom.updateGeometry();
        self.layout_rcom.addWidget(self.canv_rcom,0,0);
        self.ui.tabWidget_2.setCurrentIndex(1);


    def setSampleImage(self):
        path = self.openImageFile()
        if path == '':
            return
        self.sample_path = path
        gray,contours = ef.getValidContours(path,200,self.ui.cb_BGC_sample.currentIndex())
        self.ui.sb_snum_sample.setEnabled(True)
        self.ui.sb_snum_sample.setMinimum(1)
        self.ui.sb_snum_sample.setMaximum(len(contours))
        self.sample_contours = contours

    def openImageFile(self): # select csv file
        filePath, _ = Qw.QFileDialog.getOpenFileName(self, "Open File",
                self.sample_path, "Image file (*.png)")
        return filePath

    def openCSVFile(self): # select csv file
        filePath, _ = Qw.QFileDialog.getOpenFileName(self, "Open File",
                self.sample_path, "csv file (*.csv)")
        return filePath

    def startPlayground(self):
        if self.sample_path == '':
            return

        self.Nsample = self.ui.sb_mh_sample.value() # Maximum Harmonics
        self.sample_cnt = self.sample_contours[self.ui.sb_snum_sample.value()-1] # choose contour
        x_t,y_t = ef.adjustXYCoord(ef.getXYCoord(self.sample_cnt)[0],ef.getXYCoord(self.sample_cnt)[1]) # extract contour coordinates
        self.N_list,self.harmonics,self.harmonicsf,x_p,y_p,t = ef.fourierApproximation(self.sample_cnt,self.Nsample) # Reconstructed shape by N harmonics
        self.x_p_ori = np.copy(x_p) # Original x coordinate
        self.y_p_ori = np.copy(y_p) # Original y coordinate
        self.tsample = t
        t_a,x_Na = ef.adjustXYCoord(t,self.N_list[-1][0])
        t_a,x_pa = ef.adjustXYCoord(t,x_p)
        x_Na = -1*x_Na
        x_pa = -1*x_pa
        t_a,y_Na = ef.adjustXYCoord(t,self.N_list[-1][1])
        t_a,y_pa = ef.adjustXYCoord(t,y_p)
        y_Na = -1*y_Na
        y_pa = -1*y_pa
        self.x_pa_ori = x_pa
        self.y_pa_ori = y_pa
        self.x_Na = x_Na
        self.y_Na = y_Na
        

        #-- Sample Image view ---------
        pixmap = Qg.QPixmap(self.sample_path)
        self.scene = Qw.QGraphicsScene(self)
        item = Qw.QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.ui.gv_original.setScene(self.scene)
        
        
        #-- selected Sample view -------
        self.axss.clear();
        self.axss.set_aspect('equal', 'datalim');
        matplotlib.pyplot.xlim([min(x_t),max(x_t)]);
        matplotlib.pyplot.ylim([min(y_t),max(y_t)]);
        pd.DataFrame({"x":x_t,"y":y_t}).plot(kind="line",x="x", y="y", ax=self.axss,alpha=0);
        self.axss.fill(x_t,y_t,color='black',alpha=0.5);
        self.axss.set_xlabel("");
        self.axss.set_ylabel("");
        self.figss.canvas.draw_idle();
  
        #-- Fourier Approximation view ---------
        matplotlib.pyplot.close()
        self.ax_x.clear();
        # self.ax_x.set_aspect('equal', 'datalim');
        self.ax_x.set_xlim([min(t),max(t)]);
        self.ax_x.set_ylim([min(x_pa),max(x_pa)]);
        pd.DataFrame({"x":t,"y":x_Na}).plot(kind="line",x="x", y="y", ax=self.ax_x,color='red',alpha=0.5);
        pd.DataFrame({"x":t,"y":x_pa}).plot(kind="line",x="x", y="y", ax=self.ax_x,color='black',alpha=1);
        # label, title
        self.ax_x.set_xlabel("");
        self.ax_x.set_ylabel("");
        self.ax_x.set_title(r"Maximum Harmonics: ${\it N}\ =\ $"+str(self.Nsample))
        # Draw
        self.figx.canvas.draw_idle();
        
        matplotlib.pyplot.close()
        self.ax_y.clear();
        # self.ax_y.set_aspect('equal', 'datalim');
        self.ax_y.set_xlim([min(t),max(t)]);
        self.ax_y.set_ylim([min(y_pa),max(y_pa)]);
        pd.DataFrame({"x":t,"y":y_Na}).plot(kind="line",x="x", y="y", ax=self.ax_y,color='red',alpha=0.5);
        pd.DataFrame({"x":t,"y":y_pa}).plot(kind="line",x="x", y="y", ax=self.ax_y,color='black',alpha=1);
        # label title
        self.ax_y.set_xlabel("");
        self.ax_y.set_ylabel("");
        self.ax_y.set_title(r"Maximum Harmonics: ${\it N}\ =\ $"+str(self.Nsample))
        self.figy.canvas.draw_idle();

        #-- Reconstruct view ----------
        self.ax_rre.clear();
        self.ax_rre.set_aspect('equal', 'datalim');
        self.ax_rre.set_xlim([min(x_t),max(x_t)]);
        self.ax_rre.set_ylim([min(y_t),max(y_t)*1.3]);
        pd.DataFrame({"x":x_Na,"y":y_Na}).plot(kind="line",x="x", y="y", ax=self.ax_rre,alpha=1,color="#E8846D");
        self.ax_rre.fill(x_Na,y_Na,color='#E8846D',alpha=0.5);
        self.ax_rre.set_xlabel("");
        self.ax_rre.set_ylabel("");
        # self.ax_rre.set_title(r"Maximum Harmonics: ${\it N}\ =\ $"+str(self.Nsample))
        self.ax_rre.spines['top'].set_color('white')
        self.ax_rre.spines['bottom'].set_color('white')
        self.ax_rre.spines['left'].set_color('white')
        self.ax_rre.spines['right'].set_color('white')
        self.ax_rre.text(.5,.9,r"Maximum Harmonics: ${\it N}\ =\ $"+str(self.Nsample),horizontalalignment='center',transform=self.ax_rre.transAxes)
        self.fig_rre.canvas.draw_idle();

        self.ax_rcom.clear();
        self.ax_rcom.set_aspect('equal', 'datalim');
        self.ax_rcom.set_xlim([min(x_t),max(x_t)]);
        self.ax_rcom.set_ylim([min(y_t),max(y_t)*1.3]);
        pd.DataFrame({"x":x_pa,"y":y_pa}).plot(kind="line",x="x", y="y", ax=self.ax_rcom,alpha=0.5,color='#6DBBE8');
        pd.DataFrame({"x":x_Na,"y":y_Na}).plot(kind="line",x="x", y="y", ax=self.ax_rcom,alpha=1.0,color='red');
        self.ax_rcom.fill(x_pa,y_pa,color='#6DBBE8',alpha=0.5);
        self.ax_rcom.set_xlabel("");
        self.ax_rcom.set_ylabel("");
        # self.ax_rcom.set_title(r"Maximum Harmonics: ${\it N}\ =\ $"+str(self.Nsample))
        self.ax_rcom.spines['top'].set_color('white')
        self.ax_rcom.spines['bottom'].set_color('white')
        self.ax_rcom.spines['left'].set_color('white')
        self.ax_rcom.spines['right'].set_color('white')
        self.ax_rcom.text(.5,.9,r"Maximum Harmonics: ${\it N}\ =\ $"+str(self.Nsample),horizontalalignment='center',transform=self.ax_rre.transAxes)
        self.fig_rcom.canvas.draw_idle();
        self.isSampleMODE = True

        #-- Projection view --------------------
        self.width = max([max(x_pa),max(y_pa)])
        self.x_pap = x_pa/self.width
        self.y_pap = y_pa/self.width
        self.ax_py.clear()
        self.ax_px.clear()
        self.ax_rre.clear()
        if self.ui.tabWidget_2.currentIndex() == 1:
            self.ga.start()
        elif self.ui.tabWidget_2.currentIndex() == 3:
            self.gah.start()
    
    def projection(self,frame):
        self.figpx.canvas.draw_idle();
        self.figpy.canvas.draw_idle(); 
        
    def projection_h(self,frame):
        self.fig_rre.canvas.draw_idle();
        # self.figpy.canvas.draw_idle(); 

        

    def saveSample(self):
        if self.isSampleMODE == False:
            return
        rootpath = os.path.abspath(os.path.dirname("__file__"))
        FIG_FOLDA = Qw.QFileDialog.getExistingDirectory(None,"rootpath",rootpath)
        if FIG_FOLDA == "":
            return
        file_name = ""
        
        os.makedirs(FIG_FOLDA+os.sep+"Sample_Set",exist_ok=True)
        file_name = FIG_FOLDA+os.sep+"Sample_Set"+os.sep
        
        self.figss.savefig(file_name+"Selected_Sample.pdf",bbox_inches='tight')
        self.figx.savefig(file_name+"X_coord.pdf",bbox_inches='tight')
        self.figy.savefig(file_name+"y_coord.pdf",bbox_inches='tight')
        self.fig_rre.savefig(file_name+"Reconst.pdf")
        self.fig_rcom.savefig(file_name+"Comperison.pdf")
        try:
            self.figpx.savefig(file_name+"projection_X.pdf",bbox_inches='tight')
            self.figpy.savefig(file_name+"projection_Y.pdf",bbox_inches='tight')
        except:
            print('ERROR: Something wrong with your save folda')
            pass
        os.makedirs(FIG_FOLDA+os.sep+"Sample_Set"+os.sep+"sequence",exist_ok=True)
        os.makedirs(FIG_FOLDA+os.sep+"Sample_Set"+os.sep+"sequence"+os.sep+"comp",exist_ok=True)
        sequence_name = FIG_FOLDA+os.sep+"Sample_Set"+os.sep+"sequence"+os.sep
        sequence_c_name = FIG_FOLDA+os.sep+"Sample_Set"+os.sep+"sequence"+os.sep+"comp"+os.sep

        
        for i in range(self.Nsample):
            num = str(i+1)
            x_Na,y_Na = ef.adjustXYCoord(self.N_list[i][0],self.N_list[i][1])
            shifx = np.abs(max(self.x_pa_ori)-min(self.x_pa_ori))*1.2
            fig,ax_seq= matplotlib.pyplot.subplots()
            ax_seq.set_xlim(min(self.x_pa_ori)*1.1,max(self.x_pa_ori)*1.1+shifx)
            ax_seq.set_ylim(min(self.y_pa_ori)*1.1,max(self.y_pa_ori)*1.1)
            ax_seq.set_aspect('equal', 'datalim');
            ax_seq.plot(x_Na,-y_Na,linewidth=0.1,color='#E8846D',alpha=1)
            ax_seq.fill(x_Na,-y_Na,color='#E8846D',alpha=1)
            ax_seq.set_title(r"${\it N}\ =\ $"+num)
            ax_seq.plot(self.x_pa_ori+shifx,self.y_pa_ori,color='#6DBBE8',alpha=0.5)
            x_sNa = np.copy(x_Na+shifx)
            ax_seq.fill(self.x_pa_ori+shifx,self.y_pa_ori,color='#6DBBE8',alpha=1)
            ax_seq.set_title(r"${\it N}\ =\ $"+num)
            fig.savefig(sequence_name+"comp"+os.sep+"seq_"+num+"_c.pdf")
            matplotlib.pyplot.close()
            num = str(i+1)
            fig,ax_seq= matplotlib.pyplot.subplots()
            ax_seq.set_aspect('equal', 'datalim');
            ax_seq.plot(self.N_list[i][0],self.N_list[i][1],linewidth=0.1,color='black')
            ax_seq.fill(self.N_list[i][0],self.N_list[i][1],color='black',alpha=0.5)
            ax_seq.set_title(r"${\it N}\ =\ $"+num)
            fig.savefig(sequence_name+"seq_"+num+".pdf")
            matplotlib.pyplot.close()
    
    def getImagesFromFolder(self):
        self.FILE_LIST = []
        # Log #--------------
        File_log_str = "Folda Path: " + self.FOLDA_DIR +"\n"
        # -------------------
        if self.FOLDA_DIR=="":
            return
        else:
            # Log #--------------
            File_log_str += "#------------------" + "\n" + "Selected Files" + "\n" + "#------------------" + "\n"
            # -------------------
            for file in glob.glob(self.FOLDA_DIR+os.sep+"**",recursive=True):
                index = re.search(self.FILE_FORMAT,file)
                # iscontdir = re.search('contdir',file)
                if index:
                    self.FILE_LIST.append(file[len(self.FOLDA_DIR)+1:-1*len(self.FILE_FORMAT)])
                    # Log #--------------
                    File_log_str += "          • " + file[len(self.FOLDA_DIR)+1:-1*len(self.FILE_FORMAT)] + "\n"
                    # -------------------
            # Log #--------------
            File_log_str += "In total: " + str(len(self.FILE_LIST)) +" " + self.FILE_FORMAT + " files" + "\n"
            self.onStringChanged(File_log_str)
            # -------------------
    
    #### Set image format PNG, BMP, JPEG
    def setFormat(self):
        
        self.FILE_FORMAT = self.FORMAT_LIST[self.ui.cmb_format.currentIndex()]
        self.getImagesFromFolder()
        # self.FILE_LIST = [] # Initiate image file list for analysis

        # if self.FOLDA_DIR=="":
        #     # Log #--------------
        #     self.onStringChanged("Format: "+self.FILE_FORMAT+"\n"+"Select the Folda Directory"+"\n")
        #     # -------------------
        # else:
        #     files = os.listdir(self.FOLDA_DIR) # All File in selected directory 
        #     File_log_str = ""
        #     for file in files:
        #         index = re.search(self.FILE_FORMAT,file) # Search the image files with the selected format
        #         if index:
        #             self.FILE_LIST.append(file[0:-1*len(self.FILE_FORMAT)]) # add image to analyze
        #             # Log #--------------
        #             File_log_str += "          • " + file[0:-1*len(self.FILE_FORMAT)] + "\n"
        #             # -------------------
        #     # Log #--------------
        #     self.onStringChanged("#------------------"+"\n"+"Selected Files"+"\n"+"#------------------"+"\n"+File_log_str+"In total: " + str(len(self.FILE_LIST)) +" " + self.FILE_FORMAT + " files" + "\n")
        #     # -------------------

    #### Set categorization method #####
    def setCategorize(self):
        cgnone = self.ui.rb_none.isChecked()
        cgsubfolda = self.ui.rb_subfolda.isChecked()
        cgfilename = self.ui.rb_filename.isChecked()
        if cgnone:
            self.ui.line_header.setEnabled(False)
            self.ui.button_header.setEnabled(False)
            self.categorization = 0
        elif cgsubfolda:
            self.ui.line_header.setEnabled(True)
            self.ui.button_header.setEnabled(True)
            self.categorization = 1
        elif cgfilename:
            self.ui.line_header.setEnabled(True)
            self.ui.button_header.setEnabled(True)
            self.categorization = 2

    def setFoldaPath(self):
        # self.FILE_LIST = []
        # select Folda
        rootpath = os.path.abspath(os.path.dirname("__file__"))
        self.FOLDA_DIR = Qw.QFileDialog.getExistingDirectory(None,"rootpath",rootpath)
        # # Log #--------------
        # File_log_str = "Folda Path: " + self.FOLDA_DIR +"\n"
        # # -------------------
        self.ui.line_folda.setText(self.FOLDA_DIR)
        self.getImagesFromFolder()
        # if self.FOLDA_DIR=="":
        #     return
        # else:
        #     # Log #--------------
        #     File_log_str += "#------------------" + "\n" + "Selected Files" + "\n" + "#------------------" + "\n"
        #     # -------------------
        #     for file in glob.glob(self.FOLDA_DIR+os.sep+"**",recursive=True):
        #         index = re.search(self.FILE_FORMAT,file)
        #         # iscontdir = re.search('contdir',file)
        #         if index:
        #             self.FILE_LIST.append(file[len(self.FOLDA_DIR)+1:-1*len(self.FILE_FORMAT)])
        #             # Log #--------------
        #             File_log_str += "          • " + file[len(self.FOLDA_DIR)+1:-1*len(self.FILE_FORMAT)] + "\n"
        #             # -------------------
        #     # Log #--------------
        #     File_log_str += "In total: " + str(len(self.FILE_LIST)) +" " + self.FILE_FORMAT + " files" + "\n"
        #     self.onStringChanged(File_log_str)
        #     # -------------------
  

    def setHeaderName(self):
        headername = self.ui.line_header.text()
        # File_log_str = ""
        File_log_str = "<span style=\" font-size:13pt; font-weight:600; color:#FF2219;\" >"
        File_log_str_error = ""

        separator_list = [os.sep,'_']
        separator = separator_list[self.categorization-1]
        if headername == "":
            # Log #--------------
            File_log_str += "Please enter the header name"
            File_log_str += "</span>" + "\n"
            self.onStringChanged(File_log_str)
            # -------------------
        elif len(self.FILE_LIST) == 0:
            # Log #--------------
            File_log_str += "Please select file directory first" + "\n"
            File_log_str += "</span>" + "\n"
            self.onStringChanged(File_log_str)
            # -------------------
        else:

            Header_cand = headername.split('_')
            gnum = len(Header_cand) 
            file_contens = True
            for file in self.FILE_LIST:
                if len(file.split(separator)) != gnum:
                    file_contens = False
                    # Log #--------------
                    File_log_str_error += str(file) + "\n"
                    # -------------------
            if file_contens:
                self.HEADER_LIST = headername.split('_')
                # Log #--------------
                File_log_str += "Header is assigned." + "\n"
                File_log_str += "</span>" + "\n"
                self.onStringChanged(File_log_str)
                # -------------------
            else:
                File_log_str += "ERROR: check following file name" + "\n" + File_log_str_error + "\n"
                File_log_str += "</span>" + "\n"
                self.onStringChanged(File_log_str)
            
    def setSavePath(self):
        rootpath = os.path.abspath(os.path.dirname("__file__"))
        self.SAVE_DIR = Qw.QFileDialog.getExistingDirectory(None,"rootpath",rootpath)
        self.SAVE_PATH = self.SAVE_DIR + os.sep + self.SAVE_FILE_NAME
        self.ui.line_save.setText(self.SAVE_PATH)
        # Log #--------------
        self.onStringChanged("SAVE Path: " + self.SAVE_DIR + "\n")
        # -------------------

    def setBGC(self):
        self.BGC = self.ui.cmb_sampleType.currentIndex()
        # print("BGC: "+ self.BGC)

    def setN(self):
        self.N = self.ui.spb_N.value()
        # print("N: "+str(self.N))

    def setMPA(self):
        self.MIN_PIXEL_SIZE = self.ui.spb_mpxa.value()
        # print("MPA: "+str(self.MIN_PIXEL_SIZE))

    def setScaleType(self):
        INDEX = self.ui.cmb_scaleType.currentIndex()
        self.SCALE_TYPE = self.ui.cmb_scaleType.currentIndex()
        if INDEX == 0:
            self.ui.scale_menu.setEnabled(False)
        else:
            self.ui.scale_menu.setEnabled(True)
    def setScaleUnit(self):
        unit_str = self.ui.textBox_scaleU.text()
        if unit_str != "":
            self.SCALE_UNIT = unit_str

    def setScaleValue(self):
        VALUE = self.ui.spb_scaleSize.value()
        if VALUE == 0:
            self.SCALE_VALUE = 1.0
        else:
            self.SCALE_VALUE = VALUE

    def setScalePosition(self):
        POSITION = self.ui.cmb_scalePos.currentIndex()
        self.SCALE_POSITION = POSITION

    def checkEFASetting(self):
        checker = False
        # File_log_str = ""
        File_log_str = "<span style=\" font-size:13pt; font-weight:600; color:#FF2219;\" >"
        if self.FOLDA_DIR == "":
            checker=True
            # Log #--------------
            File_log_str += "ERROR: Image Folda is not selected"
            File_log_str += "</span>" + "\n"
            # -------------------
        elif os.path.exists(self.FOLDA_DIR) == False:
            checker=True
            # Log #--------------
            File_log_str += "ERROR: Selected Image Directory does not exist"
            File_log_str += "</span>" + "\n"
            # -------------------
        elif os.path.isdir(self.FOLDA_DIR) == False:
            checker = True
            # Log #--------------
            File_log_str += "ERROR: Selected Image Directory Path is not a directory"
            File_log_str += "</span>" + "\n"
            # -------------------
        if self.SAVE_DIR == "":
            checker = True
            # Log #--------------
            File_log_str += "ERROR: Save Folda is not selected"
            File_log_str += "</span>" + "\n"
            # self.onStringChanged("Save Folda is not selected" + "\n")
            # -------------------
        elif os.path.exists(self.SAVE_DIR) == False:
            checker = True
            # Log #--------------
            File_log_str += "ERROR: Selected Save Directory does not exist"
            File_log_str += "</span>" + "\n"
            # self.onStringChanged("ERROR: Selected Save Directory does not exist" + "\n")
            # -------------------
        elif os.path.isdir(self.SAVE_DIR) == False:
            checker = True
            # Log #--------------
            File_log_str += "ERROR: Selected Save Path is not a directory"
            File_log_str += "</span>" + "\n"
            # self.onStringChanged("ERROR: Selected Save Path is not a directory" + "\n")
            # -------------------
        self.onStringChanged(File_log_str)
        return checker

    def onCountChanged(self,value):
        self.ui.progressBar.setValue(value)

    def onStringChanged(self,value):
        # Log #--------------
        date_log = "<span style=\" font-size:13pt; font-weight:600; color:#8178FF;\" >"
        date_log += datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        date_log += "</span>" + "\n"
        self.ui.logbox.append(date_log)
        self.ui.logbox.append(str(value))
        self.ui.logbox.moveCursor(Qg.QTextCursor.MoveOperation.End)
        # self.ui.logbox.moveCursor(self.ui.logbox.textCursor().End)
        #-------------------

    def onStringChangedPCA(self,value):
        # Log #--------------
        date_log = "<span style=\" font-size:13pt; font-weight:600; color:#8178FF;\" >"
        date_log += datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        date_log += "</span>" + "\n"
        self.ui.logbox_pca.append(date_log)
        self.ui.logbox_pca.append(str(value))
        self.ui.logbox_pca.moveCursor(Qg.QTextCursor.MoveOperation.End)
        #-------------------

    def onFourierDone(self,path):
        self.ui.textbox_fpsPath.setText(path) 
        self.FPS_PATH = path
    
    def startEFA(self):
        #INITIAL SETTING
        self.FOLDA_DIR = self.ui.line_folda.text()
        self.SAVE_PATH = self.ui.line_save.text()
        #Exception handling
        if self.checkEFASetting():
            return
        #Start analyze
        File_log_str = "Fourier Analysis-----" + "\n"
        # -------------------
        if len(self.FILE_LIST) == 0:
            # Log #--------------
            File_log_str += "No File is found" + "\n" +"------------------ end." + "\n"
            self.onStringChanged(File_log_str)
            # -------------------
        else:
            self.ui.progressBar.setMaximum(len(self.FILE_LIST))
            self.update()
            self.onStringChanged(File_log_str)
            self.fa.start()

    # PCA methods ########################
    def setFPSorEFD(self):
        self.isFPS = self.ui.cmb_FPS.currentIndex()
        

    def setFPSPath(self):
        path = self.openCSVFile()
        if path == '':
            return
        # rootpath = os.path.abspath(os.path.dirname("__file__"))
        # FPS_FOLDA = Qw.QFileDialog.getExistingDirectory(None,"rootpath",rootpath)
        self.ui.textbox_fpsPath.setText(path)
        self.FPS_PATH = path

    def setSavePCAPath(self):
        rootpath = os.path.abspath(os.path.dirname("__file__"))
        self.SAVE_PCA = Qw.QFileDialog.getExistingDirectory(None,"rootpath",rootpath)
        self.ui.textbox_pcaSave.setText(self.SAVE_PCA)

    def setPCAmatrix(self):
        matrix_num =  self.ui.cmb_matrixPCA.currentIndex()
        if matrix_num == 0:
            self.isCorrelationMatrix = True
        else:
            self.isCorrelationMatrix = False

    def checkPCASetting(self):
        checker = False
        # File_log_str = ""
        File_log_str = "<span style=\" font-size:13pt; font-weight:600; color:#FF2219;\" >"
        if self.FPS_PATH == "":
            checker=True
            # Log #--------------
            File_log_str += "ERROR: File Folda is not selected"
            File_log_str += "</span>" + "\n"
            # -------------------
        elif os.path.exists(self.FPS_PATH) == False:
            checker=True
            # Log #--------------
            File_log_str += "ERROR: FPS.csv could not be found" 
            File_log_str += "</span>" + "\n"
            # -------------------
        if self.SAVE_PCA == "":
            checker = True
            # Log #--------------
            File_log_str += "ERROR: Save Folda is not selected" 
            File_log_str += "</span>" + "\n"
            # -------------------
        elif os.path.exists(self.SAVE_PCA) == False:
            checker = True
            # Log #--------------
            File_log_str += "ERROR: Selected Save Directory does not exist"
            File_log_str += "</span>" + "\n"
            # -------------------
        elif os.path.isdir(self.SAVE_PCA) == False:
            checker = True
            # Log #--------------
            File_log_str += "ERROR: Selected Save Path is not a directory"
            File_log_str += "</span>" + "\n"
            # -------------------
        else:
            if checker == False:
                df = pd.read_csv(self.FPS_PATH)
                try:
                    FPS_loc = df.columns.get_loc('FPS1')
                except:
                    checker = True
                    File_log_str += "ERROR: Selected csv file does note have FPS header columns" + "\n"
                    File_log_str += "NOTE: You must select the 'FPS.csv' which is exported at the previous tub " 
                    File_log_str += "</span>" + "\n"
                    self.onStringChangedPCA(File_log_str)
                    return checker
                N = int((len(df.columns) - FPS_loc)/5.0)
                sumple_num = len(df)
                if N > sumple_num and self.isFPS == 0:
                    checker = True
                    # Log #--------------
                    File_log_str += "ERROR: Total number of sumples must be larger than Maximum Harmonics in the case of FPS-based PCA" 
                    File_log_str += "</span>" + "\n"
                    # -------------------
                elif 2*N > sumple_num and self.isFPS == 1:
                    checker = True
                    # Log #--------------
                    File_log_str += "ERROR: Total number of sumples must be larger than Maximum Harmonics in the case of Amplitude-based PCA"
                    File_log_str += "</span>" + "\n"
                    # -------------------
                elif 4*N > sumple_num and self.isFPS == 2:
                    checker = True
                    # Log #--------------
                    File_log_str += "ERROR: Total number of sumples must be larger than Maximum Harmonics in the case of EFD-based PCA" 
                    File_log_str += "</span>" + "\n"
                    # -------------------

        if checker:
            self.onStringChangedPCA(File_log_str)
        return checker

    def startPCA(self):
        self.FPS_PATH = self.ui.textbox_fpsPath.text()
        #Exception handling
        if self.checkPCASetting():
            return
        # PCA Summary
        results_pca,cont,eigen,rot,scale_mat,center_mat,stdv,inv_rot,N = ef.conductPCA_correlation(self.FPS_PATH,self.isFPS,self.isCorrelationMatrix)
        self.N = N
        self.scale_mat = scale_mat
        self.stdv_array = stdv
        self.center_mat = center_mat
        results_pca.to_csv(self.SAVE_PCA+os.sep+"FPS_PCA.csv",header=True, index=False)
        model = PandasModel(cont.T)
        cont.T.to_csv(self.SAVE_PCA+os.sep+"Summary.csv",header=True, index=True)
        self.ui.table_summary.setModel(model)
        # PCA eigen vector
        model_ev = PandasModel(rot)
        rot.to_csv(self.SAVE_PCA+os.sep+"EigenVector.csv",header=True, index=True)
        self.inv_rot = inv_rot
        self.ui.table_vector.setModel(model_ev)
        #Graph Setting
        self.ui.frame_graph.setEnabled(True)
        self.ui.cmb_sort.setEnabled(False)
        self.ui.cmb_subsort.setEnabled(False)
        self.ui.cmb_subsubsort.setEnabled(False)
        fps_pca_header = results_pca.columns

        self.FPS_PCA = results_pca
        self.FPS_PCA_HEADER = fps_pca_header
        cmb_loc = results_pca.columns.get_loc('pixel')
        self.ui.cmb_color.clear()
        self.ui.cmb_subcolor.clear()
        self.ui.cmb_subsubcolor.clear()
        self.ui.cmb_x.clear()
        self.ui.cmb_y.clear()
        self.ui.cmb_color.addItem("None")
        self.ui.cmb_subcolor.addItem("None")
        self.ui.cmb_subsubcolor.addItem("None")
        for i,item in enumerate(fps_pca_header):
            if i < cmb_loc:
                self.ui.cmb_color.addItem(item)
                self.ui.cmb_subcolor.addItem(item)
                self.ui.cmb_subsubcolor.addItem(item)
            else:
                self.ui.cmb_x.addItem(item)
                self.ui.cmb_y.addItem(item)
        self.isPCAdone = True
        # log ------------
        File_log_str = "PCA completed." + "\n"
        File_log_str += "MODE: " +self.PCA_METHOD[self.isFPS]+"\n"
        File_log_str += "Muximum harmonics: " +str( N )+"\n"
        File_log_str += "Total number of samples: " +str( len(results_pca) )+"\n"
        File_log_str += "See summary tab for more detail. " +"\n"
        File_log_str += "You can also draw a simple graph of each parameters including PC scores at graph tab. " +"\n"
        self.onStringChangedPCA(File_log_str)
        #-----------------
        self.ui.logbox_pca.update()
        self.ui.frame_graph.update()

    def setXaxis(self):
        cmb_loc = self.FPS_PCA.columns.get_loc('pixel')
        self.x_var = self.FPS_PCA_HEADER[cmb_loc+self.ui.cmb_x.currentIndex()]

    def setYaxis(self):
        cmb_loc = self.FPS_PCA.columns.get_loc('pixel')
        self.y_var = self.FPS_PCA_HEADER[cmb_loc+self.ui.cmb_y.currentIndex()]
    # def setShape(self):
    #   self.shape_var = self.FPS_PCA_HEADER[self.ui.cmb_shape.currentIndex()]
    def setColor(self):
        ind = self.ui.cmb_color.currentIndex()
        if ind == 0:
            self.color_var = "None_"
            self.ui.cmb_sort.clear()
            self.ui.cmb_sort.setEnabled(False)
            self.ui.cmb_subcolor.setEnabled(False)
            self.ui.cmb_subsort.setEnabled(False)
            self.ui.cmb_subsubcolor.setEnabled(False)
            self.ui.cmb_subsubsort.setEnabled(False)
        else:
            self.color_var = self.FPS_PCA_HEADER[ind-1]
            self.ui.cmb_sort.setEnabled(True)
            self.ui.cmb_subcolor.setEnabled(True)
            self.ui.cmb_subsort.setEnabled(True)
            self.ui.cmb_sort.clear()
            self.ui.cmb_sort.addItem("All_")
            sort_list = pd.unique(self.FPS_PCA[self.color_var])
            for i,sort_item in enumerate(sort_list):
                self.ui.cmb_sort.addItem(str(sort_item))

    
    def setSort(self):
        ind = self.ui.cmb_sort.currentIndex()
        if ind == 0:
            self.ui.cmb_subcolor.setEnabled(False)
            self.ui.cmb_subsort.setEnabled(False)
            self.ui.cmb_subsubcolor.setEnabled(False)
            self.ui.cmb_subsubsort.setEnabled(False)
        else:
            self.ui.cmb_subcolor.setEnabled(True)
            self.ui.cmb_subsort.setEnabled(True)

    def setSubColor(self):
        ind = self.ui.cmb_subcolor.currentIndex()
        if ind == 0:
            self.subcolor_var = "None_"
            self.ui.cmb_subsort.clear()
            self.ui.cmb_subsort.setEnabled(False)
        else:
            self.subcolor_var = self.FPS_PCA_HEADER[ind-1]
            self.ui.cmb_subsort.setEnabled(True)
            self.ui.cmb_subsort.clear()
            self.ui.cmb_subsort.addItem("All_")
            sort_list = pd.unique(self.FPS_PCA[self.subcolor_var])
            for i,sort_item in enumerate(sort_list):
                self.ui.cmb_subsort.addItem(str(sort_item))

    def setSubSort(self):
        ind = self.ui.cmb_subsort.currentIndex()
        if ind == 0:
            self.ui.cmb_subsubcolor.setEnabled(False)
            self.ui.cmb_subsubsort.setEnabled(False)
        else:
            self.ui.cmb_subsubcolor.setEnabled(True)
            self.ui.cmb_subsubsort.setEnabled(True)

    def setSubsubColor(self):
        ind = self.ui.cmb_subsubcolor.currentIndex()
        if ind == 0:
            self.subsubcolor_var = "None_"
            self.ui.cmb_subsubsort.clear()
            self.ui.cmb_subsubsort.setEnabled(False)
        else:
            self.subsubcolor_var = self.FPS_PCA_HEADER[ind-1]
            self.ui.cmb_subsubsort.setEnabled(True)
            self.ui.cmb_subsubsort.clear()
            self.ui.cmb_subsubsort.addItem("All_")
            sort_list = pd.unique(self.FPS_PCA[self.subsubcolor_var])
            for i,sort_item in enumerate(sort_list):
                self.ui.cmb_subsubsort.addItem(str(sort_item))

    def setSubsubSort(self):
        pass

    def setPlotType(self):
        self.PLOT_TYPE = self.ui.cmb_plot.currentIndex()
        ptype = self.ui.cmb_plot.currentIndex()
        if ptype == 2: #Histogram
            self.ui.cmb_y.setEnabled(False)
        else:
            self.ui.cmb_y.setEnabled(True)

    def graphDraw(self):
        #- plot initial setting -------------------------------------
        colors=["#005AFF", "#FF4B00","#03AF7A", "#804000", "#990099", "#FF8082", "#4DC4FF", "#F6AA00", "#FFF100"]
        marker_list = [  "o", ",", "^", "v", "*", "<", ">", "1", ".", "2", "3","4", "8", "s", "p", "h", "H", "+", "x", "D","d", "|", "_", "None", None, "", "$x$","$\\alpha$", "$\\beta$", "$\\gamma$"]
        cmap_list = ["winter","autumn","summer","spring","pink","Wistia"]
        alpha = self.ui.dsb_alpha.value()
        bins_num = self.ui.spb_bins.value()
        hist_alpha = self.ui.dsb_alpha.value()
        clen = len( colors )
        mlen = len( marker_list )
        cmaplen = len( cmap_list )
        marker_size = self.ui.dsb_markerSize.value()
        self.fig.clf()
        self.ax1.clear()
        self.ax1 = self.fig.add_subplot(111)
        # axis label setting
        cmb_loc = self.FPS_PCA.columns.get_loc('pixel')
        pc_loc = self.FPS_PCA.columns.get_loc('PC1')
        pc_stdv = self.stdv_array
        self.ax1.set_xlabel( self.x_var )
        if self.PLOT_TYPE != 2: 
            self.ax1.set_ylabel( self.y_var )
        elif self.PLOT_TYPE == 2:
            self.ax1.set_ylabel('Frequency')
        # x axis setting
        if self.FPS_PCA.columns.get_loc( self.x_var ) >= pc_loc:
            pc_num = self.FPS_PCA.columns.get_loc( self.x_var ) - pc_loc
            self.ax1.set_xticks( np.array( [-2*pc_stdv[pc_num], -1*pc_stdv[pc_num], 0*pc_stdv[pc_num], 1*pc_stdv[pc_num], 2*pc_stdv[pc_num]] ) )
            self.ax1.set_xticklabels( ('-2$\\sigma$', '-$\\sigma$', '0', '$\\sigma$', '2$\\sigma$') )
        self.ax1.set_xlim( min(self.FPS_PCA[self.x_var]) - abs(min(self.FPS_PCA[self.x_var])/10), max(self.FPS_PCA[self.x_var]) + abs(min(self.FPS_PCA[self.x_var])/10) )
        # y axis setting
        if self.FPS_PCA.columns.get_loc( self.y_var ) >= pc_loc:
            pc_num = self.FPS_PCA.columns.get_loc( self.y_var ) - pc_loc
            self.ax1.set_yticks( np.array( [-2*pc_stdv[pc_num], -1*pc_stdv[pc_num], 0*pc_stdv[pc_num], 1*pc_stdv[pc_num], 2*pc_stdv[pc_num]] ) )
            self.ax1.set_yticklabels( ('-2$\\sigma$', '-$\\sigma$', '0', '$\\sigma$', '2$\\sigma$') )
        if self.PLOT_TYPE != 2: # Hist
            self.ax1.set_ylim( min(self.FPS_PCA[self.y_var]) - abs(min(self.FPS_PCA[self.y_var])/10), max(self.FPS_PCA[self.y_var]) + abs(min(self.FPS_PCA[self.y_var])/10) )
        
            # self.ax1.set_ylim(0,1)
        #------------------------------------------------------------
        if self.color_var == "None_": # simple plot (scatter)
            if self.PLOT_TYPE == 0:
                self.ax1.scatter(self.FPS_PCA[self.x_var], self.FPS_PCA[self.y_var], s = marker_size, alpha = alpha)
            elif self.PLOT_TYPE == 1: #density
                if len(self.FPS_PCA) != 0:
                    cfset = ef.kde2dgraphfill(self.ax1,self.FPS_PCA[self.x_var],self.FPS_PCA[self.y_var],min(self.FPS_PCA[self.x_var]),max(self.FPS_PCA[self.x_var]),min(self.FPS_PCA[self.y_var]),max(self.FPS_PCA[self.y_var]))
                    cax = self.fig.add_axes([0.8, 0.2, 0.05, 0.5])
                    self.fig.colorbar(cfset,cax=cax,orientation='vertical')
            elif self.PLOT_TYPE == 2: #Histgoram
                self.ax1.hist(self.FPS_PCA[self.x_var], bins = bins_num, alpha = alpha, weights = np.zeros_like(self.FPS_PCA[self.x_var])+1./self.FPS_PCA[self.x_var].size)
        else:
            if self.ui.cmb_sort.currentIndex() == 0: # use discrete colors and plot all
                for i, value in enumerate(pd.unique(self.FPS_PCA[self.color_var])):
                    color_index = i%clen
                    marker_index = i%mlen
                    cmap_index = i%cmaplen
                    data = self.FPS_PCA.loc[ self.FPS_PCA[self.color_var] == value ]
                    # scatter graph
                    if self.PLOT_TYPE == 0: # scatter
                        self.ax1.scatter(data[self.x_var], data[self.y_var], color = colors[color_index], marker = marker_list[marker_index], alpha = alpha, s = marker_size, label = value)
                    elif self.PLOT_TYPE == 1: # density 
                        if len(data) != 0:
                            cset = ef.kde2dgraph(self.ax1, data[self.x_var], data[self.y_var], min(self.FPS_PCA[self.x_var]),max(self.FPS_PCA[self.x_var]),min(self.FPS_PCA[self.y_var]),max(self.FPS_PCA[self.y_var]),cmap_list[cmap_index])
                    elif self.PLOT_TYPE == 2: # Histogram
                        self.ax1.hist(data[self.x_var], bins = bins_num, alpha = hist_alpha, weights = np.zeros_like(data[self.x_var])+1./data[self.x_var].size,label = value)
            else:  # Sorting by specific value
                value = self.ui.cmb_sort.currentIndex() - 1
                data = self.FPS_PCA.loc[ self.FPS_PCA[self.color_var] == pd.unique(self.FPS_PCA[self.color_var])[value] ]
                if self.subcolor_var == 'None_': 
                    # scatter graph
                    if self.PLOT_TYPE == 0:
                        self.ax1.scatter(data[self.x_var], data[self.y_var], alpha = alpha, s = marker_size, label = pd.unique(self.FPS_PCA[self.color_var])[value])
                    elif self.PLOT_TYPE == 1: # Density plot (when x and y are the same, it stacks)
                        if len(data) != 2:
                            cfset = ef.kde2dgraphfill(self.ax1,data[self.x_var],data[self.y_var],min(self.FPS_PCA[self.x_var]),max(self.FPS_PCA[self.x_var]),min(self.FPS_PCA[self.y_var]),max(self.FPS_PCA[self.y_var]))
                            cax = self.fig.add_axes([0.8, 0.2, 0.05, 0.5])
                            self.fig.colorbar(cfset,cax=cax,orientation='vertical')
                    elif self.PLOT_TYPE == 2:
                        self.ax1.hist(data[self.x_var], bins = bins_num, alpha = alpha, weights=np.zeros_like(data[self.x_var])+1./data[self.x_var].size,label = pd.unique(self.FPS_PCA[self.color_var])[value])
                else:
                    if self.ui.cmb_subsort.currentIndex() == 0: # use discrete colors and plot all
                        for i, value2 in enumerate(pd.unique(data[self.subcolor_var])):
                            color_index = i%clen
                            marker_index = i%mlen
                            cmap_index = i%cmaplen
                            data2 = data.loc[ data[self.subcolor_var] == value2 ]
                            # scatter graph
                            if self.PLOT_TYPE == 0: 
                                self.ax1.scatter(data2[self.x_var], data2[self.y_var], color = colors[color_index], marker = marker_list[marker_index], alpha = alpha, s = marker_size, label = value2)
                            elif self.PLOT_TYPE == 1: 
                                if len(data2) != 0:
                                    cset = ef.kde2dgraph(self.ax1, data2[self.x_var], data2[self.y_var], min(self.FPS_PCA[self.x_var]),max(self.FPS_PCA[self.x_var]),min(self.FPS_PCA[self.y_var]),max(self.FPS_PCA[self.y_var]),cmap_list[cmap_index])
                            elif self.PLOT_TYPE == 2:
                                self.ax1.hist(data2[self.x_var], bins=bins_num, alpha=hist_alpha, weights=np.zeros_like(data2[self.x_var])+1./data2[self.x_var].size, label = value2)
                    else:
                        value2 = self.ui.cmb_subsort.currentIndex() - 1
                        data2 = data.loc[ data[ self.subcolor_var ] == pd.unique(self.FPS_PCA[self.subcolor_var])[value2] ]
                        if self.subsubcolor_var == 'None_':
                            if self.PLOT_TYPE == 0:
                                self.ax1.scatter(data2[self.x_var], data2[self.y_var], alpha = alpha, s = marker_size, label = pd.unique(self.FPS_PCA[self.subcolor_var])[value2])
                            elif self.PLOT_TYPE == 1: # Density plot (when x and y are the same, it stacks)
                                if len(data2) != 0:
                                    cfset = ef.kde2dgraphfill(self.ax1,data2[self.x_var],data2[self.y_var],min(self.FPS_PCA[self.x_var]),max(self.FPS_PCA[self.x_var]),min(self.FPS_PCA[self.y_var]),max(self.FPS_PCA[self.y_var]))
                                    cax = self.fig.add_axes([0.8, 0.2, 0.05, 0.5])
                                    self.fig.colorbar(cfset,cax=cax,orientation='vertical')
                            elif self.PLOT_TYPE == 2:
                                self.ax1.hist(data2[self.x_var], bins=bins_num, alpha=alpha, weights=np.zeros_like(data2[self.x_var])+1./data2[self.x_var].size, label = pd.unique(self.FPS_PCA[self.subcolor_var])[value2])
                        else:
                            if self.ui.cmb_subsubsort.currentIndex() == 0: # use discrete colors and plot all
                                for i, value3 in enumerate(pd.unique(self.FPS_PCA[self.subsubcolor_var])):
                                    color_index = i%clen
                                    marker_index = i%mlen
                                    cmap_index = i%cmaplen
                                    data3 = data2.loc[ data2[self.subsubcolor_var] == value3 ]
                                    # scatter graph
                                    if self.PLOT_TYPE == 0: 
                                        self.ax1.scatter(data3[self.x_var], data3[self.y_var], color = colors[color_index], marker = marker_list[marker_index], alpha = alpha, s = marker_size, label = value3)
                                    elif self.PLOT_TYPE == 1: 
                                        if len(data3) != 0:
                                            cset = ef.kde2dgraph(self.ax1, data3[self.x_var], data3[self.y_var], min(self.FPS_PCA[self.x_var]),max(self.FPS_PCA[self.x_var]),min(self.FPS_PCA[self.y_var]),max(self.FPS_PCA[self.y_var]),cmap_list[cmap_index]) 
                                    elif self.PLOT_TYPE == 1:
                                        self.ax1.hist(data3[self.x_var], bins=bins_num, alpha=hist_alpha, weights=np.zeros_like(data3[self.x_var])+1./data3[self.x_var].size, label = value3)    
                            else:
                                value3 = self.ui.cmb_subsubsort.currentIndex() - 1
                                data3 = data2.loc[ data2[ self.subsubcolor_var ] == pd.unique(self.FPS_PCA[self.subsubcolor_var])[value3] ]
                                # scatter graph
                                if self.PLOT_TYPE == 0:
                                    self.ax1.scatter(data3[self.x_var], data3[self.y_var], alpha = alpha, s = marker_size, label = pd.unique(self.FPS_PCA[self.subsubcolor_var])[value3])
                                elif self.PLOT_TYPE == 1: # Density plot (when x and y are the same, it stacks)
                                    if len(data3) != 0:
                                        cfset = ef.kde2dgraphfill(self.ax1,data3[self.x_var],data3[self.y_var],min(self.FPS_PCA[self.x_var]),max(self.FPS_PCA[self.x_var]),min(self.FPS_PCA[self.y_var]),max(self.FPS_PCA[self.y_var]))
                                        cax = self.fig.add_axes([0.8, 0.2, 0.05, 0.5])
                                        self.fig.colorbar(cfset,cax=cax,orientation='vertical')
                                elif self. PLOT_TYPE == 2:
                                    self.ax1.hist(data3[self.x_var], bins=bins_num, alpha=alpha, weights=np.zeros_like(data3[self.x_var])+1./data3[self.x_var].siz,label = pd.unique(self.FPS_PCA[self.subsubcolor_var])[value3])
        self.ax1.legend()   
        self.fig.canvas.draw_idle()
        self.ui.tab_pca.setCurrentIndex(2)

    def savePCFig(self):
        file_name, _ = Qw.QFileDialog.getSaveFileName(self)
        if len(file_name)==0:
            return
        file_name = str(Path(file_name).with_suffix(".pdf"))
        self.fig.savefig(file_name,bbox_inches='tight')


    #### RECONST METHOD #################
    def setReconstMode(self):
        MODE = self.ui.cmb_recType.currentIndex()
        if MODE == 0:
            self.isSummaryMODE = True
            self.ui.cmb_sumDev.setEnabled(True)
            self.ui.cmb_IP.setEnabled(False)
            self.ui.cmb_IIP.setEnabled(False)
            self.ui.cmb_IIIP.setEnabled(False)
            self.ui.dsb_Id.setEnabled(False)
            self.ui.dsb_IId.setEnabled(False)
            self.ui.dsb_IIId.setEnabled(False)
        else:
            self.isSummaryMODE = False
            self.ui.cmb_sumDev.setEnabled(False)
            self.ui.cmb_IP.setEnabled(True)
            self.ui.cmb_IIP.setEnabled(True)
            self.ui.cmb_IIIP.setEnabled(True)
            self.ui.dsb_Id.setEnabled(True)
            self.ui.dsb_IId.setEnabled(True)
            self.ui.dsb_IIId.setEnabled(True)
            self.ui.cmb_IP.clear()
            self.ui.cmb_IIP.clear()
            self.ui.cmb_IIIP.clear()
            for i in range(self.N):
                self.ui.cmb_IP.addItem(str(i+1))
                self.ui.cmb_IIP.addItem(str(i+1))
                self.ui.cmb_IIIP.addItem(str(i+1))
            self.ui.cmb_IIP.setCurrentIndex(1)
            self.ui.cmb_IIIP.setCurrentIndex(2)


    def setSumDev(self):
        self.sumdev = (self.ui.cmb_sumDev.currentIndex() + 1) * 0.5
    def setIPCaxis(self):
        pass
    def setIIPCaxis(self):
        pass
    def setIIIPCaxis(self):
        pass
    def setIdev(self):
        pass
    def setIIdev(self):
        pass
    def setIIIdev(self):
        pass
    def reconstGraph(self):
        if self.isPCAdone == False:
            return
        
        num_efa = self.N
        
        if self.isFPS == 1:
            num_efa = self.N * 2
        if self.isFPS == 2:
            num_efa = self.N * 4
        PC_SCORE = np.zeros(num_efa)
        if self.isSummaryMODE:
            PC_SCORE[0] = -1*self.sumdev
            fps1 = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax11.clear()
            self.ax11.set_aspect('equal', 'datalim')
            self.ax11.set_xlim([-1.5,1.5])
            self.ax11.set_ylim([-1.0,1.0])
            
            PC_SCORE = np.zeros(num_efa)
            PC_SCORE[0] = 0*self.sumdev
            fps2 = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax12.clear()
            self.ax12.set_aspect('equal', 'datalim')
            self.ax12.set_xlim([-1.5,1.5])
            self.ax12.set_ylim([-1.0,1.0])

            PC_SCORE = np.zeros(num_efa)
            PC_SCORE[0] = self.sumdev
            fps3 = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax13.clear()
            self.ax13.set_aspect('equal', 'datalim')
            self.ax13.set_xlim([-1.5,1.5])
            self.ax13.set_ylim([-1.0,1.0])
            
            PC_SCORE = np.zeros(num_efa)
            PC_SCORE[1] = -1*self.sumdev
            fps4 = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax21.clear()
            self.ax21.set_aspect('equal', 'datalim')
            self.ax21.set_xlim([-1.5,1.5])
            self.ax21.set_ylim([-1.0,1.0])

            PC_SCORE = np.zeros(num_efa)
            PC_SCORE[1] = 0*self.sumdev
            fps5 = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax22.clear()
            self.ax22.set_aspect('equal', 'datalim')
            self.ax22.set_xlim([-1.5,1.5])
            self.ax22.set_ylim([-1.0,1.0])

            PC_SCORE = np.zeros(num_efa)
            PC_SCORE[1] = 1*self.sumdev
            fps6 = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax23.clear()
            self.ax23.set_aspect('equal', 'datalim')
            self.ax23.set_xlim([-1.5,1.5])
            self.ax23.set_ylim([-1.0,1.0])

            PC_SCORE = np.zeros(num_efa)
            PC_SCORE[2] = -1*self.sumdev
            fps7 = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax31.clear()
            self.ax31.set_aspect('equal', 'datalim')
            self.ax31.set_xlim([-1.5,1.5])
            self.ax31.set_ylim([-1.0,1.0])

            PC_SCORE = np.zeros(num_efa)
            PC_SCORE[2] = 0*self.sumdev
            fps8 = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax32.clear()
            self.ax32.set_aspect('equal', 'datalim')
            self.ax32.set_xlim([-1.5,1.5])
            self.ax32.set_ylim([-1.0,1.0])

            PC_SCORE = np.zeros(num_efa)
            PC_SCORE[2] = 1*self.sumdev
            fps9 = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax33.clear()
            self.ax33.set_aspect('equal', 'datalim')
            self.ax33.set_xlim([-1.5,1.5])
            self.ax33.set_ylim([-1.0,1.0])


            for i in range(20):
                if self.isFPS !=2: # FPS or AMp
                    pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps1,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps1,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c=np.random.rand(3,), ax=self.ax11)#,s=0.5)
                    pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps2,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps2,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c=np.random.rand(3,), ax=self.ax12)#,s=0.5)
                    pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps3,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps3,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c=np.random.rand(3,), ax=self.ax13)#,s=0.5)
                    pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps4,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps4,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c=np.random.rand(3,), ax=self.ax21)#,s=0.5)
                    pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps5,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps5,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c=np.random.rand(3,), ax=self.ax22)#,s=0.5)
                    pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps6,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps6,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c=np.random.rand(3,), ax=self.ax23)#,s=0.5)
                    pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps7,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps7,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c=np.random.rand(3,), ax=self.ax31)#,s=0.5)
                    pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps8,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps8,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c=np.random.rand(3,), ax=self.ax32)#,s=0.5)
                    pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps9,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps9,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c=np.random.rand(3,), ax=self.ax33)#,s=0.5)
            if self.isFPS ==2:
                pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps1,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps1,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c='k', ax=self.ax11)#,s=0.5)
                pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps2,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps2,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c='k', ax=self.ax12)#,s=0.5)
                pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps3,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps3,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c='k', ax=self.ax13)#,s=0.5)
                pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps4,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps4,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c='k', ax=self.ax21)#,s=0.5)
                pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps5,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps5,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c='k', ax=self.ax22)#,s=0.5)
                pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps6,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps6,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c='k', ax=self.ax23)#,s=0.5)
                pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps7,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps7,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c='k', ax=self.ax31)#,s=0.5)
                pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps8,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps8,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c='k', ax=self.ax32)#,s=0.5)
                pd.DataFrame({"x":ef.reconstContourCoord(self.N,fps9,self.isFPS)[0],"y":ef.reconstContourCoord(self.N,fps9,self.isFPS)[1]}).plot(kind="line",x="x", y="y",c='k', ax=self.ax33)#,s=0.5)
            self.ax11.set_ylabel("PC1 -" + str(self.sumdev) + "$\\sigma$");self.ax12.set_ylabel("PC1 mean");self.ax13.set_ylabel("PC1 " + str(self.sumdev) + "$\\sigma$");
            self.ax21.set_ylabel("PC2 -" + str(self.sumdev) + "$\\sigma$");self.ax22.set_ylabel("PC2 mean");self.ax23.set_ylabel("PC2 " + str(self.sumdev) + "$\\sigma$");
            self.ax31.set_ylabel("PC3 -" + str(self.sumdev) + "$\\sigma$");self.ax32.set_ylabel("PC3 mean");self.ax33.set_ylabel("PC3 " + str(self.sumdev) + "$\\sigma$");
            self.ax11.set_xlabel("");self.ax12.set_xlabel("");self.ax13.set_xlabel("");
            self.ax21.set_xlabel("");self.ax22.set_xlabel("");self.ax23.set_xlabel("");
            self.ax31.set_xlabel("");self.ax32.set_xlabel("");self.ax33.set_xlabel("");

            self.ax11.get_legend().remove();self.ax12.get_legend().remove();self.ax13.get_legend().remove();
            self.ax21.get_legend().remove();self.ax22.get_legend().remove();self.ax23.get_legend().remove();
            self.ax31.get_legend().remove();self.ax32.get_legend().remove();self.ax33.get_legend().remove();
            self.fig11.canvas.draw_idle(); self.fig12.canvas.draw_idle(); self.fig13.canvas.draw_idle();
            self.fig21.canvas.draw_idle(); self.fig22.canvas.draw_idle(); self.fig23.canvas.draw_idle(); 
            self.fig31.canvas.draw_idle(); self.fig32.canvas.draw_idle(); self.fig33.canvas.draw_idle();
        else:
            PC_SCORE = np.zeros(num_efa)
            PC_SCORE[self.ui.cmb_IP.currentIndex()] = self.ui.dsb_Id.value()
            PC_SCORE[self.ui.cmb_IIP.currentIndex()] = self.ui.dsb_IId.value()
            PC_SCORE[self.ui.cmb_IIIP.currentIndex()] = self.ui.dsb_IIId.value()
            fps = ef.fps(self.isFPS,self.stdv_array * PC_SCORE,self.inv_rot,self.scale_mat,self.center_mat)
            self.ax11.clear();self.ax12.clear();self.ax13.clear()
            self.ax21.clear();self.ax22.clear();self.ax23.clear()
            self.ax31.clear();self.ax32.clear();self.ax33.clear()
            self.ax11.set_aspect('equal', 'datalim');self.ax12.set_aspect('equal', 'datalim');self.ax13.set_aspect('equal', 'datalim')
            self.ax21.set_aspect('equal', 'datalim');self.ax22.set_aspect('equal', 'datalim');self.ax23.set_aspect('equal', 'datalim')
            self.ax31.set_aspect('equal', 'datalim');self.ax32.set_aspect('equal', 'datalim');self.ax33.set_aspect('equal', 'datalim')
            self.ax11.set_xlim([-1.5,1.5]);self.ax12.set_xlim([-1.5,1.5]);self.ax13.set_xlim([-1.5,1.5]);
            self.ax21.set_xlim([-1.5,1.5]);self.ax22.set_xlim([-1.5,1.5]);self.ax23.set_xlim([-1.5,1.5]);
            self.ax31.set_xlim([-1.5,1.5]);self.ax32.set_xlim([-1.5,1.5]);self.ax33.set_xlim([-1.5,1.5]);
            self.ax11.set_ylim([-1.0,1.0]);self.ax12.set_ylim([-1.0,1.0]);self.ax13.set_ylim([-1.0,1.0]);
            self.ax21.set_ylim([-1.0,1.0]);self.ax22.set_ylim([-1.0,1.0]);self.ax23.set_ylim([-1.0,1.0]);
            self.ax31.set_ylim([-1.0,1.0]);self.ax32.set_ylim([-1.0,1.0]);self.ax33.set_ylim([-1.0,1.0]);
            x1,y1,e1 = ef.reconstContourCoord(self.N,fps,self.isFPS); x2,y2,e2 = ef.reconstContourCoord(self.N,fps,self.isFPS); x3,y3,e3 = ef.reconstContourCoord(self.N,fps,self.isFPS); 
            x1,y1,e1 = ef.reconstContourCoord(self.N,fps,self.isFPS); x2,y2,e2 = ef.reconstContourCoord(self.N,fps,self.isFPS); x3,y3,e3 = ef.reconstContourCoord(self.N,fps,self.isFPS); 
            x4,y4,e4 = ef.reconstContourCoord(self.N,fps,self.isFPS); x5,y5,e5 = ef.reconstContourCoord(self.N,fps,self.isFPS); x6,y6,e6 = ef.reconstContourCoord(self.N,fps,self.isFPS);
            x7,y7,e7 = ef.reconstContourCoord(self.N,fps,self.isFPS); x8,y8,e8 = ef.reconstContourCoord(self.N,fps,self.isFPS); x9,y9,e9 = ef.reconstContourCoord(self.N,fps,self.isFPS);
            pd.DataFrame({"x":x1,"y":y1}).plot(kind="line",x="x", y="y", ax=self.ax11); pd.DataFrame({"x":x2,"y":y2}).plot(kind="line",x="x", y="y", ax=self.ax12); pd.DataFrame({"x":x3,"y":y3}).plot(kind="line",x="x", y="y", ax=self.ax13); 
            pd.DataFrame({"x":x4,"y":y4}).plot(kind="line",x="x", y="y", ax=self.ax21); pd.DataFrame({"x":x5,"y":y5}).plot(kind="line",x="x", y="y", ax=self.ax22); pd.DataFrame({"x":x6,"y":y6}).plot(kind="line",x="x", y="y", ax=self.ax23); 
            pd.DataFrame({"x":x7,"y":y7}).plot(kind="line",x="x", y="y", ax=self.ax31); pd.DataFrame({"x":x8,"y":y8}).plot(kind="line",x="x", y="y", ax=self.ax32); pd.DataFrame({"x":x9,"y":y9}).plot(kind="line",x="x", y="y", ax=self.ax33);
            self.ax11.fill_between(x1,y1); self.ax12.fill_between(x2,y2); self.ax13.fill_between(x3,y3)
            self.ax21.fill_between(x4,y4); self.ax22.fill_between(x5,y5); self.ax23.fill_between(x6,y6)
            self.ax31.fill_between(x7,y7); self.ax32.fill_between(x8,y8); self.ax33.fill_between(x9,y9)
            self.ax11.get_legend().remove(); self.ax12.get_legend().remove(); self.ax13.get_legend().remove();
            self.ax21.get_legend().remove(); self.ax22.get_legend().remove(); self.ax23.get_legend().remove();
            self.ax31.get_legend().remove(); self.ax32.get_legend().remove(); self.ax33.get_legend().remove();
            self.ax11.set_ylabel(""); self.ax12.set_ylabel(""); self.ax13.set_ylabel("");
            self.ax21.set_ylabel(""); self.ax22.set_ylabel(""); self.ax23.set_ylabel("");
            self.ax31.set_ylabel(""); self.ax32.set_ylabel(""); self.ax33.set_ylabel("");
            self.fig11.canvas.draw_idle(); self.fig12.canvas.draw_idle(); self.fig13.canvas.draw_idle();
            self.fig21.canvas.draw_idle(); self.fig22.canvas.draw_idle(); self.fig23.canvas.draw_idle();
            self.fig31.canvas.draw_idle(); self.fig32.canvas.draw_idle(); self.fig33.canvas.draw_idle();




    def saveReconstGraph(self):
        if self.isPCAdone == False:
            return
        rootpath = os.path.abspath(os.path.dirname("__file__"))
        FIG_FOLDA = Qw.QFileDialog.getExistingDirectory(None,"rootpath",rootpath)
        if FIG_FOLDA == "":
            return
        file_name = ""
        if self.isSummaryMODE:
            os.makedirs(FIG_FOLDA+os.sep+"Reconst_Sum",exist_ok=True)
            file_name = FIG_FOLDA+os.sep+"Reconst_Sum"+os.sep
        else:
            os.makedirs(FIG_FOLDA+os.sep+"Reconst",exist_ok=True)
            file_name = FIG_FOLDA+os.sep+"Reconst"+os.sep
        self.fig11.savefig(file_name+"fig_11.pdf",bbox_inches='tight')
        self.fig12.savefig(file_name+"fig_12.pdf",bbox_inches='tight')
        self.fig13.savefig(file_name+"fig_13.pdf",bbox_inches='tight')
        self.fig21.savefig(file_name+"fig_21.pdf",bbox_inches='tight')
        self.fig22.savefig(file_name+"fig_22.pdf",bbox_inches='tight')
        self.fig23.savefig(file_name+"fig_23.pdf",bbox_inches='tight')
        self.fig31.savefig(file_name+"fig_31.pdf",bbox_inches='tight')
        self.fig32.savefig(file_name+"fig_32.pdf",bbox_inches='tight')
        self.fig33.savefig(file_name+"fig_33.pdf",bbox_inches='tight')
   
def buildGUI():
    app = Qw.QApplication(sys.argv)         
    wmain = MyForm()                        
    wmain.show()                            
    sys.exit(app.exec())

