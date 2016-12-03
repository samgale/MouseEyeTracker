# -*- coding: utf-8 -*-
"""
GUI for tracking mouse pupil area and position/rotation
Acquire data with camera or analyze data from hdf5 or video file

@author: samgale
"""

import sip
sip.setapi('QString', 2)
import os, copy, time, math, cv2, h5py
import numpy as np
import scipy.io
import scipy.signal
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
from matplotlib import pyplot as plt


class QtSignalGenerator(QtCore.QObject):

    camFrameCapturedSignal = QtCore.pyqtSignal(np.ndarray,float)
    
    def __init__(self):
        QtCore.QObject.__init__(self)


# frame captured callback must be thread safe, hence
# signal generator is used to send frame data to gui thread
qtSignalGeneratorObj = QtSignalGenerator()


def camFrameCaptured(frame):
    img = np.ndarray(buffer=frame.getBufferByteData(),dtype=np.uint8,shape=(frame.height,frame.width))
    qtSignalGeneratorObj.camFrameCapturedSignal.emit(img,frame._frame.timestamp)
    frame.queueFrameCapture(frameCallback=camFrameCaptured)


def start():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    eyeTrackerObj = EyeTracker(app)
    qtSignalGeneratorObj.camFrameCapturedSignal.connect(eyeTrackerObj.processCamFrame)
    app.exec_()


class EyeTracker():
    
    def __init__(self,app):
        self.app = app
        self.fileOpenSavePath = os.path.dirname(os.path.realpath(__file__))
        self.camSavePath = copy.copy(self.fileOpenSavePath)
        self.camSaveBaseName = 'MouseEyeTracker'
        self.nidaq = False
        self.vimba = None
        self.cam = None
        self.camFrames = []
        self.video = None
        self.dataFileIn = None
        self.dataFileOut = None
        self.image = None
        self.roi = None
        self.stopTracking = False
        self.setDataNan = False
        self.pupilCenterSeed = None
        self.pupilRoi = None
        self.pupilCircularityThresh = 0.65
        self.pupilGradientDownsample = 0.5
        self.reflectCenterSeed = None
        self.reflectRoi = []
        self.reflectThresh = 254
        self.maskRoi = []
        self.mmPerPixel = np.nan
        self.lensRotRadius = 1.25
        self.lensOffset = 0.1
        self.corneaOffset = 0.2
        self.defaultDataPlotDur = 2.0
        self.dataIsLoaded = False
        self.selectedSaccade = None
        self.negSaccades = np.array([],dtype=int)
        self.posSaccades = self.negSaccades.copy()
        self.saccadeSmoothPts = 3
        self.saccadeThresh = 5
        self.saccadeRefractoryPeriod = 0.1
        
        # main window
        winWidth = 1000
        winHeight = 500
        self.mainWin = QtGui.QMainWindow()
        self.mainWin.setWindowTitle('MouseEyeTracker')
        self.mainWin.keyPressEvent = self.mainWinKeyPressEvent
        self.mainWin.closeEvent = self.mainWinCloseEvent
        self.mainWin.resize(winWidth,winHeight)
        screenCenter = QtGui.QDesktopWidget().availableGeometry().center()
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(screenCenter)
        self.mainWin.move(mainWinRect.topLeft())
        
        # file menu
        self.menuBar = self.mainWin.menuBar()
        self.menuBar.setNativeMenuBar(False)
        self.fileMenu = self.menuBar.addMenu('File')
        self.fileMenuOpen = QtGui.QAction('Open',self.mainWin)
        self.fileMenuOpen.triggered.connect(self.loadFrameData)
        self.fileMenu.addAction(self.fileMenuOpen)
        
        self.fileMenuSave = self.fileMenu.addMenu('Save')
        self.fileMenuSave.setEnabled(False)
        self.fileMenuSaveData = self.fileMenuSave.addMenu('Data')
        self.fileMenuSaveDataHdf5 = QtGui.QAction('hdf5',self.mainWin)
        self.fileMenuSaveDataHdf5.triggered.connect(self.saveData)
        self.fileMenuSaveDataNpz = QtGui.QAction('npz',self.mainWin)
        self.fileMenuSaveDataNpz.triggered.connect(self.saveData)
        self.fileMenuSaveDataMat = QtGui.QAction('mat',self.mainWin)
        self.fileMenuSaveDataMat.triggered.connect(self.saveData)
        self.fileMenuSaveData.addActions([self.fileMenuSaveDataHdf5,self.fileMenuSaveDataNpz,self.fileMenuSaveDataMat])
        self.fileMenuSaveMovie = QtGui.QAction('Movie',self.mainWin)
        self.fileMenuSaveMovie.triggered.connect(self.saveMovie)
        self.fileMenuSaveAnnotatedMovie = QtGui.QAction('Annotated Movie',self.mainWin,enabled=False)
        self.fileMenuSaveAnnotatedMovie.triggered.connect(self.saveMovie)
        self.fileMenuSave.addActions([self.fileMenuSaveMovie,self.fileMenuSaveAnnotatedMovie])
        
        # camera menu
        self.cameraMenu = self.menuBar.addMenu('Camera')         
        self.cameraMenuUseCam = QtGui.QAction('Use Camera',self.mainWin,checkable=True)
        self.cameraMenuUseCam.triggered.connect(self.initCamera)
        self.cameraMenu.addAction(self.cameraMenuUseCam)
        
        self.cameraMenuSettings = self.cameraMenu.addMenu('Settings')
        self.cameraMenuSettings.setEnabled(False)
        self.cameraMenuSettingsBufferSize = QtGui.QAction('Buffer Size',self.mainWin)
        self.cameraMenuSettingsBufferSize.triggered.connect(self.setCamBufferSize)
        self.cameraMenuSettingsBinning = QtGui.QAction('Spatial Binning',self.mainWin)
        self.cameraMenuSettingsBinning.triggered.connect(self.setCamBinning)
        self.cameraMenuSettingsExposure = QtGui.QAction('Exposure',self.mainWin)
        self.cameraMenuSettingsExposure.triggered.connect(self.setCamExposure)
        self.cameraMenuSettingsFrameRate = QtGui.QAction('Frame Rate',self.mainWin)
        self.cameraMenuSettingsFrameRate.triggered.connect(self.setCamFrameRate)
        self.cameraMenuSettings.addActions([self.cameraMenuSettingsBufferSize,self.cameraMenuSettingsBinning,self.cameraMenuSettingsExposure,self.cameraMenuSettingsFrameRate])
        
        self.cameraMenuNidaq = self.cameraMenu.addMenu('NIDAQ IO')
        self.cameraMenuNidaq.setEnabled(False)
        self.cameraMenuNidaqIn = QtGui.QAction('Use Save Trigger (NIDAQ Input P0.0)',self.mainWin,checkable=True)
        self.cameraMenuNidaqIn.triggered.connect(self.setNidaqIO)
        self.cameraMenuNidaqOut = QtGui.QAction('Signal Saved Frames (NIDAQ Output P1.0)',self.mainWin,checkable=True)
        self.cameraMenuNidaqOut.triggered.connect(self.setNidaqIO)
        self.cameraMenuNidaq.addActions([self.cameraMenuNidaqIn,self.cameraMenuNidaqOut])
        
        self.cameraMenuSavePath = QtGui.QAction('Save Path',self.mainWin)
        self.cameraMenuSavePath.triggered.connect(self.setCamSavePath)
        self.cameraMenuSaveBaseName = QtGui.QAction('Save Basename',self.mainWin)
        self.cameraMenuSaveBaseName.triggered.connect(self.setCamSaveBaseName)
        self.cameraMenu.addActions([self.cameraMenuSavePath,self.cameraMenuSaveBaseName])
        
        # tracking options menu
        self.trackMenu = self.menuBar.addMenu('Track')
        self.trackMenuStopTracking = QtGui.QAction('Stop Tracking',self.mainWin,checkable=True)
        self.trackMenuStopTracking.triggered.connect(self.setStopTracking)
        self.trackMenu.addAction(self.trackMenuStopTracking)
        
        self.trackMenuMmPerPix = self.trackMenu.addMenu('mm/pixel')
        self.trackMenuMmPerPixSet = QtGui.QAction('Set',self.mainWin)
        self.trackMenuMmPerPixSet.triggered.connect(self.setMmPerPix)
        self.trackMenuMmPerPixMeasure = QtGui.QAction('Measure',self.mainWin,enabled=False)
        self.trackMenuMmPerPixMeasure.triggered.connect(self.measureMmPerPix)
        self.trackMenuMmPerPix.addActions([self.trackMenuMmPerPixSet,self.trackMenuMmPerPixMeasure])
        
        self.trackMenuReflectType = self.trackMenu.addMenu('Reflection Type')
        self.trackMenuReflectTypeSpot = QtGui.QAction('Spot',self.mainWin,checkable=True)
        self.trackMenuReflectTypeSpot.setChecked(True)
        self.trackMenuReflectTypeSpot.triggered.connect(self.setReflectType)
        self.trackMenuReflectTypeRing = QtGui.QAction('Ring',self.mainWin,checkable=True)
        self.trackMenuReflectTypeRing.triggered.connect(self.setReflectType)
        self.trackMenuReflectType.addActions([self.trackMenuReflectTypeSpot,self.trackMenuReflectTypeRing])
        self.trackMenuReflectThresh = QtGui.QAction('Reflection Threshold',self.mainWin)
        self.trackMenuReflectThresh.triggered.connect(self.setReflectThresh)
        self.trackMenu.addAction(self.trackMenuReflectThresh)
        
        self.trackMenuPupilSign = self.trackMenu.addMenu('Pupil Sign')
        self.trackMenuPupilSignNeg = QtGui.QAction('Negative',self.mainWin,checkable=True)
        self.trackMenuPupilSignNeg.setChecked(True)
        self.trackMenuPupilSignNeg.triggered.connect(self.setPupilSign)
        self.trackMenuPupilSignPos = QtGui.QAction('Positive',self.mainWin,checkable=True)
        self.trackMenuPupilSignPos.triggered.connect(self.setPupilSign)
        self.trackMenuPupilSign.addActions([self.trackMenuPupilSignNeg,self.trackMenuPupilSignPos])
        
        self.trackMenuPupilMethod = self.trackMenu.addMenu('Pupil Track Method')
        self.trackMenuPupilMethodStarburst = QtGui.QAction('Starburst',self.mainWin,checkable=True)
        self.trackMenuPupilMethodStarburst.setChecked(True)
        self.trackMenuPupilMethodStarburst.triggered.connect(self.setPupilTrackMethod)
        self.trackMenuPupilMethodLine = QtGui.QAction('Line',self.mainWin,checkable=True)
        self.trackMenuPupilMethodLine.triggered.connect(self.setPupilTrackMethod)
        self.trackMenuPupilMethodGradients = QtGui.QAction('Gradients',self.mainWin,checkable=True)
        self.trackMenuPupilMethodGradients.triggered.connect(self.setPupilTrackMethod)
        self.trackMenuPupilMethod.addActions([self.trackMenuPupilMethodStarburst,self.trackMenuPupilMethodLine,self.trackMenuPupilMethodGradients])
        
        self.trackMenuCircularity = QtGui.QAction('Circularity',self.mainWin)
        self.trackMenuCircularity.triggered.connect(self.setCircularityThresh)
        self.trackMenu.addAction(self.trackMenuCircularity)
        
        self.trackMenuLineOrigin = self.trackMenu.addMenu('Line Origin')
        self.trackMenuLineOrigin.setEnabled(False)
        self.trackMenuLineOriginLeft = QtGui.QAction('Left',self.mainWin,checkable=True)
        self.trackMenuLineOriginLeft.setChecked(True)
        self.trackMenuLineOriginLeft.triggered.connect(self.setPupilEdgeLineOrigin)
        self.trackMenuLineOriginRight = QtGui.QAction('Right',self.mainWin,checkable=True)
        self.trackMenuLineOriginRight.triggered.connect(self.setPupilEdgeLineOrigin)
        self.trackMenuLineOrigin.addActions([self.trackMenuLineOriginLeft,self.trackMenuLineOriginRight])        
        
        self.trackMenuGradientDownsamp = QtGui.QAction('Gradient Downsample',self.mainWin,enabled=False)
        self.trackMenuGradientDownsamp.triggered.connect(self.setPupilGradientDownsample)
        self.trackMenu.addAction(self.trackMenuGradientDownsamp)
        
        # analysis menu
        self.analysisMenu = self.menuBar.addMenu('Analysis')
        self.analysisMenu.setEnabled(False)
        self.analysisMenuConvert = self.analysisMenu.addMenu('Convert')
        self.analysisMenuConvertPixToDeg = QtGui.QAction('Pixels to Degrees',self.mainWin)
        self.analysisMenuConvertPixToDeg.triggered.connect(self.pixToDeg)
        self.analysisMenuConvertDegToPix = QtGui.QAction('Degrees to Pixels',self.mainWin)
        self.analysisMenuConvertDegToPix.triggered.connect(self.degToPix)
        self.analysisMenuConvert.addActions([self.analysisMenuConvertPixToDeg,self.analysisMenuConvertDegToPix])
        
        self.analysisMenuLoadData = QtGui.QAction('Load Tracking Data',self.mainWin)
        self.analysisMenuLoadData.triggered.connect(self.loadTrackingData)
        self.analysisMenuAnalyzeAll = QtGui.QAction('Analyze All Frames',self.mainWin)
        self.analysisMenuAnalyzeAll.triggered.connect(self.analyzeAllFrames)
        self.analysisMenuFrameIntervals = QtGui.QAction('Plot Frame Intervals',self.mainWin)
        self.analysisMenuFrameIntervals.triggered.connect(self.plotFrameIntervals)
        self.analysisMenu.addActions([self.analysisMenuLoadData,self.analysisMenuAnalyzeAll,self.analysisMenuFrameIntervals])
        
        self.analysisMenuSaccades = self.analysisMenu.addMenu('Saccades')
        self.analysisMenuSaccadesFind = QtGui.QAction('Find',self.mainWin)
        self.analysisMenuSaccadesFind.triggered.connect(self.findSaccades)
        self.analysisMenuSaccadesDelete = QtGui.QAction('Delete All',self.mainWin)
        self.analysisMenuSaccadesDelete.triggered.connect(self.deleteAllSaccades)
        self.analysisMenuSaccadesSmooth = QtGui.QAction('Smoothing',self.mainWin)
        self.analysisMenuSaccadesSmooth.triggered.connect(self.setSaccadeSmooth)
        self.analysisMenuSaccadesThresh = QtGui.QAction('Threshold',self.mainWin)
        self.analysisMenuSaccadesThresh.triggered.connect(self.setSaccadeThresh)
        self.analysisMenuSaccadesRefractory = QtGui.QAction('Refractory Period',self.mainWin)
        self.analysisMenuSaccadesRefractory.triggered.connect(self.setSaccadeRefractoryPeriod)
        self.analysisMenuSaccades.addActions([self.analysisMenuSaccadesFind,self.analysisMenuSaccadesDelete,self.analysisMenuSaccadesThresh,self.analysisMenuSaccadesSmooth,self.analysisMenuSaccadesRefractory])
        
        # image window
        self.imageLayout = pg.GraphicsLayoutWidget()
        self.imageViewBox = self.imageLayout.addViewBox(lockAspect=1,invertY=True,enableMouse=False,enableMenu=False)
        self.imageViewBox.keyPressEvent = self.mainWinKeyPressEvent
        self.imageItem = pg.ImageItem()
        self.imageItem.mouseClickEvent = self.imageMouseClickEvent
        self.imageItem.mouseDoubleClickEvent = self.imageDoubleClickEvent
        self.imageViewBox.addItem(self.imageItem)
        self.pupilCenterPlot = pg.PlotDataItem(x=[],y=[],symbol='+',symbolSize=10,symbolPen='y')
        self.imageViewBox.addItem(self.pupilCenterPlot)
        self.pupilEllipsePlot = pg.PlotDataItem(x=[],y=[],pen='y')
        self.imageViewBox.addItem(self.pupilEllipsePlot)
        self.pupilEdgePtsPlot = pg.PlotDataItem(x=[],y=[],pen=None,symbol='o',symbolSize=4,symbolPen='y')
        self.imageViewBox.addItem(self.pupilEdgePtsPlot)
        self.reflectCenterPlot = pg.PlotDataItem(x=[],y=[],symbol='+',symbolSize=10,symbolPen='r')
        self.imageViewBox.addItem(self.reflectCenterPlot)
        
        # buttons
        self.startVideoButton = QtGui.QPushButton('Start Video',checkable=True)
        self.startVideoButton.clicked.connect(self.startVideo)
        
        self.roiButton = QtGui.QPushButton('Set ROI',checkable=True)
        self.roiButton.clicked.connect(self.setROI)
        
        self.findPupilButton = QtGui.QPushButton('Find Pupil',checkable=True)
        self.findPupilButton.clicked.connect(self.findPupil)
        
        self.findReflectButton = QtGui.QPushButton('Find Reflection',checkable=True)
        self.findReflectButton.clicked.connect(self.findReflect)
        
        self.setMaskButton = QtGui.QPushButton('Set Masks',checkable=True)
        self.setMaskButton.clicked.connect(self.setMask)
        
        self.buttons = (self.startVideoButton,self.roiButton,self.findPupilButton,self.findReflectButton,self.setMaskButton)
        
        # data plots
        self.dataPlotLayout = pg.GraphicsLayoutWidget()
        self.pupilAreaPlotItem = self.dataPlotLayout.addPlot(row=0,col=0,enableMenu=False)
        self.pupilAreaPlotItem.setMouseEnabled(x=False,y=False)
        self.pupilAreaPlotItem.hideButtons()
        self.pupilAreaPlotItem.setLabel('left','Pupil Area')
        self.pupilAreaPlot = self.pupilAreaPlotItem.plot(x=[0,self.defaultDataPlotDur],y=[0,0])
        self.pupilAreaPlotItem.disableAutoRange()
        self.pupilXPlotItem = self.dataPlotLayout.addPlot(row=1,col=0,enableMenu=False)
        self.pupilXPlotItem.setMouseEnabled(x=False,y=False)
        self.pupilXPlotItem.hideButtons()
        self.pupilXPlotItem.setLabel('left','Pupil X')
        self.pupilXPlotItem.mouseClickEvent = self.dataPlotMouseClickEvent
        self.pupilXPlotItem.mouseDoubleClickEvent = self.dataPlotDoubleClickEvent
        self.pupilXPlot = self.pupilXPlotItem.plot(x=[0,self.defaultDataPlotDur],y=[0,0])
        self.pupilXPlotItem.disableAutoRange()
        self.pupilYPlotItem = self.dataPlotLayout.addPlot(row=2,col=0,enableMenu=False)
        self.pupilYPlotItem.setMouseEnabled(x=False,y=False)
        self.pupilYPlotItem.hideButtons()
        self.pupilYPlotItem.setLabel('left','Pupil Y')
        self.pupilYPlotItem.setLabel('bottom','Time (s)')
        self.pupilYPlot = self.pupilYPlotItem.plot(x=[0,self.defaultDataPlotDur],y=[0,0])
        self.pupilYPlotItem.disableAutoRange()
        
        # saccade plots
        triangles = [QtGui.QPainterPath() for _ in range(2)]
        xpts = [(-0.5,0,0.5)]*2
        ypts = [(-0.5,0.5,-0.5),(0.5,-0.5,0.5)]
        for tri,x,y in zip(triangles,xpts,ypts):
            tri.moveTo(x[0],y[0])
            for i in (1,2):
                tri.lineTo(x[i],y[i])
            tri.closeSubpath()
        downTriangle,upTriangle = triangles
        self.negSaccadesPlot = self.pupilXPlotItem.plot(x=[],y=[],pen=None,symbol=downTriangle,symbolSize=10,symbolPen='b')
        self.posSaccadesPlot = self.pupilXPlotItem.plot(x=[],y=[],pen=None,symbol=upTriangle,symbolSize=10,symbolPen='b')
        
        # pupil tracking parameter plots
        numPupilEdges = 18
        self.radialProfilePlot = []
        self.radialProfilePixAboveThreshPlot = []
        for i in range(numPupilEdges):
            self.radialProfilePlot.append(self.pupilAreaPlotItem.plot(x=[0],y=[0]))
            self.radialProfilePixAboveThreshPlot.append(self.pupilXPlotItem.plot(x=[0],y=[0]))
        self.edgeDistPlot = self.pupilYPlotItem.plot(x=[0],y=[0])
        self.pupilEdgeThreshLine = pg.InfiniteLine(pos=0,angle=0,pen='r',movable=True,bounds=(0,254))
        self.pupilEdgeThreshLine.sigPositionChangeFinished.connect(self.setPupilEdgeThresh)
        self.numPixAboveThreshLine = pg.InfiniteLine(pos=0,angle=0,pen='r',movable=True,bounds=(1,1e4))
        self.numPixAboveThreshLine.sigPositionChangeFinished.connect(self.setMinNumPixAboveThresh)
        self.edgeDistUpperThreshLine = pg.InfiniteLine(pos=0,angle=0,pen='r',movable=True,bounds=(0,1e4))
        self.edgeDistUpperThreshLine.sigPositionChangeFinished.connect(self.setEdgeDistThresh)
        self.edgeDistLowerThreshLine = pg.InfiniteLine(pos=0,angle=0,pen='r',movable=True,bounds=(0,1e4))
        self.edgeDistLowerThreshLine.sigPositionChangeFinished.connect(self.setEdgeDistThresh)
        
        # save and mask checkboxes
        self.saveCheckBox = QtGui.QCheckBox('Save Video Data',enabled=False)
        self.useMaskCheckBox = QtGui.QCheckBox('Use Masks')
        
        # frame navigation
        self.frameNumSpinBox = QtGui.QSpinBox()
        self.frameNumSpinBox.setPrefix('Frame: ')
        self.frameNumSpinBox.setSuffix(' of 0')
        self.frameNumSpinBox.setRange(0,1)
        self.frameNumSpinBox.setSingleStep(1)
        self.frameNumSpinBox.setValue(0)
        self.frameNumSpinBox.setEnabled(False)
        self.frameNumSpinBox.valueChanged.connect(self.goToFrame)
        self.frameNumSpinBox.blockSignals(True)
        
        self.pupilAreaFrameNumLine = pg.InfiniteLine(pos=0,pen='r',movable=True,bounds=(0,1))
        self.pupilAreaFrameNumLine.sigDragged.connect(self.frameNumLineDragged)
        self.pupilAreaFrameNumLine.sigPositionChangeFinished.connect(self.frameNumLinePosChangeFin)
        self.pupilXFrameNumLine = pg.InfiniteLine(pos=0,pen='r',movable=True,bounds=(0,1))
        self.pupilXFrameNumLine.sigDragged.connect(self.frameNumLineDragged)
        self.pupilXFrameNumLine.sigPositionChangeFinished.connect(self.frameNumLinePosChangeFin)
        self.pupilYFrameNumLine = pg.InfiniteLine(pos=0,pen='r',movable=True,bounds=(0,1))
        self.pupilYFrameNumLine.sigDragged.connect(self.frameNumLineDragged)
        self.pupilYFrameNumLine.sigPositionChangeFinished.connect(self.frameNumLinePosChangeFin)
        self.frameNumLines = (self.pupilAreaFrameNumLine,self.pupilXFrameNumLine,self.pupilYFrameNumLine)
        
        # data plot duration control
        self.plotDurLayout = QtGui.QFormLayout()
        self.plotDurEdit = QtGui.QLineEdit(str(self.defaultDataPlotDur))
        self.plotDurEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.plotDurEdit.editingFinished.connect(self.changePlotWindowDur)
        self.plotDurLayout.addRow('Plot Duration',self.plotDurEdit)
        
        # layout
        self.mainWidget = QtGui.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtGui.QGridLayout()
        nCols = 20
        nRows = 4
        for col in range(nCols):
            self.mainLayout.setColumnMinimumWidth(col,winWidth/nCols)
            self.mainLayout.setColumnStretch(col,1)
        rowHeights = np.zeros(nRows)
        rowHeights[[0,-1]] = 0.05*winHeight
        rowHeights[1:-1] = 0.9*winHeight/(nRows-2)
        for row in range(nRows):
            self.mainLayout.setRowMinimumHeight(row,rowHeights[row])
            self.mainLayout.setRowStretch(row,1)
        self.mainWidget.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.imageLayout,0,0,4,10)
        self.mainLayout.addWidget(self.startVideoButton,0,10,1,2)
        self.mainLayout.addWidget(self.roiButton,0,12,1,2)
        self.mainLayout.addWidget(self.findPupilButton,0,14,1,2)
        self.mainLayout.addWidget(self.findReflectButton,0,16,1,2)
        self.mainLayout.addWidget(self.setMaskButton,0,18,1,2)
        self.mainLayout.addWidget(self.dataPlotLayout,1,10,2,10)
        self.mainLayout.addWidget(self.saveCheckBox,3,10,1,2)
        self.mainLayout.addWidget(self.useMaskCheckBox,3,12,1,2)
        self.mainLayout.addWidget(self.frameNumSpinBox,3,14,1,3)
        self.mainLayout.addLayout(self.plotDurLayout,3,17,1,3)
        self.mainWin.show()
        
    def mainWinCloseEvent(self,event):
        if self.cam is not None:
            self.closeCamera()
        elif self.video is not None:
            self.closeVideo()
        elif self.dataFileIn is not None:
            self.closeDataFileIn()
        if self.nidaq:
            self.nidaqDigInputs.ClearTask()
            self.nidaqDigOutputs.ClearTask()
        event.accept()
        
    def saveData(self):
        if self.mainWin.sender()==self.fileMenuSaveDataHdf5:
            fileType = 'hdf5'
        elif self.mainWin.sender()==self.fileMenuSaveDataNpz:
            fileType = 'npz'
        else:
            fileType = 'mat'
        filePath = QtGui.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileOpenSavePath,'*.'+fileType)
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        self.reflectCenter += self.roiPos
        self.pupilCenter += self.roiPos
        params = ('mmPerPixel','frameTimes','reflectCenter','pupilCenter','pupilArea','pupilX','pupilY','negSaccades','posSaccades')
        if self.dataFileIn is not None:
            self.getFrameTimes()
        if fileType=='hdf5':
            dataFile = h5py.File(filePath,'w',libver='latest')
            dataFile.attrs.create('mmPerPixel',self.mmPerPixel)
            for param in params[1:]:
                dataFile.create_dataset(param,data=getattr(self,param),compression='gzip',compression_opts=1)
            dataFile.close()
        else:
            data = {param: getattr(self,param) for param in params}
            if fileType=='npz':
                np.savez_compressed(filePath,**data)
            else:
                scipy.io.savemat(filePath,data,do_compression=True)
        
    def saveMovie(self):
        startFrame,endFrame = (self.frameNum-self.numDataPlotPts,self.frameNum) if self.frameNum>self.numDataPlotPts else (1,self.numDataPlotPts)
        text,ok = QtGui.QInputDialog.getText(self.mainWin,'Save Movie','Enter frames range:',text=str(startFrame)+'-'+str(endFrame))
        if not ok:
            return
        startFrame,endFrame = [int(n) for n in text.split('-')]
        if startFrame<1:
            startFrame = 1
        if endFrame>self.numFrames:
            endFrame = self.numFrames
        filePath = QtGui.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileOpenSavePath,'*.avi')
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        vidOut = cv2.VideoWriter(filePath,-1,self.frameRate,self.roiSize)
        if self.dataFileIn is None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES,startFrame-1)
        for frame in range(startFrame,endFrame+1):
            if self.dataFileIn is None:
                isImage,image = self.video.read()
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            else:
                image = self.dataFileIn[str(frame)][:,:]
            vidOut.write(image[self.roiInd])
        vidOut.release()
        if self.dataFileIn is None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES,self.frameNum-1)
               
    def loadFrameData(self):
        filePath = QtGui.QFileDialog.getOpenFileName(self.mainWin,'Choose File',self.fileOpenSavePath,'*.avi *.mov *.hdf5')
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        if self.cam is not None:
            self.cameraMenuUseCam.setChecked(False)
            self.closeCamera()
        elif self.dataFileIn is not None:
            self.closeDataFileIn()
        elif self.video is not None:
            self.closeVideo()
        if os.path.splitext(os.path.basename(filePath))[1]=='.hdf5':
            self.dataFileIn = h5py.File(filePath,'r')
            self.frameRate = self.dataFileIn.attrs.get('frameRate')
            if 'numFrames' in self.dataFileIn.attrs.keys():
                self.numFrames = self.dataFileIn.attrs.get('numFrames')
            else:
                self.numFrames = sum(1 for _ in self.dataFileIn.keys())
            if 'mmPerPixel' in self.dataFileIn.attrs.keys():
                self.mmPerPixel = self.dataFileIn.attrs.get('mmPerPixel')
        else:
            self.video = cv2.VideoCapture(filePath)
            self.frameRate = self.video.get(cv2.CAP_PROP_FPS)
            self.numFrames = int(round(self.video.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.frameTimes = np.full(self.numFrames,np.nan)
        if self.defaultDataPlotDur>self.numFrames/self.frameRate:
            self.dataPlotDur = self.numFrames/self.frameRate
        else:
            self.dataPlotDur = self.defaultDataPlotDur
        self.frameNumSpinBox.setRange(1,self.numFrames)
        self.frameNumSpinBox.setValue(1)
        self.frameNumSpinBox.setSuffix(' of '+str(self.numFrames))
        self.frameNumSpinBox.blockSignals(False)
        self.frameNumSpinBox.setEnabled(True)
        self.fileMenuSave.setEnabled(True)
        self.analysisMenu.setEnabled(True)
        for line in self.frameNumLines:
            line.setBounds((0,(self.numFrames-1)/self.frameRate))
        self.addFrameNumLines()
        self.plotDurEdit.setText(str(round(self.dataPlotDur,3)))
        self.frameNum = 1
        self.getVideoImage()
        self.initDisplay()
        
    def closeDataFileIn(self):
        self.closeFileCleanup()
        self.dataIsLoaded = False
        self.dataFileIn.close()
        self.dataFileIn = None
        
    def closeVideo(self):
        self.closeFileCleanup()
        self.video.release()
        self.video = None        
        
    def closeFileCleanup(self):
        self.turnOffButtons()
        self.frameNumSpinBox.setEnabled(False)
        self.frameNumSpinBox.blockSignals(True)
        self.frameNumSpinBox.setRange(0,1)
        self.frameNumSpinBox.setValue(0)
        self.frameNumSpinBox.setSuffix(' of 0')
        self.fileMenuSave.setEnabled(False)
        self.analysisMenu.setEnabled(False)
        self.removeFrameNumLines()
        for line in self.frameNumLines:
            line.setValue(0)
        self.frameTimes = []
        self.resetPupilTracking()
        self.deleteAllSaccades()
       
    def initCamera(self):
        if self.cameraMenuUseCam.isChecked():
            if self.dataFileIn is not None:
                self.closeDataFileIn()
            elif self.video is not None:
                self.closeVideo()
            if self.vimba is None:
                try:
                    import pymba
                    self.vimba = pymba.Vimba()
                except:
                    raise ImportError('pymba')
            try:
                self.vimba.startup()
                system = self.vimba.getSystem()
                system.runFeatureCommand("GeVDiscoveryAllOnce")
                time.sleep(0.2)
                cameraIds = self.vimba.getCameraIds()
                self.cam = self.vimba.getCamera(cameraIds[0])
            except:
                raise RuntimeError('Unable to open camera')
            if not self.nidaq:
                try:
                    import nidaq
                    self.nidaqDigInputs = nidaq.DigitalInputs(device='Dev1',port=0)
                    self.nidaqDigOutputs = nidaq.DigitalOutputs(device='Dev1',port=1,initialState='low')
                    self.nidaqCh = 0
                    self.cameraMenuNidaq.setEnabled(True)
                    self.cameraMenuNidaqOut.setChecked(True)
                    self.nidaq = True
                except:
                    print('Unable to import/use nidaq')
            self.cam.openCamera()
            self.setCamProps()
            self.dataPlotDur = self.defaultDataPlotDur
            self.frameNum = 0
            self.initDisplay()
            self.cameraMenuSettings.setEnabled(True)
            self.trackMenuMmPerPixMeasure.setEnabled(True)
            if not self.cameraMenuNidaqIn.isChecked():
                self.saveCheckBox.setEnabled(True)
        else:
            self.closeCamera() 
            
    def closeCamera(self):
        self.turnOffButtons()
        self.cam.closeCamera()
        self.vimba.shutdown()
        self.cam = None
        self.cameraMenuSettings.setEnabled(False)
        self.trackMenuMmPerPixMeasure.setEnabled(False)
        self.saveCheckBox.setEnabled(False)
        self.resetPupilTracking()
        
    def startCamera(self,bufferSize=1):
        self.cameraMenuSettings.setEnabled(False)
        if self.nidaq:
            self.nidaqDigInputs.StartTask()
            self.nidaqDigOutputs.StartTask()
            self.nidaqDigOutputs.Write(np.zeros(self.nidaqDigOutputs.deviceLines,dtype=np.uint8))
        for _ in range(bufferSize):
            frame = self.cam.getFrame()
            frame.announceFrame()
            self.camFrames.append(frame)
        self.cam.startCapture()
        
    def stopCamera(self):
        self.cam.runFeatureCommand("AcquisitionStop")
        self.cam.endCapture()
        self.cam.revokeAllFrames()
        self.camFrames = []
        if self.dataFileOut is not None:
            self.closeDataFileOut()
        if self.nidaq:
            self.nidaqDigInputs.StopTask()
            self.nidaqDigOutputs.StopTask()
        self.cameraMenuSettings.setEnabled(True)
        
    def getCamImage(self):
        self.startCamera()
        frame = self.camFrames[0]
        frame.queueFrameCapture()
        self.cam.runFeatureCommand("AcquisitionStart")
        frame.waitFrameCapture()
        self.image = np.ndarray(buffer=frame.getBufferByteData(),dtype=np.uint8,shape=(frame.height,frame.width))
        self.stopCamera()
            
    def setCamProps(self):
        self.frameRate = 60.0
        self.camBufferSize = 60
        self.camExposure = 0.9
        self.cam.PixelFormat='Mono8'
        self.cam.BinningHorizontal = 1
        self.cam.BinningVertical = 1
        self.cam.OffsetX = 0
        self.cam.OffsetY = 0
        self.cam.Width = self.cam.WidthMax
        self.cam.Height = self.cam.HeightMax
        self.cam.ExposureAuto = 'Off'
        self.cam.ExposureTimeAbs = self.camExposure*1e6/self.frameRate
        self.cam.AcquisitionFrameRateAbs = self.frameRate
        self.cam.AcquisitionMode = 'Continuous'
        self.cam.TriggerMode = 'Off'
        self.cam.TriggerSource = 'FixedRate'
        self.cam.SyncOutSelector = 'SyncOut2'
        self.cam.SyncOutSource = 'Exposing'
        self.cam.SyncOutPolarity = 'Normal'
        
    def setCamBufferSize(self):
        val,ok = QtGui.QInputDialog.getInt(self.mainWin,'Set Camera Buffer Size','Frames:',value=self.camBufferSize,min=1)
        if not ok:
            return
        self.camBufferSize = val
            
    def setCamBinning(self):
        val,ok = QtGui.QInputDialog.getInt(self.mainWin,'Set Camera Spatial Binning','Pixels:',value=self.cam.BinningHorizontal,min=1,max=8)
        if not ok:
            return
        scaleFactor = self.cam.BinningHorizontal/val
        if self.pupilCenterSeed is not None:
            self.pupilCenterSeed = [int(n*scaleFactor) for n in self.pupilCenterSeed]
        if self.reflectCenterSeed is not None:
            self.reflectCenterSeed = [int(n*scaleFactor) for n in self.reflectCenterSeed]
        if self.pupilRoi is not None:
            self.pupilRoiPos = [int(n*scaleFactor) for n in self.pupilRoiPos]
            self.pupilRoiSize = [int(n*scaleFactor) for n in self.pupilRoiSize]
        for i,roi in enumerate(self.reflectRoi):
            self.reflectRoiPos[i] = [int(n*scaleFactor) for n in roi.pos()]
            self.reflectRoiSize[i] = [int(n*scaleFactor) for n in roi.size()]
        if len(self.maskRoi)>0:
            for roi in self.maskRoi:
                roi.setPos([int(n*scaleFactor) for n in roi.pos()])
                roi.setSize([int(n*scaleFactor) for n in roi.size()])
            self.updateMaskIndex()
        self.cam.BinningHorizontal = val
        self.cam.BinningVertical = val
        self.resetROI()
        self.resetImage()
    
    def setCamExposure(self):
        val,ok = QtGui.QInputDialog.getDouble(self.mainWin,'Set Camera Exposure','Fraction of frame interval:',value=self.camExposure,min=0.01,max=0.99,decimals=2)
        if not ok:
            return
        self.camExposure = val
        self.cam.ExposureTimeAbs = self.camExposure*1e6/self.frameRate
    
    def setCamFrameRate(self):
        val,ok = QtGui.QInputDialog.getDouble(self.mainWin,'Set Camera Frame Rate','Frames/s:',value=self.frameRate,min=0.01,max=119.30,decimals=2)
        if not ok:
            return
        self.frameRate = val
        self.cam.AcquisitionFrameRateAbs = self.frameRate
        self.cam.ExposureTimeAbs = self.camExposure*1e6/self.frameRate
        self.changePlotWindowDur()
    
    def setCamSavePath(self):
        dirPath = QtGui.QFileDialog.getExistingDirectory(self.mainWin,'Choose Directory',self.camSavePath)
        if dirPath!='':
            self.camSavePath = dirPath
        
    def setCamSaveBaseName(self):
        val,ok = QtGui.QInputDialog.getText(self.mainWin,'Set File Base Name','',text=self.camSaveBaseName)
        if ok:
            self.camSaveBaseName = val
        
    def setNidaqIO(self):
        if self.mainWin.sender() is self.cameraMenuNidaqIn:
            if self.cameraMenuNidaqIn.isChecked():
                self.saveCheckBox.setEnabled(False)
            else:
                if self.cam is not None:
                    self.saveCheckBox.setEnabled(True)
                    
    def getVideoImage(self):
        if self.dataFileIn is not None:
            self.image = self.dataFileIn[str(self.frameNum)][:,:]
            if np.isnan(self.frameTimes[self.frameNum-1]):
                self.frameTimes[self.frameNum-1] = self.dataFileIn[str(self.frameNum)].attrs.get('acquisitionTime')
        else:
            isImage,image = self.video.read()
            self.image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    
    def initDisplay(self):
        self.resetROI()
        self.resetImage()
        self.setDataPlotTime()
        self.resetPupilData()
        self.resetPupilDataPlot()        
        self.setDataPlotXRange()
               
    def resetROI(self):
        if self.cam is not None:
            self.cam.OffsetX = 0
            self.cam.OffsetY = 0
            self.cam.Width = self.cam.WidthMax
            self.cam.Height = self.cam.HeightMax
            self.getCamImage()
        self.roiPos = (0,0)
        self.roiSize = (self.image.shape[1],self.image.shape[0])
        self.fullRoiSize = copy.copy(self.roiSize)
        self.roiInd = np.s_[0:self.roiSize[1],0:self.roiSize[0]]
        
    def resetImage(self):
        self.imageItem.setImage(self.image[self.roiInd].T,autoLevels=False,autoDownSample=True)
        self.imageViewBox.autoRange(items=[self.imageItem])
        if self.pupilCenterSeed is not None:
            self.pupilCenterPlot.setData(x=[self.pupilCenterSeed[0]],y=[self.pupilCenterSeed[1]])
            self.pupilEllipsePlot.setData(x=[],y=[])
        if self.reflectCenterSeed is not None:
            self.reflectCenterPlot.setData(x=[self.reflectCenterSeed[0]],y=[self.reflectCenterSeed[1]])
        self.getRadialLines()
        
    def resetPupilTracking(self):
        self.pupilCenterPlot.setData(x=[],y=[])
        self.pupilEllipsePlot.setData(x=[],y=[])
        self.pupilCenterSeed = None
        self.reflectCenterPlot.setData(x=[],y=[])
        self.reflectCenterSeed = None
        if self.pupilRoi is not None:
            self.imageViewBox.removeItem(self.pupilRoi)
            self.pupilRoi = None
        for roi in self.reflectRoi:
            self.imageViewBox.removeItem(roi)
        self.reflectRoi = []
        for roi in self.maskRoi:
            self.imageViewBox.removeItem(roi)
        self.maskRoi = []
        self.trackMenuStopTracking.setChecked(False)
        self.stopTracking = False
        self.setDataNan = False
        
    def resetPupilData(self):
        self.dataPlotIndex = 0
        n = self.numFrames if self.cam is None else self.numDataPlotPts
        self.reflectCenter = np.full((n,2),np.nan)
        self.pupilCenter = np.full((n,2),np.nan)
        self.pupilArea = np.full(n,np.nan)
        self.pupilX = np.full(n,np.nan)
        self.pupilY = np.full(n,np.nan)
            
    def resetPupilDataPlot(self):
        self.pupilAreaPlot.setData(x=[0,self.dataPlotDur],y=[0,0])
        self.pupilXPlot.setData(x=[0,self.dataPlotDur],y=[0,0])
        self.pupilYPlot.setData(x=[0,self.dataPlotDur],y=[0,0])
        
    def startVideo(self):
        if self.startVideoButton.isChecked():
            self.turnOffButtons(source=self.startVideoButton)
            self.startVideoButton.setText('Stop Video')
            if self.cam is None:
                self.frameNumSpinBox.blockSignals(True)
                if self.frameNum==self.numFrames:
                    self.frameNum = 0
                    self.setDataPlotXRange()
                    if self.video is not None:
                        self.video.set(cv2.CAP_PROP_POS_FRAMES,0)
                while self.startVideoButton.isChecked():
                    if self.frameNum==self.numFrames:
                        self.startVideoButton.click()
                        break
                    self.frameNum += 1
                    self.frameNumSpinBox.setValue(self.frameNum)
                    self.getVideoImage()
                    self.updateDisplay()
                    self.app.processEvents()
            else:
                self.frameNum = 0
                self.resetPupilData()
                self.startCamera(bufferSize=self.camBufferSize)
                for frame in self.camFrames:
                    frame.queueFrameCapture(frameCallback=camFrameCaptured)
                self.cam.runFeatureCommand("AcquisitionStart")
        else:
            if self.cam is not None:
                self.stopCamera()
            self.updateDisplay(updateAll=True)
            if self.cam is None:
                self.frameNumSpinBox.blockSignals(False)
            self.startVideoButton.setText('Start Video')
            
    def updateDisplay(self,updateNone=False,updateAll=False):
        n = (self.frameNum-1)%5
        if updateAll or (not updateNone and n==0):
            self.imageItem.setImage(self.image[self.roiInd].T,autoLevels=False,autoDownsample=True)
        if self.cam is None:
            self.selectedSaccade = None
        else:
            if self.dataPlotIndex==self.numDataPlotPts-1:
                self.dataPlotIndex = 0
            else:
                self.dataPlotIndex += 1
        if not self.stopTracking and (self.setDataNan or self.reflectCenterSeed is not None):
            self.trackReflect()
            if updateAll or (not updateNone and n==1):
                if self.reflectFound:
                    self.reflectCenterPlot.setData(x=[self.reflectCenterSeed[0]],y=[self.reflectCenterSeed[1]])
                else:
                    self.reflectCenterPlot.setData(x=[],y=[])
        if not self.stopTracking and (self.setDataNan or self.pupilCenterSeed is not None):
            self.trackPupil()
            if updateAll or (not updateNone and n==1):
                self.updatePupilPlot()
        if self.pupilCenterSeed is not None or self.dataIsLoaded:
            if updateAll:
                self.updatePupilDataPlot()
            elif not updateNone and n>1:
                updatePlotN = [False,False,False]
                updatePlotN[n-2] = True
                self.updatePupilDataPlot(updatePlotN)
        
    def processCamFrame(self,img,timestamp):
        self.frameNum += 1
        self.image = img
        skipDisplayUpdate = False
        if (self.saveCheckBox.isChecked() and not self.cameraMenuNidaqIn.isChecked()) or (self.cameraMenuNidaqIn.isChecked() and not self.nidaqDigInputs.Read()[self.nidaqCh]):
            if self.dataFileOut is None:
                self.frameNum = 0
                if self.cameraMenuNidaqIn.isChecked():
                    self.saveCheckBox.setChecked(True)
                self.dataFileOut = h5py.File(os.path.join(self.camSavePath,self.camSaveBaseName+'_'+time.strftime('%Y%m%d_%H%M%S.hdf5')),'w',libver='latest')
                self.dataFileOut.attrs.create('frameRate',self.frameRate)
                self.dataFileOut.attrs.create('mmPerPixel',self.mmPerPixel)
                skipDisplayUpdate = True
            else:
                self.nidaqDigOutputs.WriteBit(self.nidaqCh,1)
                dataset = self.dataFileOut.create_dataset(str(self.frameNum),data=self.image,chunks=self.image.shape,compression='gzip',compression_opts=1)
                dataset.attrs.create('acquisitionTime',timestamp/self.cam.GevTimestampTickFrequency)
                self.nidaqDigOutputs.WriteBit(self.nidaqCh,0)
        elif self.dataFileOut is not None:
            self.closeDataFileOut()
            skipDisplayUpdate = True
        self.updateDisplay(updateNone=skipDisplayUpdate)       
        
    def closeDataFileOut(self):
        self.dataFileOut.attrs.create('numFrames',self.frameNum-1)
        self.dataFileOut.close()
        self.dataFileOut = None
        self.frameNum = 0
        if self.cameraMenuNidaqIn.isChecked():
            self.saveCheckBox.setChecked(False)
            
    def setStopTracking(self):
        self.stopTracking = not self.stopTracking
        if self.stopTracking:
            self.reflectCenterPlot.setData(x=[],y=[])
            self.pupilCenterPlot.setData(x=[],y=[])
            self.pupilEllipsePlot.setData(x=[],y=[])
            
    def setCurrentFrameDataNan(self):
        self.reflectCenterPlot.setData(x=[],y=[])
        self.pupilCenterPlot.setData(x=[],y=[])
        self.pupilEllipsePlot.setData(x=[],y=[])
        self.pupilArea[self.frameNum-1] = np.nan
        self.pupilX[self.frameNum-1] = np.nan
        self.pupilY[self.frameNum-1] = np.nan
        self.updatePupilDataPlot()
            
    def mainWinKeyPressEvent(self,event):
        key = event.key()
        modifiers = QtGui.QApplication.keyboardModifiers()
        if key in (QtCore.Qt.Key_Comma,QtCore.Qt.Key_Period):
            if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                frameShift = int(0.9*self.numDataPlotPts) if int(modifiers & QtCore.Qt.ControlModifier)>0 else 1
                if key==QtCore.Qt.Key_Comma:
                    self.frameNum -= frameShift
                    if self.frameNum<1:
                        self.frameNum = 1
                else:
                    self.frameNum += frameShift
                    if self.frameNum>self.numFrames:
                        self.frameNum = self.numFrames
                self.frameNumSpinBox.setValue(self.frameNum)
        elif key==QtCore.Qt.Key_N:
            if int(modifiers & QtCore.Qt.ControlModifier)>0:
                self.setDataNan = not self.setDataNan
            elif self.cam is None and not any([button.isChecked() for button in self.buttons]):
                self.setCurrentFrameDataNan()
        elif key in (QtCore.Qt.Key_Left,QtCore.Qt.Key_Right,QtCore.Qt.Key_Up,QtCore.Qt.Key_Down,QtCore.Qt.Key_Minus,QtCore.Qt.Key_Equal):
            if key in (QtCore.Qt.Key_Left,QtCore.Qt.Key_Right) and self.cam is None and not any([button.isChecked() for button in self.buttons]) and self.selectedSaccade is not None:
                move = -1 if key==QtCore.Qt.Key_Left else 1
                selSaccade = self.getSaccadesOnDisplay()[self.selectedSaccade]
                for saccades in (self.negSaccades,self.posSaccades):
                    if selSaccade in saccades:
                        saccades[saccades==selSaccade] += move
                        break
                self.plotSaccades()
            else:
                if self.roiButton.isChecked():
                    roi = self.roi
                elif self.findPupilButton.isChecked and self.pupilRoi is not None:
                    roi = self.pupilRoi
                elif self.findReflectButton.isChecked() and len(self.reflectRoi)>0:
                    roi = self.reflectRoi[-1]
                elif self.setMaskButton.isChecked() and len(self.maskRoi)>0:
                    roi = self.maskRoi[-1]
                else:
                    return
                roiPos = [int(n) for n in roi.pos()]
                roiSize = [int(n) for n in roi.size()]
                if key in (QtCore.Qt.Key_Left,QtCore.Qt.Key_Equal):
                    if roiPos[0]>0:
                        roiPos[0] -= 1
                elif key in (QtCore.Qt.Key_Right,QtCore.Qt.Key_Minus):
                    if (roiPos[0]+roiSize[0])<self.roiSize[0]:
                        roiPos[0] += 1
                if key in (QtCore.Qt.Key_Up,QtCore.Qt.Key_Equal):
                    if roiPos[1]>0:
                        roiPos[1] -= 1
                elif key in (QtCore.Qt.Key_Down,QtCore.Qt.Key_Minus):
                    if (roiPos[1]+roiSize[1])<self.roiSize[1]:
                        roiPos[1] += 1
                if key==QtCore.Qt.Key_Minus:
                    roiSize = [n-2 for n in roiSize]
                elif key==QtCore.Qt.Key_Equal:
                    if (roiPos[0]+roiSize[0])<self.roiSize[0]:
                        roiSize[0] += 2
                    if (roiPos[1]+roiSize[1])<self.roiSize[1]:
                        roiSize[1] += 2
                roi.setPos(roiPos)
                roi.setSize(roiSize)
        elif key==QtCore.Qt.Key_Space:
            if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                self.changePlotWindowDur(fullRange=True)
        elif key==QtCore.Qt.Key_Delete:
            if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                if int(modifiers & QtCore.Qt.ControlModifier)>0:
                    self.deleteAllSaccades()
                elif self.selectedSaccade is not None:
                    saccades = self.getSaccadesOnDisplay()
                    self.negSaccades = self.negSaccades[self.negSaccades!=saccades[self.selectedSaccade]]
                    self.posSaccades = self.posSaccades[self.posSaccades!=saccades[self.selectedSaccade]]
                    self.selectedSaccade = None
                    self.plotSaccades()
            elif self.setMaskButton.isChecked() and len(self.maskRoi)>0:
                self.imageViewBox.removeItem(self.maskRoi[-1])
                del(self.maskRoi[-1])
                del(self.maskIndex[-1])
        elif key==QtCore.Qt.Key_F:
            if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                self.findSaccades()
    
    def imageMouseClickEvent(self,event):
        if event.button()==QtCore.Qt.RightButton and not self.roiButton.isChecked() and not self.findReflectButton.isChecked() and self.reflectCenterSeed is not None:
            x,y = event.pos().x(),event.pos().y()
            self.reflectRoiPos = [[int(x-self.reflectRoiSize[0][0]/2),int(y-self.reflectRoiSize[0][1]/2)]]
            if not self.startVideoButton.isChecked():
                self.trackReflect()
                if self.reflectFound:
                    self.reflectCenterPlot.setData(x=[self.reflectCenterSeed[0]],y=[self.reflectCenterSeed[1]])
                else:
                    self.reflectCenterPlot.setData(x=[],y=[])
            
    def imageDoubleClickEvent(self,event):
        x,y = event.pos().x(),event.pos().y()
        if self.findReflectButton.isChecked():
            n = len(self.reflectRoi)
            if n<1 or (self.trackMenuReflectTypeRing.isChecked() and n<4):
                if n<1:
                    roiSize = (math.ceil(0.1*max(self.roiSize)),)*2
                else:
                    roiSize = self.reflectRoi[0].size()
                self.reflectRoi.append(pg.ROI((int(x-roiSize[0]/2),int(y-roiSize[1]/2)),roiSize,pen='r'))
                self.reflectRoi[-1].addScaleHandle(pos=(1,1),center=(0.5,0.5))
                self.imageViewBox.addItem(self.reflectRoi[-1])
        elif self.setMaskButton.isChecked():
            roiSize = math.ceil(0.1*max(self.roiSize))
            self.maskRoi.append(pg.ROI((int(x-roiSize/2),int(y-roiSize/2)),(roiSize,)*2,pen='r'))
            self.maskRoi[-1].addScaleHandle(pos=(1,1),center=(0.5,0.5))
            self.imageViewBox.addItem(self.maskRoi[-1])
        elif not self.roiButton.isChecked() and (self.findPupilButton.isChecked() or self.pupilCenterSeed is not None):
            self.pupilCenterSeed = (x,y)
            if (self.findPupilButton.isChecked() or not self.startVideoButton.isChecked()) and self.trackMenuPupilMethodStarburst.isChecked():
                self.trackPupil()
                self.updatePupilPlot()
                if self.findPupilButton.isChecked():
                    self.updatePupilTrackParamPlots()
                else:
                    self.updatePupilDataPlot()
                    
    def dataPlotMouseClickEvent(self,event):
        if event.button()==QtCore.Qt.RightButton and self.cam is None and not any([button.isChecked() for button in self.buttons]) and (self.negSaccades.size>0 or self.posSaccades.size>0):
            pos = self.pupilXPlotItem.getViewBox().mapSceneToView(event.pos())
            frame = int(round(pos.x()*self.frameRate))+1
            saccades = self.getSaccadesOnDisplay()
            if saccades.size>0:
                self.selectedSaccade = np.argmin(np.absolute(saccades-frame))
            
    def dataPlotDoubleClickEvent(self,event):
        if self.cam is None and not any([button.isChecked() for button in self.buttons]):
            pos = self.pupilXPlotItem.getViewBox().mapSceneToView(event.pos())
            frame = int(round(pos.x()*self.frameRate))+1
            vel,_ = self.getPupilVelocity()
            n = self.saccadeSmoothPts//2
            vel = vel[frame-n:frame-n+self.saccadeSmoothPts]
            maxInd = np.argmax(np.absolute(vel))
            if not np.isnan(vel[maxInd]):
                frame += maxInd-n
                if vel[maxInd]<0:
                    self.negSaccades = np.unique(np.concatenate((self.negSaccades,[frame])))
                else:
                    self.posSaccades = np.unique(np.concatenate((self.posSaccades,[frame])))
                saccades = self.getSaccadesOnDisplay()
                self.selectedSaccade = np.where(saccades==frame)
                self.plotSaccades()
        
    def setROI(self):
        if self.roiButton.isChecked():
            self.turnOffButtons(source=self.roiButton)
            maxBoundsRect = self.imageViewBox.itemBoundingRect(self.imageItem)
            if self.roi is None:
                self.roi = pg.ROI((0,0),self.fullRoiSize,maxBounds=maxBoundsRect,pen='r')
                self.roi.addScaleHandle(pos=(1,1),center=(0.5,0.5))
                self.imageViewBox.addItem(self.roi)
            else:
                self.roi.setPos((0,0))
                self.roi.setSize(self.fullRoiSize)
                self.roi.setVisible(True)
            self.pupilCenter += self.roiPos
            self.reflectCenter += self.roiPos
            if self.pupilCenterSeed is not None:
                self.pupilCenterSeed = (self.pupilCenterSeed[0]+self.roiPos[0],self.pupilCenterSeed[1]+self.roiPos[1])
            if self.reflectCenterSeed is not None:
                self.reflectCenterSeed = (self.reflectCenterSeed[0]+self.roiPos[0],self.reflectCenterSeed[1]+self.roiPos[1])
            if self.pupilRoi is not None:
                self.pupilRoiPos[0] += self.roiPos[0]
                self.pupilRoiPos[1] += self.roiPos[1]
            for i,roi in enumerate(self.reflectRoi):
                self.reflectRoiPos[i][0] += self.roiPos[0]
                self.reflectRoiPos[i][1] += self.roiPos[1]
            if len(self.maskRoi)>0:
                for roi in self.maskRoi:
                    roi.setPos((roi.pos()[0]+self.roiPos[0],roi.pos()[1]+self.roiPos[1]))
                self.updateMaskIndex()
            self.resetROI()
            self.resetImage()
        else:
            self.roiPos = [int(n) for n in self.roi.pos()]
            self.roiSize = [int(n) for n in self.roi.size()]
            self.pupilCenter -= self.roiPos
            self.reflectCenter -= self.roiPos
            if self.pupilCenterSeed is not None:
                self.pupilCenterSeed = (self.pupilCenterSeed[0]-self.roiPos[0],self.pupilCenterSeed[1]-self.roiPos[1])
            if self.reflectCenterSeed is not None:
                self.reflectCenterSeed = (self.reflectCenterSeed[0]-self.roiPos[0],self.reflectCenterSeed[1]-self.roiPos[1])
            if self.pupilRoi is not None:
                self.pupilRoiPos[0] -= self.roiPos[0]
                self.pupilRoiPos[1] -= self.roiPos[1]
            for i,roi in enumerate(self.reflectRoi):
                self.reflectRoiPos[i][0] -= self.roiPos[0]
                self.reflectRoiPos[i][1] -= self.roiPos[1]
            if len(self.maskRoi)>0:
                for roi in self.maskRoi:
                    roi.setPos((roi.pos()[0]-self.roiPos[0],roi.pos()[1]-self.roiPos[1]))
                self.updateMaskIndex()       
            if self.cam is None:
                self.roiInd = np.s_[self.roiPos[1]:self.roiPos[1]+self.roiSize[1],self.roiPos[0]:self.roiPos[0]+self.roiSize[0]]
            else:
                self.roiInd = np.s_[0:self.roiSize[1],0:self.roiSize[0]]
                self.cam.OffsetX = self.roiPos[0]
                self.cam.OffsetY = self.roiPos[1]
                self.cam.Width = self.roiSize[0]
                self.cam.Height = self.roiSize[1]
                self.getCamImage()
            self.roi.setVisible(False)
            self.resetImage()
            
    def setMask(self):
        if self.setMaskButton.isChecked():
            self.turnOffButtons(source=self.setMaskButton)
            for roi in self.maskRoi:
                roi.setVisible(True)
        else:
            for roi in self.maskRoi:
                roi.setVisible(False)
            self.updateMaskIndex()
            if self.pupilCenterSeed is not None:
                self.trackPupil()
                self.updatePupilPlot()
                self.updatePupilDataPlot()
                
    def updateMaskIndex(self):
        self.maskIndex = []
        for roi in self.maskRoi:
            x,y = [int(n) for n in roi.pos()]
            w,h = [int(n) for n in roi.size()]
            self.maskIndex.append(np.s_[y:y+h,x:x+w])
     
    def findPupil(self):
        if self.findPupilButton.isChecked():
            self.turnOffButtons(source=self.findPupilButton)
            if not self.trackMenuPupilMethodGradients.isChecked():
                if self.pupilCenterSeed is None:
                    self.pupilEdgeThresh = 2*self.image[self.roiInd][self.image[self.roiInd]>0].min()
                    self.minNumPixAboveThresh = 2
                    self.edgeFilt = np.ones(self.minNumPixAboveThresh)
                    self.edgeDistThresh = np.array([-6,6])
                self.pupilAreaPlot.setData(x=[],y=[])
                self.pupilXPlot.setData(x=[],y=[])
                self.pupilYPlot.setData(x=[],y=[])
                if self.cam is None:
                    for line in self.frameNumLines:
                        line.setVisible(False)
                self.pupilAreaPlotItem.addItem(self.pupilEdgeThreshLine)
                self.pupilEdgeThreshLine.setValue(self.pupilEdgeThresh)
                self.pupilXPlotItem.addItem(self.numPixAboveThreshLine)
                self.numPixAboveThreshLine.setValue(self.minNumPixAboveThresh)
                self.pupilYPlotItem.addItem(self.edgeDistUpperThreshLine)
                self.pupilYPlotItem.addItem(self.edgeDistLowerThreshLine)
                self.pupilAreaPlotItem.setLabel('left','Pixel Intensity')
                self.pupilXPlotItem.setLabel('left','Pixels > Thresh')
                self.pupilYPlotItem.setLabel('left','Pupil Edge Dist')
                self.pupilYPlotItem.setLabel('bottom','')
                if self.pupilCenterSeed is not None:
                    self.updatePupilTrackParamPlots()
            if not self.trackMenuPupilMethodStarburst.isChecked():
                if self.pupilRoi is None:
                    maxBoundsRect = self.imageViewBox.itemBoundingRect(self.imageItem)
                    self.pupilRoi = pg.ROI((0,0),self.fullRoiSize,maxBounds=maxBoundsRect,pen='r')
                    self.pupilRoi.addScaleHandle(pos=(1,1),center=(0.5,0.5))
                    self.pupilRoi.sigRegionChangeFinished.connect(self.pupilRoiRegionChanged)
                    self.imageViewBox.addItem(self.pupilRoi)
                else:
                    self.pupilRoi.setPos(self.pupilRoiPos)
                    self.pupilRoi.setSize(self.pupilRoiSize)
                    self.pupilRoi.setVisible(True)
        else:
            if not self.trackMenuPupilMethodGradients.isChecked():
                self.pupilEdgePtsPlot.setData(x=[],y=[])
                for i in range(len(self.radialProfilePlot)):
                    self.radialProfilePlot[i].setData(x=[],y=[])
                    self.radialProfilePixAboveThreshPlot[i].setData(x=[],y=[])
                self.edgeDistPlot.setData(x=[],y=[])
                self.pupilAreaPlotItem.removeItem(self.pupilEdgeThreshLine)
                self.pupilXPlotItem.removeItem(self.numPixAboveThreshLine)
                self.pupilYPlotItem.removeItem(self.edgeDistUpperThreshLine)
                self.pupilYPlotItem.removeItem(self.edgeDistLowerThreshLine)
                if self.cam is None:
                    for line in self.frameNumLines:
                        line.setVisible(True)
                self.pupilAreaPlotItem.setLabel('left','Pupil Area')
                self.pupilXPlotItem.setLabel('left','Pupil X')
                self.pupilYPlotItem.setLabel('left','Pupil Y')
                self.pupilYPlotItem.setLabel('bottom','Time (s)')
            if not self.trackMenuPupilMethodStarburst.isChecked():
                self.pupilRoi.setVisible(False)
            if self.pupilCenterSeed is not None:
                self.pupilAreaRange = [self.pupilArea[self.dataPlotIndex]]*2
                self.pupilXRange = [self.pupilX[self.dataPlotIndex]]*2
                self.pupilYRange = [self.pupilY[self.dataPlotIndex]]*2
                self.setDataPlotXRange()
                self.updatePupilDataPlot()
                
    def pupilRoiRegionChanged(self):
        self.pupilRoiPos = [int(n) for n in self.pupilRoi.pos()]
        self.pupilRoiSize = [int(n) for n in self.pupilRoi.size()]
        self.trackPupil()
        self.updatePupilPlot()
        if self.trackMenuPupilMethodLine.isChecked():
            self.updatePupilTrackParamPlots()
                
    def trackPupil(self):
        self.pupilFound = False
        if not self.setDataNan:
            img = self.image[self.roiInd].copy()
            if self.trackMenuPupilSignPos.isChecked():
                img = 255-img
            if self.useMaskCheckBox.isChecked() and len(self.maskRoi)>0:
                for ind in self.maskIndex:
                    img[ind] = 0
            if self.reflectCenterSeed is None or self.reflectFound:
                if self.trackMenuPupilMethodStarburst.isChecked():
                    self.findPupilWithStarburst(img)
                elif self.trackMenuPupilMethodLine.isChecked():
                    self.findPupilWithLine(img)
                else:
                    self.findPupilWithGradients(img)
        self.updatePupilData()
        
    def findPupilWithStarburst(self,img):
        # get radial profiles and find pupil edges
        # radial profile must cross edge thresh for min num of consecutive pix
        # radial profile = 0 for masked pixels
        if 0<self.pupilCenterSeed[0]<self.roiSize[0]-1 and 0<self.pupilCenterSeed[1]<self.roiSize[1]-1:
            x = self.radialLinesX+int(self.pupilCenterSeed[0])
            y = self.radialLinesY+int(self.pupilCenterSeed[1])
            inFrame = np.logical_and(np.logical_and(x>=0,x<self.roiSize[0]),np.logical_and(y>=0,y<self.roiSize[1]))
            self.radialProfiles[:] = 0
            self.pupilEdges = np.zeros((self.numRadialLines*2,2),dtype=np.float32)
            for i in range(self.numRadialLines):
                xInFrame = x[i,inFrame[i,:]]
                yInFrame = y[i,inFrame[i,:]]
                lineProfile = img[yInFrame,xInFrame]
                centerInd = np.where(np.logical_and(xInFrame==int(self.pupilCenterSeed[0]),yInFrame==int(self.pupilCenterSeed[1])))[0][0]
                self.radialProfiles[i,:lineProfile.size-centerInd] = lineProfile[centerInd:]
                self.radialProfiles[i+self.numRadialLines,:centerInd+1] = lineProfile[centerInd::-1]
                edgeInd1 = self.findPupilEdgeIndex(self.radialProfiles[i])
                edgeInd2 = self.findPupilEdgeIndex(self.radialProfiles[i+self.numRadialLines])
                if edgeInd1 is not None:
                    self.pupilEdges[i,0] = xInFrame[centerInd+edgeInd1]
                    self.pupilEdges[i,1] = yInFrame[centerInd+edgeInd1]
                if edgeInd2 is not None:
                    self.pupilEdges[i+self.numRadialLines,0] = xInFrame[centerInd-edgeInd2]
                    self.pupilEdges[i+self.numRadialLines,1] = yInFrame[centerInd-edgeInd2]
            # throw out edge points with outlier distances from center
            self.pupilEdges = self.pupilEdges[self.pupilEdges.any(axis=1)]
            if self.pupilEdges.shape[0]>0:
                self.pupilEdgeDist = np.sqrt(np.sum((self.pupilEdges-self.pupilCenterSeed)**2,axis=1))
                normDist = self.pupilEdgeDist-self.pupilEdgeDist.mean()
                distThresh = self.edgeDistThresh*self.pupilEdgeDist.std()
                self.pupilEdges = self.pupilEdges[np.logical_and(normDist>distThresh[0],normDist<distThresh[1])]
                # fit ellipse to edge points
                if self.pupilEdges.shape[0]>4:
                    center,diameter,angle = cv2.fitEllipse(self.pupilEdges)
                    if 0<center[0]<self.roiSize[0]-1 and 0<center[1]<self.roiSize[1]-1 and diameter[1]>0 and diameter[0]/diameter[1]>self.pupilCircularityThresh:
                        self.pupilCenterSeed,self.pupilEllipseRadii,self.pupilEllipseAngle = center,[d/2 for d in diameter],angle
                        self.pupilFound = True
        
    def getRadialLines(self):
        angles = np.arange(0,90,20)
        slopes = np.append(np.nan,1/np.tan(np.radians(angles[1:])))
        self.numRadialLines = 2*angles.size-1
        maxLength = max(self.roiSize)
        self.radialLinesX = np.zeros((self.numRadialLines,maxLength*2),dtype=np.int16)
        self.radialLinesY = np.zeros((self.numRadialLines,maxLength*2),dtype=np.int16)
        for i,angle in enumerate(angles):
            if angle==0:
                self.radialLinesY[i] = np.arange(-maxLength,maxLength)+1
            elif angle==90:
                self.radialLinesX[i] = np.arange(-maxLength,maxLength)+1
            elif angle==45:
                self.radialLinesX[i] = np.arange(-maxLength,maxLength)+1
                self.radialLinesY[i] = np.arange(-maxLength,maxLength)+1
            elif angle<45:
                self.radialLinesY[i] = np.arange(-maxLength,maxLength)+1
                self.radialLinesX[i] = self.radialLinesY[i,:]/slopes[i] # x = y/m
            elif angle>45:
                self.radialLinesX[i] = np.arange(-maxLength,maxLength)+1
                self.radialLinesY[i] = slopes[i]*self.radialLinesX[i,:] # y = mx
        self.radialLinesX[angles.size:] = self.radialLinesX[1:angles.size]
        self.radialLinesY[angles.size:] = -self.radialLinesY[1:angles.size]
        self.radialProfiles = np.zeros((self.numRadialLines*2,max(self.roiSize)),dtype=np.uint8)
        
    def findPupilEdgeIndex(self,lineProfile):
        if self.minNumPixAboveThresh>1:
            edgeInd = np.where(np.correlate(lineProfile>self.pupilEdgeThresh,self.edgeFilt,mode='valid')==self.minNumPixAboveThresh)[0]
        else:
            edgeInd = np.where(lineProfile>self.pupilEdgeThresh)[0]
        edgeInd = edgeInd[0] if edgeInd.size>0 else None
        return edgeInd
        
    def findPupilWithLine(self,img):
        if self.reflectCenterSeed is not None:
            self.pupilRoiPos[1] 
        self.radialProfiles[:] = 0
        lineProfile = img[self.pupilRoiPos[1]:self.pupilRoiPos[1]+self.pupilRoiSize[1],self.pupilRoiPos[0]:self.pupilRoiPos[0]+self.pupilRoiSize[0]].mean(axis=0)
        if self.trackMenuLineOriginLeft.isChecked():
            self.radialProfiles[0,:lineProfile.size] = lineProfile
            edgeInd = self.findPupilEdgeIndex(lineProfile)
        else:
            self.radialProfiles[0,:lineProfile.size] = lineProfile[::-1]
            edgeInd = self.findPupilEdgeIndex(lineProfile[::-1])
            if edgeInd is not None:
                edgeInd = lineProfile.size-1-edgeInd
        if edgeInd is not None:
            edgeInd += self.pupilRoiPos[0]
            if not self.findPupilButton.isChecked() and self.pupilCenterSeed is not None:
                self.pupilRoiPos[0] += edgeInd-self.pupilCenterSeed[0]
            self.pupilCenterSeed = [edgeInd,self.pupilRoiPos[1]+self.pupilRoiSize[1]//2]
            self.pupilFound = True
        
    def findPupilWithGradients(self,img):
        # method described by:
        # Timm and Barth. Accurate eye centre localisation by means of gradients.
        # In Proceedings of the Int. Conference on Computer Theory and Applications
        # (VISAPP), volume 1, pages 125-130, Algarve, Portugal, 2011.
        # with further details in Tristan Hume's blogpost Nov 4, 2012:
        # http://thume.ca/projects/2012/11/04/simple-accurate-eye-center-tracking-in-opencv/
        img = img[self.pupilRoiPos[1]:self.pupilRoiPos[1]+self.pupilRoiSize[1],self.pupilRoiPos[0]:self.pupilRoiPos[0]+self.pupilRoiSize[0]]
        img[img>230] = 0
        if self.pupilGradientDownsample<1:
            img = cv2.resize(img,(0,0),fx=self.pupilGradientDownsample,fy=self.pupilGradientDownsample,interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(img,(0,0),0.005*img.shape[1])
        gradY,gradX = np.gradient(img.astype(float))
        gradLength = np.sqrt(gradX**2+gradY**2)
        gradIndex = gradLength>np.mean(gradLength)+0.3*np.std(gradLength)
        x = np.arange(img.shape[1],dtype=float)
        y = np.arange(img.shape[0],dtype=float)
        meshX,meshY = np.meshgrid(x,y)
        distX = np.tile(np.ravel(meshX[gradIndex])-x[:,None],(img.shape[0],1))
        distY = np.repeat(np.ravel(meshY[gradIndex])-y[:,None],img.shape[1],axis=0)
        distLength = np.sqrt(distX**2+distY**2)
        distLength[distLength==0] = 1
        # dot = ((distX/distLength)*(gradX[gradIndex]/gradLength[gradIndex]))+((distY/distLength)*(gradY[gradIndex]/gradLength[gradIndex]))
        # use in place array manipulations
        distX /= distLength
        distY /= distLength
        gradLength = gradLength[gradIndex]
        gradX = gradX[gradIndex]
        gradY = gradY[gradIndex]
        gradX /= gradLength
        gradY /= gradLength
        distX *= gradX
        distY *= gradY
        distX += distY
        dot = distX
        dot.clip(min=0,out=dot)
        # equation 3 in Timm and Barth 2011
        centerWeight = (255-img)[gradIndex]
        dot **= 2
        dot *= centerWeight
        f = np.reshape(dot.sum(axis=1)/dot.shape[1],img.shape)
        # remove high f regions connected to edge
        _,contours,_ = cv2.findContours((f>0.9*f.max()).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>1:
            mask = np.zeros_like(img)
            for i,c in enumerate(contours):
                if np.in1d(c[:,0,0],[1,img.shape[1]-2]).any() or np.in1d(c[:,0,1],[1,img.shape[0]-2]).any():
                    cv2.drawContours(mask,contours,i,1,-1)
            f[mask.astype(np.bool)] = 0
        f[:,[0,-1]] = 0
        f[[0,-1],:] = 0
        center = np.unravel_index(f.argmax(),f.shape)[::-1]
        self.pupilCenterSeed = [int(center[i]/self.pupilGradientDownsample)+self.pupilRoiPos[i] for i in (0,1)]
        self.pupilFound = True
        
    def setPupilSign(self):
        if self.mainWin.sender() is self.trackMenuPupilSignNeg:
            self.trackMenuPupilSignNeg.setChecked(True)
            self.trackMenuPupilSignPos.setChecked(False)
        else:
            self.trackMenuPupilSignNeg.setChecked(False)
            self.trackMenuPupilSignPos.setChecked(True)
        self.pupilCenterSeed = None
        self.pupilCenterPlot.setData(x=[],y=[])
        self.pupilEllipsePlot.setData(x=[],y=[])
        
    def setPupilTrackMethod(self):
        methods = (self.trackMenuPupilMethodStarburst,self.trackMenuPupilMethodLine,self.trackMenuPupilMethodGradients) 
        params = (self.trackMenuCircularity,self.trackMenuLineOrigin,self.trackMenuGradientDownsamp)
        for m,p in zip(methods,params):
            isSelected = m is self.mainWin.sender()
            m.setChecked(isSelected)
            p.setEnabled(isSelected)
        self.pupilCenterSeed = None
        self.pupilCenterPlot.setData(x=[],y=[])
        self.pupilEllipsePlot.setData(x=[],y=[])
        
    def setCircularityThresh(self):
        val,ok = QtGui.QInputDialog.getDouble(self.mainWin,'Set pupil circularity threshold','ellipse axis length ratio:',value=self.pupilCircularityThresh,min=0.01,max=0.99,decimals=2)
        if ok:
            self.pupilCircularityThresh = val
        
    def setPupilEdgeLineOrigin(self):
        if self.mainWin.sender() is self.trackMenuLineOriginLeft:
            self.trackMenuLineOriginLeft.setChecked(True)
            self.trackMenuLineOriginRight.setChecked(False)
        else:
            self.trackMenuLineOriginLeft.setChecked(False)
            self.trackMenuLineOriginRight.setChecked(True)
        
    def setPupilGradientDownsample(self):
        val,ok = QtGui.QInputDialog.getDouble(self.mainWin,'Set pupil gradient downsample','fraction of pixels:',value=self.pupilGradientDownsample,min=0.1,max=1,decimals=2)
        if ok:
            self.pupilGradientDownsample = val
        
    def updatePupilPlot(self):
        if self.pupilFound and (self.reflectCenterSeed is None or self.reflectFound):
            self.pupilCenterPlot.setData(x=[self.pupilCenterSeed[0]],y=[self.pupilCenterSeed[1]])
            if self.trackMenuPupilMethodStarburst.isChecked():
                angle = self.pupilEllipseAngle*math.pi/180
                sinx = np.sin(np.arange(0,370,10)*math.pi/180)
                cosx = np.cos(np.arange(0,370,10)*math.pi/180)
                self.pupilEllipsePlot.setData(x=self.pupilCenterSeed[0]+self.pupilEllipseRadii[0]*cosx*math.cos(angle)-self.pupilEllipseRadii[1]*sinx*math.sin(angle),
                                              y=self.pupilCenterSeed[1]+self.pupilEllipseRadii[0]*cosx*math.sin(angle)+self.pupilEllipseRadii[1]*sinx*math.cos(angle))
        else:
            self.pupilCenterPlot.setData(x=[],y=[])
            self.pupilEllipsePlot.setData(x=[],y=[])
            
    def updatePupilData(self):
        if self.cam is None:
            self.dataPlotIndex = self.frameNum-1
        else:
            leadingPtsInd = np.s_[self.dataPlotIndex+1:self.dataPlotIndex+math.ceil(self.numDataPlotPts/10)]
            self.pupilArea[leadingPtsInd] = np.nan
            self.pupilX[leadingPtsInd,:] = np.nan
            self.pupilY[leadingPtsInd,:] = np.nan
        if self.pupilFound and (self.reflectCenterSeed is None or self.reflectFound):
            self.pupilCenter[self.dataPlotIndex,:] = self.pupilCenterSeed
            if self.trackMenuPupilMethodStarburst.isChecked():
                self.pupilArea[self.dataPlotIndex] = math.pi*self.pupilEllipseRadii[1]**2
                if not np.isnan(self.mmPerPixel):
                    self.pupilArea[self.dataPlotIndex] *= self.mmPerPixel**2
            if self.reflectCenterSeed is None:
                self.pupilX[self.dataPlotIndex] = self.pupilCenterSeed[0]
                self.pupilY[self.dataPlotIndex] = self.pupilCenterSeed[1] 
            elif np.isnan(self.mmPerPixel) or self.trackMenuReflectTypeSpot.isChecked() or self.trackMenuPupilMethodGradients.isChecked():
                self.pupilX[self.dataPlotIndex] = self.pupilCenterSeed[0]-self.reflectCenterSeed[0]
                self.pupilY[self.dataPlotIndex] = self.pupilCenterSeed[1]-self.reflectCenterSeed[1]
            else:
                try:
                    pupilRotRadius = (self.lensRotRadius**2-(self.pupilEllipseRadii[1]*self.mmPerPixel)**2)**0.5-self.lensOffset
                    self.pupilX[self.dataPlotIndex],self.pupilY[self.dataPlotIndex] = [180/math.pi*math.asin((self.mmPerPixel*(self.reflectCenterSeed[i]-self.pupilCenterSeed[i])*pupilRotRadius/(pupilRotRadius-self.corneaOffset))/pupilRotRadius) for i in (0,1)]
                except:
                    self.pupilArea[self.dataPlotIndex] = np.nan
                    self.pupilX[self.dataPlotIndex] = np.nan
                    self.pupilY[self.dataPlotIndex] = np.nan 
        else:
            self.pupilArea[self.dataPlotIndex] = np.nan
            self.pupilX[self.dataPlotIndex] = np.nan
            self.pupilY[self.dataPlotIndex] = np.nan
        
    def updatePupilDataPlot(self,updatePlotN=[True,True,True]):
        if self.cam is None:
            if self.frameNum>self.numDataPlotPts:
                plotTime = self.dataPlotTime+(self.frameNum-self.numDataPlotPts)/self.frameRate
                dataPlotInd = np.s_[self.frameNum-self.numDataPlotPts:self.frameNum]
            else:
                plotTime = self.dataPlotTime
                dataPlotInd = np.s_[0:self.numDataPlotPts]
            if updatePlotN[0]:
                self.pupilAreaPlotItem.setXRange(plotTime[0],plotTime[-1])
            if updatePlotN[1]:
                self.pupilXPlotItem.setXRange(plotTime[0],plotTime[-1])
            if updatePlotN[2]:
                self.pupilYPlotItem.setXRange(plotTime[0],plotTime[-1])
        else:
            plotTime = self.dataPlotTime
            dataPlotInd = np.s_[0:self.numDataPlotPts]
        connectPts = np.logical_not(np.isnan(self.pupilX[dataPlotInd])).astype(np.uint32)
        if updatePlotN[0]:
            self.setDataPlotYRange(self.pupilAreaPlotItem,self.pupilAreaRange,np.nanmin(self.pupilArea[dataPlotInd]),np.nanmax(self.pupilArea[dataPlotInd]))
            self.pupilAreaPlot.setData(x=plotTime,y=self.pupilArea[dataPlotInd],connect=connectPts)
            if self.cam is None:
                self.pupilAreaFrameNumLine.setValue((self.frameNum-1)/self.frameRate)
        if updatePlotN[1]:
            self.setDataPlotYRange(self.pupilXPlotItem,self.pupilXRange,np.nanmin(self.pupilX[dataPlotInd]),np.nanmax(self.pupilX[dataPlotInd]))
            self.pupilXPlot.setData(x=plotTime,y=self.pupilX[dataPlotInd],connect=connectPts)
            if self.cam is None:
                self.pupilXFrameNumLine.setValue((self.frameNum-1)/self.frameRate)
        if updatePlotN[2]:
            self.setDataPlotYRange(self.pupilYPlotItem,self.pupilYRange,np.nanmin(self.pupilY[dataPlotInd]),np.nanmax(self.pupilY[dataPlotInd]))
            self.pupilYPlot.setData(x=plotTime,y=self.pupilY[dataPlotInd],connect=connectPts)
            if self.cam is None:
                self.pupilYFrameNumLine.setValue((self.frameNum-1)/self.frameRate)
    
    def setDataPlotXRange(self):
        self.pupilAreaPlotItem.setXRange(0,self.dataPlotDur)
        self.pupilXPlotItem.setXRange(0,self.dataPlotDur)
        self.pupilYPlotItem.setXRange(0,self.dataPlotDur)
        tickSpacing = self.getTickSpacing(self.dataPlotDur)
        self.pupilAreaPlotItem.getAxis('bottom').setTickSpacing(levels=[(tickSpacing,0)])
        self.pupilXPlotItem.getAxis('bottom').setTickSpacing(levels=[(tickSpacing,0)])
        self.pupilYPlotItem.getAxis('bottom').setTickSpacing(levels=[(tickSpacing,0)])
    
    def setDataPlotYRange(self,dataPlotItem,dataPlotRange,Ymin,Ymax):
        if not np.isnan(Ymin) and not np.isnan(Ymax):
            midRange = (dataPlotRange[1]-dataPlotRange[0])/2
            if not dataPlotRange[0]<Ymin<midRange or not midRange<Ymax<dataPlotRange[1]:
                dataPlotRange[0] = Ymin*0.8 if Ymin>0 else Ymin*1.2
                dataPlotRange[1] = Ymax*1.2 if Ymax>0 else Ymax*0.8
                dataPlotItem.setYRange(dataPlotRange[0],dataPlotRange[1])
                dataPlotItem.getAxis('left').setTickSpacing(levels=[(self.getTickSpacing(dataPlotRange[1]-dataPlotRange[0]),0)])
            
    def getTickSpacing(self,dataRange):
        spacing = 10**(math.floor(math.log10(dataRange)))
        spacing *= 0.5*(dataRange//spacing)
        return spacing
    
    def updatePupilTrackParamPlots(self):
        # recall trackPupil() first so that measurments reflect calculated pupil center
        self.trackPupil()
        self.updatePupilPlot()
        xmax = 0
        for i in range(len(self.radialProfilePlot)):
            if any(self.radialProfiles[i]):
                xmax = max([xmax,np.where(self.radialProfiles[i])[0][-1]])
            self.radialProfilePlot[i].setData(self.radialProfiles[i])
            self.radialProfilePixAboveThreshPlot[i].setData(np.correlate(self.radialProfiles[i]>self.pupilEdgeThresh,np.ones(self.minNumPixAboveThresh)))
        xTickSpacing = self.getTickSpacing(xmax)
        self.pupilAreaPlotItem.setRange(xRange=(0,xmax),yRange=(max(0,2*self.pupilEdgeThresh-255),min(255,2*self.pupilEdgeThresh)))
        self.pupilAreaPlotItem.getAxis('left').setTickSpacing(levels=[(self.getTickSpacing(self.pupilEdgeThresh*2),0)])
        self.pupilAreaPlotItem.getAxis('bottom').setTickSpacing(levels=[(xTickSpacing,0)])
        self.pupilXPlotItem.setRange(xRange=(0,xmax),yRange=(0,2*self.minNumPixAboveThresh))
        self.pupilXPlotItem.getAxis('left').setTickSpacing(levels=[(round(self.minNumPixAboveThresh/2),0)])
        self.pupilXPlotItem.getAxis('bottom').setTickSpacing(levels=[(xTickSpacing,0)])
        if self.trackMenuPupilMethodStarburst.isChecked():
            self.pupilEdgePtsPlot.setData(self.pupilEdges)
            if self.pupilEdges.shape[0]>0:
                self.edgeDistPlot.setData(x=np.arange(self.pupilEdgeDist.size)+1,y=self.pupilEdgeDist)
                self.edgeDistUpperThreshLine.setValue(self.pupilEdgeDist.mean()+self.edgeDistThresh[1]*self.pupilEdgeDist.std())
                self.edgeDistLowerThreshLine.setValue(self.pupilEdgeDist.mean()+self.edgeDistThresh[0]*self.pupilEdgeDist.std())
                self.pupilYPlotItem.setRange(xRange=(1,self.pupilEdgeDist.size),yRange=(0,max(np.append(self.pupilEdgeDist,self.edgeDistUpperThreshLine.value()))))
                self.pupilYPlotItem.getAxis('left').setTickSpacing(levels=[(self.getTickSpacing(self.pupilEdgeDist.mean()*2),0)])
                self.pupilYPlotItem.getAxis('bottom').setTickSpacing(levels=[(self.getTickSpacing(self.pupilEdges.shape[0]),0)])
        else:
            self.edgeDistPlot.setData(x=[],y=[])
            self.edgeDistUpperThreshLine.setValue(2)
            self.edgeDistLowerThreshLine.setValue(-1)
            self.pupilYPlotItem.setRange(xRange=(0,1),yRange=(0,1))
            self.pupilYPlotItem.getAxis('left').setTickSpacing(levels=[(1,0)])
            self.pupilYPlotItem.getAxis('bottom').setTickSpacing(levels=[(1,0)])
        
    def setPupilEdgeThresh(self):
        self.pupilEdgeThresh = self.pupilEdgeThreshLine.value()
        self.trackPupil()
        self.updatePupilPlot()
        self.updatePupilTrackParamPlots()
        
    def setMinNumPixAboveThresh(self):
        self.minNumPixAboveThresh = round(self.numPixAboveThreshLine.value())
        self.edgeFilt = np.ones(self.minNumPixAboveThresh)
        self.trackPupil()
        self.updatePupilPlot()
        self.updatePupilTrackParamPlots()
    
    def setEdgeDistThresh(self):
        meanEdgeDist = self.pupilEdgeDist.mean()
        upperThresh = self.edgeDistUpperThreshLine.value()
        if upperThresh<meanEdgeDist:
            upperThresh = math.ceil(1.2*meanEdgeDist)
        lowerThresh = self.edgeDistLowerThreshLine.value()
        if lowerThresh>meanEdgeDist:
            lowerThresh = math.floor(0.8*meanEdgeDist)
        self.edgeDistThresh = (np.array([lowerThresh,upperThresh])-meanEdgeDist)/self.pupilEdgeDist.std()
        self.trackPupil()
        self.updatePupilPlot()
        self.updatePupilTrackParamPlots()
        
    def findReflect(self):
        if self.findReflectButton.isChecked():
            self.turnOffButtons(source=self.findReflectButton)
            self.reflectCenterPlot.setData(x=[],y=[])
            for i,roi in enumerate(self.reflectRoi):
                roi.setPos(self.reflectRoiPos[i])
                roi.setSize(self.reflectRoiSize[i])
                roi.setVisible(True)
        elif len(self.reflectRoi)>0:
            self.reflectRoiPos = []
            self.reflectRoiSize = []
            for roi in self.reflectRoi:
                roi.setVisible(False)
                self.reflectRoiPos.append([int(n) for n in roi.pos()])
                self.reflectRoiSize.append([int(n) for n in roi.size()])
            if self.trackMenuReflectTypeSpot.isChecked() or len(self.reflectRoi)==4:
                if self.trackMenuReflectTypeRing.isChecked():
                    self.getReflectTemplate()
                    if not self.reflectFound:
                        return
                self.trackReflect()
                if self.reflectFound:
                    self.reflectCenterPlot.setData(x=[self.reflectCenterSeed[0]],y=[self.reflectCenterSeed[1]])
                    if self.pupilCenterSeed is not None:
                        self.updatePupilData()
                        self.updatePupilDataPlot()
            
    def trackReflect(self):
        self.reflectFound = False
        if not self.setDataNan:
            roiPos,roiSize = self.reflectRoiPos[0],self.reflectRoiSize[0]
            if self.trackMenuReflectTypeSpot.isChecked():
                y,x = np.where(self.image[self.roiInd][roiPos[1]:roiPos[1]+roiSize[1],roiPos[0]:roiPos[0]+roiSize[0]]>self.reflectThresh)
                if any(y):
                    self.reflectCenterSeed = (roiPos[0]+x.mean(),roiPos[1]+y.mean())
                else:
                    return
            else:
                y,x = np.unravel_index(np.argmax(scipy.signal.fftconvolve(self.image[self.roiInd][roiPos[1]:roiPos[1]+roiSize[1],roiPos[0]:roiPos[0]+roiSize[0]],self.reflectTemplate,mode='same')),roiSize)          
                center = (roiPos[0]+x,roiPos[1]+y)
                if any((center[i]-roiSize[i]<0 or center[i]+roiSize[i]>self.roiSize[i]-1 for i in [0,1])):
                    return
                self.reflectCenterSeed = center
            self.reflectRoiPos = [[int(self.reflectCenterSeed[0]-roiSize[0]/2),int(self.reflectCenterSeed[1]-roiSize[1]/2)]]
            self.reflectFound = True        
            if self.cam is None:
                self.reflectCenter[self.frameNum-1,:] = self.reflectCenterSeed
            else:
                self.reflectCenter[self.dataPlotIndex,:] = self.reflectCenterSeed
        
    def getReflectTemplate(self):
        spotCenters = np.zeros((4,2))
        ptsAboveThresh = []
        for i,(roiPos,roiSize) in enumerate(zip(self.reflectRoiPos,self.reflectRoiSize)):
            y,x = np.where(self.image[self.roiInd][roiPos[1]:roiPos[1]+roiSize[1],roiPos[0]:roiPos[0]+roiSize[0]]>self.reflectThresh)
            if any(y):
                spotCenters[i,:] = (roiPos[0]+x.mean(),roiPos[1]+y.mean())
                ptsAboveThresh = min(ptsAboveThresh,len(y))
            else:
                self.reflectFound = False
                return
        self.reflectFound = True
        for roi in self.reflectRoi[1:]:
            self.imageViewBox.removeItem(roi)
        del(self.reflectRoi[1:])
        self.reflectCenterSeed = spotCenters.mean(axis=0)
        roiSize = 4*int(max(spotCenters.max(axis=0)-spotCenters.min(axis=0)))
        self.reflectRoiSize = [[roiSize]*2]
        self.reflectTemplate = np.zeros((roiSize,)*2,dtype=bool)
        self.reflectRoiPos = [[int(self.reflectCenterSeed[0]-roiSize/2),int(self.reflectCenterSeed[1]-roiSize/2)]]
        spotCenters = (spotCenters-(self.reflectCenterSeed-roiSize/2)).astype(int)
        m,n = int(ptsAboveThresh/2),int(round(ptsAboveThresh/2))
        for center in spotCenters:     
            self.reflectTemplate[center[1]-m:center[1]+n,center[0]-m:center[0]+n] = True
            
    def setReflectType(self):
        if self.mainWin.sender() is self.trackMenuReflectTypeSpot:
            self.trackMenuReflectTypeSpot.setChecked(True)
            self.trackMenuReflectTypeRing.setChecked(False)
        else:
            self.trackMenuReflectTypeSpot.setChecked(False)
            self.trackMenuReflectTypeRing.setChecked(True)
        if len(self.reflectRoi)>0:
            for roi in self.reflectRoi:
                self.imageViewBox.removeItem(roi)
            self.reflectRoi = []
            self.reflectCenterSeed = None
            self.reflectCenterPlot.setData(x=[],y=[])
            
    def setReflectThresh(self):
        val,ok = QtGui.QInputDialog.getInt(self.mainWin,'Set Reflection Threshold','Pixel intensity:',value=self.reflectThresh,min=0,max=254)
        if ok:
            self.reflectThresh = val
        
    def turnOffButtons(self,source=None):
        for button in self.buttons:
            if button is not source and button.isChecked():
                button.click()
                
    def addFrameNumLines(self):
        self.pupilAreaPlotItem.addItem(self.pupilAreaFrameNumLine)
        self.pupilXPlotItem.addItem(self.pupilXFrameNumLine)
        self.pupilYPlotItem.addItem(self.pupilYFrameNumLine)
        
    def removeFrameNumLines(self):
        self.pupilAreaPlotItem.removeItem(self.pupilAreaFrameNumLine)
        self.pupilXPlotItem.removeItem(self.pupilXFrameNumLine)
        self.pupilYPlotItem.removeItem(self.pupilYFrameNumLine)
        
    def frameNumLineDragged(self):
        source = self.mainWin.sender()
        for line in self.frameNumLines:
            if line is not source:
                line.setValue(source.value())
    
    def frameNumLinePosChangeFin(self):
        source = self.mainWin.sender()
        self.frameNumSpinBox.setValue(round(source.value()*self.frameRate)+1)
                
    def goToFrame(self):
        self.frameNum = self.frameNumSpinBox.value()
        self.getVideoImage()
        self.updateDisplay(updateAll=True)
    
    def changePlotWindowDur(self,fullRange=False):
        if fullRange:
            newVal = self.numFrames/self.frameRate
        else:
            newVal = float(self.plotDurEdit.text())
            if newVal<3/self.frameRate:
                newVal = 3/self.frameRate
            elif self.cam is None and newVal>self.numFrames/self.frameRate:
                newVal = self.numFrames/self.frameRate
        self.plotDurEdit.setText(str(round(newVal,3)))
        self.dataPlotDur = newVal
        if self.cam is not None:
            self.resetPupilData()
        self.setDataPlotTime()
        self.setDataPlotXRange()
        if all(np.isnan(self.pupilX)):
            self.resetPupilDataPlot()
        else:
            self.updatePupilDataPlot()
            
    def setDataPlotTime(self):
        self.dataPlotTime = np.arange(0,self.dataPlotDur-0.5/self.frameRate,1/self.frameRate)
        self.numDataPlotPts = self.dataPlotTime.size
        
    def setMmPerPix(self):
        val = 0 if np.isnan(self.mmPerPixel) else self.mmPerPixel
        val,ok = QtGui.QInputDialog.getDouble(self.mainWin,'Set mm/pixel','mm/pixel:',value=val,min=0,decimals=4)
        if ok:
            self.mmPerPixel = val if val>0 else np.nan
    
    def measureMmPerPix(self):
        if self.reflectCenterSeed is None:
            QtGui.QMessageBox.about(self.mainWin,'Set mm/pixel','First find reflection')
        else:
            p = int(0.5*self.frameRate)
            avgPts = p if self.numDataPlotPts>=p else self.numDataPlotPts
            initialReflectCenter = self.getAvgReflectCenter(avgPts)
            QtGui.QMessageBox.about(self.mainWin,'Set mm/pixel','Move camera 0.5 mm; then press ok')
            finalReflectCenter = self.getAvgReflectCenter(avgPts)
            self.mmPerPixel = 0.5/np.sqrt(np.sum((finalReflectCenter-initialReflectCenter)**2))
            
    def getAvgReflectCenter(self,avgPts):
        return np.mean(np.tile(self.reflectCenter,(2,1))[:self.numDataPlotPts+self.dataPlotIndex,:][-avgPts:,:],axis=0)
            
    def analyzeAllFrames(self):
        while self.frameNum<self.numFrames:
            self.frameNum += 1
            self.getVideoImage()
            if self.frameNum==self.numFrames:
                self.updateDisplay(updateAll=True)
                self.changePlotWindowDur(fullRange=True)
            else:
                self.updateDisplay(updateNone=True)
                
    def loadTrackingData(self):
        filePath = QtGui.QFileDialog.getOpenFileName(self.mainWin,'Choose File',self.fileOpenSavePath,'Files (*.hdf5 *.npz *.mat)')
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        fileType = os.path.splitext(filePath)[1][1:]
        params = ('mmPerPixel','frameTimes','reflectCenter','pupilCenter','pupilArea','pupilX','pupilY','negSaccades','posSaccades')
        if fileType=='hdf5':
            dataFile = h5py.File(filePath,'r')
            if 'mmPerPixel' in dataFile.attrs.keys():
                self.mmPerPixel = dataFile.attrs.get('mmPerPixel')
            for param in set(dataFile.keys()) & set(params[1:]):
                setattr(self,param,dataFile[param][:])
            dataFile.close()
        else:
            data = np.load(filePath) if fileType=='npz' else scipy.io.loadmat(filePath,squeeze_me=True)
            for param in set(data.keys()) & set(params):
                setattr(self,param,data[param])
        self.dataIsLoaded = True
        self.reflectCenter -= self.roiPos
        self.pupilCenter -= self.roiPos
        self.pupilAreaRange = [self.pupilArea[self.frameNum-1]]*2
        self.pupilXRange = [self.pupilX[self.frameNum-1]]*2
        self.pupilYRange = [self.pupilY[self.frameNum-1]]*2
        self.setDataPlotXRange()
        self.updatePupilDataPlot()
        self.plotSaccades()
        
    def pixToDeg(self):
        pupilEllipseRadius = (self.pupilArea/math.pi)**0.5
        pupilRotRadius = (self.lensRotRadius**2-(pupilEllipseRadius*self.mmPerPixel)**2)**0.5-self.lensOffset
        self.pupilX,self.pupilY = [180/math.pi*np.arcsin((self.mmPerPixel*(self.reflectCenter[:,i]-self.pupilCenter[:,i])*pupilRotRadius/(pupilRotRadius-self.corneaOffset))/pupilRotRadius) for i in (0,1)]
        self.pupilArea *= self.mmPerPixel**2
        self.updatePupilDataPlot()
                    
    def degToPix(self):
        self.pupilArea /= self.mmPerPixel**2
        self.pupilX,self.pupilY = [self.pupilCenter[:,i]-self.reflectCenter[:,i] for i in (0,1)]
        self.updatePupilDataPlot()
    
    def plotFrameIntervals(self):
        if all(np.isnan(self.frameTimes)):
            return
        self.getFrameTimes()
        frameIntervals = np.diff(self.frameTimes)*1e3
        plt.figure()
        plt.plot(range(2,self.numFrames+1),frameIntervals)
        plt.axis([1,self.numFrames,0,max(frameIntervals)*1.1])
        plt.xlabel('Frame Number')
        plt.ylabel('Frame Interval (ms)')
        plt.show()
        
    def getFrameTimes(self):
        for i in np.where(np.isnan(self.frameTimes))[0]:
            self.frameTimes[i] = self.dataFileIn[str(i+1)].attrs.get('acquisitionTime')
        self.frameTimes -= self.frameTimes[0]
        
    def findSaccades(self):
        # find peaks in pupil velocity
        vel,t = self.getPupilVelocity()
        thresh = self.saccadeThresh*np.nanstd(vel)
        self.negSaccades = np.where((vel<-thresh) & np.concatenate(([False],vel[1:]<vel[:-1])) & np.concatenate((vel[:-1]<vel[1:],[False])))[0]
        self.posSaccades = np.where((vel>thresh) & np.concatenate(([False],vel[1:]>vel[:-1])) & np.concatenate((vel[:-1]>vel[1:],[False])))[0]
        # remove peaks that are too close in time
        self.negSaccades = self.negSaccades[np.concatenate(([True],np.diff(t[self.negSaccades])>self.saccadeRefractoryPeriod))]
        self.posSaccades = self.posSaccades[np.concatenate(([True],np.diff(t[self.posSaccades])>self.saccadeRefractoryPeriod))]
        # remove negative peaks too closely following positive peaks and vice versa
        peakTimeDiff = t[self.negSaccades]-t[self.posSaccades][:,None]
        self.negSaccades = self.negSaccades[np.all(np.logical_or(peakTimeDiff<0,peakTimeDiff>self.saccadeRefractoryPeriod),axis=0)]
        self.posSaccades = self.posSaccades[np.all(np.logical_or(peakTimeDiff>0,peakTimeDiff<-self.saccadeRefractoryPeriod),axis=1)]
        self.selectedSaccade = None
        self.plotSaccades()
        
    def getPupilVelocity(self):
        if all(np.isnan(self.frameTimes)):
            t = np.arange(self.numFrames)/self.frameRate
        else:
            self.getFrameTimes()
            t = self.frameTimes
        n = self.saccadeSmoothPts//2
        vel = np.diff(self.pupilX)/np.diff(t)
        velSmoothed = np.convolve(vel,np.ones(self.saccadeSmoothPts)/self.saccadeSmoothPts,mode='same')
        velSmoothed[:n] = vel[:n].mean()
        velSmoothed[-n:] = vel[-n:].mean()
        return velSmoothed,t
    
    def plotSaccades(self):
        t = np.arange(self.numFrames)/self.frameRate
        self.negSaccadesPlot.setData(x=t[self.negSaccades],y=self.pupilX[self.negSaccades])
        self.posSaccadesPlot.setData(x=t[self.posSaccades],y=self.pupilX[self.posSaccades])
        
    def getSaccadesOnDisplay(self):
        saccades = np.sort(np.concatenate((self.negSaccades,self.posSaccades)))
        if saccades.size>0:
            if self.frameNum>self.numDataPlotPts:
                onDisplay = np.logical_and(saccades>self.frameNum-self.numDataPlotPts,saccades<self.frameNum)
            else:
                onDisplay = saccades<self.numDataPlotPts
            saccades = saccades[onDisplay]
        return saccades
    
    def deleteAllSaccades(self):
        self.negSaccadesPlot.setData(x=[],y=[])
        self.posSaccadesPlot.setData(x=[],y=[])
        self.negSaccades = np.array([],dtype=int)
        self.posSaccades = self.negSaccades.copy()
        self.selectedSaccade = None
    
    def setSaccadeSmooth(self):
        val,ok = QtGui.QInputDialog.getInt(self.mainWin,'Set Saccade Smoothing','number of points:',value=self.saccadeSmoothPts,min=1)
        if ok:
            self.saccadeSmoothPts = val
        
    def setSaccadeThresh(self):
        val,ok = QtGui.QInputDialog.getDouble(self.mainWin,'Set Saccade Threshold','standard devations from baseline:',value=self.saccadeThresh,min=0.1,decimals=1)
        if ok:
            self.saccadeThresh = val
            
    def setSaccadeRefractoryPeriod(self):
        val,ok = QtGui.QInputDialog.getDouble(self.mainWin,'Set Saccade Refractory Period','seconds:',value=self.saccadeRefractoryPeriod,min=0.001,decimals=3)
        if ok:
            self.saccadeRefractoryPeriod = val
        

if __name__=="__main__":
    start()