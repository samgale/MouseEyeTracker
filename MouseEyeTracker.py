# -*- coding: utf-8 -*-
"""
GUI for tracking mouse pupil area and position/rotation
Acquire data with camera or analyze data from hdf5 or video file

@author: samgale
"""

from __future__ import division
import sip
sip.setapi('QString', 2)
import h5py, json, math, os, time
import cv2
import numpy as np
import scipy.io
import scipy.signal
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from matplotlib import pyplot as plt


class QtSignalGenerator(QtCore.QObject):

    camFrameCapturedSignal = QtCore.pyqtSignal(np.ndarray,float)
    
    def __init__(self):
        QtCore.QObject.__init__(self)


# frame captured callback must be thread safe
# signal generator is used to send frame data to gui thread
qtSignalGeneratorObj = QtSignalGenerator()


def camFrameCaptured(frame):
    img = frame.buffer_data_numpy()
    qtSignalGeneratorObj.camFrameCapturedSignal.emit(img,frame.data.timestamp)
    frame.queue_for_capture(frame_callback=camFrameCaptured)


def start():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    eyeTrackerObj = EyeTracker(app)
    qtSignalGeneratorObj.camFrameCapturedSignal.connect(eyeTrackerObj.processCamFrame)
    app.exec_()


class EyeTracker():
    
    def __init__(self,app):
        self.app = app
        self.fileOpenSavePath = os.path.dirname(os.path.realpath(__file__))
        self.camSavePath = self.fileOpenSavePath
        self.camSaveBaseName = 'MouseEyeTracker'
        self.camSaveFileType = '.hdf5'
        self.camConfig = False
        self.skvideo = None
        self.ffmpeg = None
        self.nidaq = None
        self.vimba = None
        self.cam = None
        self.camFrames = []
        self.videoIn = None
        self.videoOut = None
        self.dataFileIn = None
        self.dataFileOut = None
        self.image = None
        self.displayUpdateInterval = 1
        self.roi = None
        self.blurSigma = 2.0
        self.imgExponent = 2.0
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
        self.configItems = ('camName',
                            'camType',
                            'camSavePath',
                            'camSaveBaseName',
                            'camSaveFileType',
                            'camBufferSize',
                            'camBinning',
                            'camExposure',
                            'frameRate',
                            'displayUpdateInterval',
                            'roiPos',
                            'roiSize',
                            'ffmpeg',
                            'nidaq',
                            'cameraMenuNidaqIn',
                            'cameraMenuNidaqOut')
        
        # main window
        self.mainWin = QtWidgets.QMainWindow()
        self.mainWin.setWindowTitle('MouseEyeTracker')
        self.mainWin.closeEvent = self.mainWinCloseEvent
        
        # file menu
        self.menuBar = self.mainWin.menuBar()
        self.menuBar.setNativeMenuBar(False)
        self.fileMenu = self.menuBar.addMenu('File')
        self.fileMenuOpen = self.fileMenu.addMenu('Open')
        self.fileMenuOpenFrames = QtWidgets.QAction('Frame Data',self.mainWin)
        self.fileMenuOpenFrames.triggered.connect(self.loadFrameData)
        self.fileMenuOpenData = QtWidgets.QAction('Tracking Data',self.mainWin,enabled=False)
        self.fileMenuOpenData.triggered.connect(self.loadTrackingData)
        self.fileMenuOpen.addActions([self.fileMenuOpenFrames,self.fileMenuOpenData])
        
        self.fileMenuSave = self.fileMenu.addMenu('Save')
        self.fileMenuSave.setEnabled(False)
        self.fileMenuSaveFrames = self.fileMenuSave.addMenu('Frame Data')
        self.fileMenuSaveFramesNpz = QtWidgets.QAction('npz',self.mainWin)
        self.fileMenuSaveFramesNpz.triggered.connect(self.saveFrameData)
        self.fileMenuSaveFramesMat = QtWidgets.QAction('mat',self.mainWin)
        self.fileMenuSaveFramesMat.triggered.connect(self.saveFrameData)
        self.fileMenuSaveFrames.addActions([self.fileMenuSaveFramesNpz,self.fileMenuSaveFramesMat])
        self.fileMenuSaveData = self.fileMenuSave.addMenu('Tracking Data')
        self.fileMenuSaveDataHdf5 = QtWidgets.QAction('hdf5',self.mainWin)
        self.fileMenuSaveDataHdf5.triggered.connect(self.saveTrackingData)
        self.fileMenuSaveDataNpz = QtWidgets.QAction('npz',self.mainWin)
        self.fileMenuSaveDataNpz.triggered.connect(self.saveTrackingData)
        self.fileMenuSaveDataMat = QtWidgets.QAction('mat',self.mainWin)
        self.fileMenuSaveDataMat.triggered.connect(self.saveTrackingData)
        self.fileMenuSaveData.addActions([self.fileMenuSaveDataHdf5,self.fileMenuSaveDataNpz,self.fileMenuSaveDataMat])
        self.fileMenuSaveImage = QtWidgets.QAction('Image',self.mainWin)
        self.fileMenuSaveImage.triggered.connect(self.saveImage)
        self.fileMenuSaveMovie = QtWidgets.QAction('Movie',self.mainWin)
        self.fileMenuSaveMovie.triggered.connect(self.saveMovie)
        self.fileMenuSaveAnnotatedMovie = QtWidgets.QAction('Annotated Movie',self.mainWin,enabled=False)
        self.fileMenuSaveAnnotatedMovie.triggered.connect(self.saveMovie)
        self.fileMenuSave.addActions([self.fileMenuSaveImage,self.fileMenuSaveMovie,self.fileMenuSaveAnnotatedMovie])
        
        # options menu
        self.optionsMenu = self.menuBar.addMenu('Options')
        self.optionsMenuShowTracking = QtWidgets.QAction('Show Pupil Tracking Plots',self.mainWin,checkable=True)
        self.optionsMenuShowTracking.triggered.connect(self.showPupilTrackingPlots)
        self.optionsMenuSetDisplayUpdate = QtWidgets.QAction('Set Display Update Interval',self.mainWin)
        self.optionsMenuSetDisplayUpdate.triggered.connect(self.setDisplayUpdateInterval)
        self.optionsMenu.addActions([self.optionsMenuShowTracking,self.optionsMenuSetDisplayUpdate])
        
        # camera menu
        self.cameraMenu = self.menuBar.addMenu('Camera')         
        self.cameraMenuUseCam = QtWidgets.QAction('Use Camera',self.mainWin,checkable=True)
        self.cameraMenuUseCam.triggered.connect(self.initCamera)
        self.cameraMenu.addAction(self.cameraMenuUseCam)
        
        self.cameraMenuLoadConfig = QtWidgets.QAction('Load Configuration',self.mainWin)
        self.cameraMenuLoadConfig.triggered.connect(self.loadCamConfig)
        self.cameraMenuClearConfig = QtWidgets.QAction('Clear Configuration',self.mainWin)
        self.cameraMenuClearConfig.triggered.connect(self.clearCamConfig)
        self.cameraMenuSaveConfig = QtWidgets.QAction('Save Configuration',self.mainWin)
        self.cameraMenuSaveConfig.triggered.connect(self.saveCamConfig)
        self.cameraMenuSaveConfig.setEnabled(False)
        self.cameraMenu.addActions([self.cameraMenuLoadConfig,self.cameraMenuClearConfig,self.cameraMenuSaveConfig])
        
        self.cameraMenuSettings = self.cameraMenu.addMenu('Settings')
        self.cameraMenuSettings.setEnabled(False)
        self.cameraMenuSettingsBufferSize = QtWidgets.QAction('Buffer Size',self.mainWin)
        self.cameraMenuSettingsBufferSize.triggered.connect(self.setCamBufferSize)
        self.cameraMenuSettingsBinning = QtWidgets.QAction('Spatial Binning',self.mainWin)
        self.cameraMenuSettingsBinning.triggered.connect(self.setCamBinning)
        self.cameraMenuSettingsExposure = QtWidgets.QAction('Exposure',self.mainWin)
        self.cameraMenuSettingsExposure.triggered.connect(self.setCamExposure)
        self.cameraMenuSettingsFrameRate = QtWidgets.QAction('Frame Rate',self.mainWin)
        self.cameraMenuSettingsFrameRate.triggered.connect(self.setCamFrameRate)
        self.cameraMenuSettingsItems = (self.cameraMenuSettingsBufferSize,self.cameraMenuSettingsBinning,self.cameraMenuSettingsExposure,self.cameraMenuSettingsFrameRate)
        self.cameraMenuSettings.addActions(self.cameraMenuSettingsItems)
        
        self.cameraMenuNidaq = self.cameraMenu.addMenu('NIDAQ IO')
        self.cameraMenuNidaq.setEnabled(False)
        self.cameraMenuNidaqIn = QtWidgets.QAction('Use Save Trigger (NIDAQ Input P0.0)',self.mainWin,checkable=True)
        self.cameraMenuNidaqIn.triggered.connect(self.setNidaqIO)
        self.cameraMenuNidaqOut = QtWidgets.QAction('Signal Saved Frames (NIDAQ Output P1.0)',self.mainWin,checkable=True)
        self.cameraMenuNidaqOut.triggered.connect(self.setNidaqIO)
        self.cameraMenuNidaq.addActions([self.cameraMenuNidaqIn,self.cameraMenuNidaqOut])
        
        self.cameraMenuSetSavePath = QtWidgets.QAction('Set Save Path',self.mainWin)
        self.cameraMenuSetSavePath.triggered.connect(self.setCamSavePath)
        self.cameraMenuSetSaveBaseName = QtWidgets.QAction('Set Save Basename',self.mainWin)
        self.cameraMenuSetSaveBaseName.triggered.connect(self.setCamSaveBaseName)
        self.cameraMenuSetSaveFileType = QtWidgets.QAction('Set Save File Type',self.mainWin)
        self.cameraMenuSetSaveFileType.triggered.connect(self.setCamSaveFileType)
        self.cameraMenu.addActions([self.cameraMenuSetSavePath,self.cameraMenuSetSaveBaseName,self.cameraMenuSetSaveFileType])
        
        # pupil tracking menu
        self.trackMenu = self.menuBar.addMenu('Pupil Tracking')
        self.trackMenu.setEnabled(False)
        self.trackMenuStopTracking = QtWidgets.QAction('Stop Tracking',self.mainWin,checkable=True)
        self.trackMenuStopTracking.triggered.connect(self.toggleStopTracking)
        self.trackMenuSetDataNan = QtWidgets.QAction('Set Data NaN',self.mainWin,checkable=True)
        self.trackMenuSetDataNan.triggered.connect(self.toggleSetDataNan)
        self.trackMenu.addActions([self.trackMenuStopTracking,self.trackMenuSetDataNan])
        
        self.trackMenuMmPerPix = self.trackMenu.addMenu('mm/pixel')
        self.trackMenuMmPerPixSet = QtWidgets.QAction('Set',self.mainWin)
        self.trackMenuMmPerPixSet.triggered.connect(self.setMmPerPix)
        self.trackMenuMmPerPixMeasure = QtWidgets.QAction('Measure',self.mainWin,enabled=False)
        self.trackMenuMmPerPixMeasure.triggered.connect(self.measureMmPerPix)
        self.trackMenuMmPerPix.addActions([self.trackMenuMmPerPixSet,self.trackMenuMmPerPixMeasure])
        
        self.trackMenuBlurImage = QtWidgets.QAction('Guassian Blur Image',self.mainWin,checkable=True)
        self.trackMenuBlurImage.triggered.connect(self.setBlurImage)
        self.trackMenuSetBlurSigma = QtWidgets.QAction('Set Blur Sigma',self.mainWin)
        self.trackMenuSetBlurSigma.triggered.connect(self.setBlurSigma)
        self.trackMenuExpImage = QtWidgets.QAction('Exponentiate Image',self.mainWin,checkable=True)
        self.trackMenuExpImage.triggered.connect(self.setExponentiateImage)
        self.trackMenuSetExp = QtWidgets.QAction('Set Exponent',self.mainWin)
        self.trackMenuSetExp.triggered.connect(self.setExponent)
        self.trackMenu.addActions([self.trackMenuBlurImage,self.trackMenuSetBlurSigma,self.trackMenuExpImage,self.trackMenuSetExp])
        
        self.trackMenuReflectType = self.trackMenu.addMenu('Reflection Type')
        self.trackMenuReflectTypeSpot = QtWidgets.QAction('Spot',self.mainWin,checkable=True)
        self.trackMenuReflectTypeSpot.setChecked(True)
        self.trackMenuReflectTypeSpot.triggered.connect(self.setReflectType)
        self.trackMenuReflectTypeRing = QtWidgets.QAction('Ring',self.mainWin,checkable=True)
        self.trackMenuReflectTypeRing.triggered.connect(self.setReflectType)
        self.trackMenuReflectType.addActions([self.trackMenuReflectTypeSpot,self.trackMenuReflectTypeRing])
        self.trackMenuReflectThresh = QtWidgets.QAction('Reflection Threshold',self.mainWin)
        self.trackMenuReflectThresh.triggered.connect(self.setReflectThresh)
        self.trackMenu.addAction(self.trackMenuReflectThresh)
        
        self.trackMenuPupilSign = self.trackMenu.addMenu('Pupil Sign')
        self.trackMenuPupilSignNeg = QtWidgets.QAction('Negative',self.mainWin,checkable=True)
        self.trackMenuPupilSignNeg.setChecked(True)
        self.trackMenuPupilSignNeg.triggered.connect(self.setPupilSign)
        self.trackMenuPupilSignPos = QtWidgets.QAction('Positive',self.mainWin,checkable=True)
        self.trackMenuPupilSignPos.triggered.connect(self.setPupilSign)
        self.trackMenuPupilSign.addActions([self.trackMenuPupilSignNeg,self.trackMenuPupilSignPos])
        
        self.trackMenuPupilMethod = self.trackMenu.addMenu('Pupil Track Method')
        self.trackMenuPupilMethodStarburst = QtWidgets.QAction('Starburst',self.mainWin,checkable=True)
        self.trackMenuPupilMethodStarburst.setChecked(True)
        self.trackMenuPupilMethodStarburst.triggered.connect(self.setPupilTrackMethod)
        self.trackMenuPupilMethodLine = QtWidgets.QAction('Line',self.mainWin,checkable=True)
        self.trackMenuPupilMethodLine.triggered.connect(self.setPupilTrackMethod)
        self.trackMenuPupilMethodGradients = QtWidgets.QAction('Gradients',self.mainWin,checkable=True)
        self.trackMenuPupilMethodGradients.triggered.connect(self.setPupilTrackMethod)
        self.trackMenuPupilMethodIntensity = QtWidgets.QAction('Intensity',self.mainWin,checkable=True)
        self.trackMenuPupilMethodIntensity.triggered.connect(self.setPupilTrackMethod)
        self.trackMenuPupilMethod.addActions([self.trackMenuPupilMethodStarburst,self.trackMenuPupilMethodLine,self.trackMenuPupilMethodGradients,self.trackMenuPupilMethodIntensity])
        
        self.trackMenuAdaptThresh = QtWidgets.QAction('Adaptive Threshold',self.mainWin,checkable=True)
        self.trackMenuAdaptThresh.triggered.connect(self.setAdaptiveThreshold)
        self.trackMenuCircularity = QtWidgets.QAction('Circularity',self.mainWin)
        self.trackMenuCircularity.triggered.connect(self.setCircularityThresh)
        self.trackMenu.addActions([self.trackMenuAdaptThresh,self.trackMenuCircularity])        
        
        self.trackMenuLineOrigin = self.trackMenu.addMenu('Line Origin')
        self.trackMenuLineOrigin.setEnabled(False)
        self.trackMenuLineOriginLeft = QtWidgets.QAction('Left',self.mainWin,checkable=True)
        self.trackMenuLineOriginLeft.setChecked(True)
        self.trackMenuLineOriginLeft.triggered.connect(self.setPupilEdgeLineOrigin)
        self.trackMenuLineOriginRight = QtWidgets.QAction('Right',self.mainWin,checkable=True)
        self.trackMenuLineOriginRight.triggered.connect(self.setPupilEdgeLineOrigin)
        self.trackMenuLineOrigin.addActions([self.trackMenuLineOriginLeft,self.trackMenuLineOriginRight])        
        
        self.trackMenuGradientDownsamp = QtWidgets.QAction('Gradient Downsample',self.mainWin,enabled=False)
        self.trackMenuGradientDownsamp.triggered.connect(self.setPupilGradientDownsample)
        self.trackMenu.addAction(self.trackMenuGradientDownsamp)
        
        # analysis menu
        self.analysisMenu = self.menuBar.addMenu('Analysis')
        self.analysisMenu.setEnabled(False)
        self.analysisMenuConvert = self.analysisMenu.addMenu('Convert')
        self.analysisMenuConvertPixToDeg = QtWidgets.QAction('Pixels to Degrees',self.mainWin)
        self.analysisMenuConvertPixToDeg.triggered.connect(self.pixToDeg)
        self.analysisMenuConvertDegToPix = QtWidgets.QAction('Degrees to Pixels',self.mainWin)
        self.analysisMenuConvertDegToPix.triggered.connect(self.degToPix)
        self.analysisMenuConvert.addActions([self.analysisMenuConvertPixToDeg,self.analysisMenuConvertDegToPix])
        
        self.analysisMenuAnalyzeAll = QtWidgets.QAction('Analyze All Frames',self.mainWin)
        self.analysisMenuAnalyzeAll.triggered.connect(self.analyzeAllFrames)
        self.analysisMenuFrameIntervals = QtWidgets.QAction('Plot Frame Intervals',self.mainWin)
        self.analysisMenuFrameIntervals.triggered.connect(self.plotFrameIntervals)
        self.analysisMenu.addActions([self.analysisMenuAnalyzeAll,self.analysisMenuFrameIntervals])
        
        self.analysisMenuSaccades = self.analysisMenu.addMenu('Saccades')
        self.analysisMenuSaccadesFind = QtWidgets.QAction('Find',self.mainWin)
        self.analysisMenuSaccadesFind.triggered.connect(self.findSaccades)
        self.analysisMenuSaccadesDelete = QtWidgets.QAction('Delete All',self.mainWin)
        self.analysisMenuSaccadesDelete.triggered.connect(self.deleteAllSaccades)
        self.analysisMenuSaccadesSmooth = QtWidgets.QAction('Smoothing',self.mainWin)
        self.analysisMenuSaccadesSmooth.triggered.connect(self.setSaccadeSmooth)
        self.analysisMenuSaccadesThresh = QtWidgets.QAction('Threshold',self.mainWin)
        self.analysisMenuSaccadesThresh.triggered.connect(self.setSaccadeThresh)
        self.analysisMenuSaccadesRefractory = QtWidgets.QAction('Refractory Period',self.mainWin)
        self.analysisMenuSaccadesRefractory.triggered.connect(self.setSaccadeRefractoryPeriod)
        self.analysisMenuSaccades.addActions([self.analysisMenuSaccadesFind,self.analysisMenuSaccadesDelete,self.analysisMenuSaccadesThresh,self.analysisMenuSaccadesSmooth,self.analysisMenuSaccadesRefractory])
        
        # layout
        self.createVideoLayout()
        self.mainWin.show()
        
    def createLayoutItems(self):
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
        self.startVideoButton = QtWidgets.QPushButton('Start Video',checkable=True)
        self.startVideoButton.clicked.connect(self.startVideo)
        self.roiButton = QtWidgets.QPushButton('Set ROI',checkable=True)
        self.roiButton.clicked.connect(self.setROI)
        self.buttons = (self.startVideoButton,self.roiButton)
        
        self.saveCheckBox = QtWidgets.QCheckBox('Save Video Data',enabled=False)
        
        # frame navigation
        self.frameNumSpinBox = QtWidgets.QSpinBox()
        self.frameNumSpinBox.setPrefix('Frame: ')
        self.frameNumSpinBox.setSuffix(' of 0')
        self.frameNumSpinBox.setRange(0,1)
        self.frameNumSpinBox.setSingleStep(1)
        self.frameNumSpinBox.setValue(0)
        self.frameNumSpinBox.setEnabled(False)
        self.frameNumSpinBox.valueChanged.connect(self.goToFrame)
        self.frameNumSpinBox.blockSignals(True)
        
    def createVideoLayout(self):
        self.mainWidget = QtWidgets.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayoutSize(500,600,2,4)
        self.createLayoutItems()
        self.mainWidget.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.startVideoButton,0,0,1,1)
        self.mainLayout.addWidget(self.roiButton,0,1,1,1)
        self.mainLayout.addWidget(self.imageLayout,1,0,2,2)
        self.mainLayout.addWidget(self.saveCheckBox,3,0,1,1)
        self.mainLayout.addWidget(self.frameNumSpinBox,3,1,1,1)
    
    def createPupilTrackingLayout(self):
        self.mainWidget = QtWidgets.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayoutSize(1000,500,20,4)
        self.mainWidget.setLayout(self.mainLayout)
        self.createLayoutItems()
        
        # buttons
        self.findPupilButton = QtWidgets.QPushButton('Find Pupil',checkable=True)
        self.findPupilButton.clicked.connect(self.findPupil)
        self.findReflectButton = QtWidgets.QPushButton('Find Reflection',checkable=True)
        self.findReflectButton.clicked.connect(self.findReflect)
        self.setMaskButton = QtWidgets.QPushButton('Set Masks',checkable=True)
        self.setMaskButton.clicked.connect(self.setMask)
        self.buttons += (self.findPupilButton,self.findReflectButton,self.setMaskButton)
        
        self.useMaskCheckBox = QtWidgets.QCheckBox('Use Masks')
        self.useMaskCheckBox.clicked.connect(self.setUseMask)
        
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
        self.negSaccadesPlot = self.pupilXPlotItem.plot(x=[],y=[],pen=None,symbol=downTriangle,symbolSize=10,symbolPen='g',symbolBrush='g')
        self.posSaccadesPlot = self.pupilXPlotItem.plot(x=[],y=[],pen=None,symbol=upTriangle,symbolSize=10,symbolPen='b',symbolBrush='b')
        
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
        
        # frame navigation
        self.pupilAreaFrameNumLine = pg.InfiniteLine(pos=0,pen='r',movable=True,bounds=(0,1))
        self.pupilXFrameNumLine = pg.InfiniteLine(pos=0,pen='r',movable=True,bounds=(0,1))
        self.pupilYFrameNumLine = pg.InfiniteLine(pos=0,pen='r',movable=True,bounds=(0,1))
        self.frameNumLines = (self.pupilAreaFrameNumLine,self.pupilXFrameNumLine,self.pupilYFrameNumLine)
        for line in self.frameNumLines:
            line.sigDragged.connect(self.frameNumLineDragged)
            line.sigPositionChangeFinished.connect(self.frameNumLinePosChangeFin)
            
        # data plot duration control
        self.plotDurLayout = QtWidgets.QFormLayout()
        self.plotDurEdit = QtWidgets.QLineEdit(str(self.defaultDataPlotDur))
        self.plotDurEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.plotDurEdit.editingFinished.connect(self.changePlotWindowDur)
        self.plotDurLayout.addRow('Plot Duration',self.plotDurEdit)
        
        # layout
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
        self.mainWin.keyPressEvent = self.mainWinKeyPressEvent
        
    def setLayoutSize(self,winWidth,winHeight,nCols,nRows):
        self.app.processEvents()
        self.mainWin.resize(winWidth,winHeight)
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(QtWidgets.QDesktopWidget().availableGeometry().center())
        self.mainWin.move(mainWinRect.topLeft())
        for col in range(nCols):
            self.mainLayout.setColumnMinimumWidth(col,winWidth/nCols)
            self.mainLayout.setColumnStretch(col,1)
        rowHeights = np.zeros(nRows)
        rowHeights[[0,-1]] = 0.05*winHeight
        rowHeights[1:-1] = 0.9*winHeight/(nRows-2)
        for row in range(nRows):
            self.mainLayout.setRowMinimumHeight(row,rowHeights[row])
            self.mainLayout.setRowStretch(row,1)
            
    def showPupilTrackingPlots(self):
        self.turnOffButtons()
        if self.optionsMenuShowTracking.isChecked():
            self.createPupilTrackingLayout()
        else:
            self.createVideoLayout()
        if self.image is not None:
            self.initDisplay()
        
    def mainWinCloseEvent(self,event):
        if self.cam is not None:
            self.closeCamera()
        elif self.videoIn is not None:
            self.closeVideo()
        elif self.dataFileIn is not None:
            self.closeDataFileIn()
        event.accept()
        
    def saveFrameData(self):
        if self.mainWin.sender()==self.fileMenuSaveDataNpz:
            fileType = '.npz'
        else:
            fileType = '.mat'
        filePath,fileType = QtWidgets.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileOpenSavePath,'*'+fileType)
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        startFrame,endFrame = self.getFrameSaveRange()
        if startFrame is None:
            return
        frameData = np.zeros((endFrame-startFrame+1,self.roiSize[1],self.roiSize[0]),dtype=self.image.dtype)
        if self.dataFileIn is None:
            self.videoIn.set(cv2.CAP_PROP_POS_FRAMES,startFrame-1)
        for i,frame in enumerate(range(startFrame,endFrame+1)):
            if self.dataFileIn is None:
                isImage,image = self.videoIn.read()
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            else:
                image = self.dataFileIn['frame'][frame-1]
            frameData[i] = image[self.roiInd]
        if self.dataFileIn is None:
            self.videoIn.set(cv2.CAP_PROP_POS_FRAMES,self.frameNum-1)
        data = {'frameData': frameData}
        if fileType=='.npz':
            np.savez_compressed(filePath,**data)
        else:
            scipy.io.savemat(filePath,data,do_compression=True)
        
    def saveTrackingData(self):
        if self.mainWin.sender()==self.fileMenuSaveDataHdf5:
            fileType = '.hdf5'
        elif self.mainWin.sender()==self.fileMenuSaveDataNpz:
            fileType = '.npz'
        else:
            fileType = '.mat'
        filePath,fileType = QtWidgets.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileOpenSavePath,'*'+fileType)
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        self.reflectCenter += self.roiPos
        self.pupilCenter += self.roiPos
        params = ('mmPerPixel','frameTimes','reflectCenter','pupilCenter','pupilArea','pupilX','pupilY','negSaccades','posSaccades')
        if fileType=='.hdf5':
            dataFile = h5py.File(filePath,'w',libver='latest')
            dataFile.attrs.create('mmPerPixel',self.mmPerPixel)
            for param in params[1:]:
                dataFile.create_dataset(param,data=getattr(self,param),compression='gzip',compression_opts=1)
            dataFile.close()
        else:
            data = {param: getattr(self,param) for param in params}
            if fileType=='.npz':
                np.savez_compressed(filePath,**data)
            else:
                scipy.io.savemat(filePath,data,do_compression=True)
                
    def saveImage(self):
        if self.image is None:
            return
        filePath,fileType = QtWidgets.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileOpenSavePath,'*.png')
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        cv2.imwrite(filePath,self.image)
        
    def saveMovie(self):
        filePath,fileType = QtWidgets.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileOpenSavePath,'*.mp4')
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        if self.skvideo is None:
            self.initSkvideo()
            if self.skvideo is None:
                return
        startFrame,endFrame = self.getFrameSaveRange()
        if startFrame is None:
            return
        vidOut = self.skvideo.io.FFmpegWriter(filePath,inputdict={'-r':str(self.frameRate)},outputdict={'-r':str(self.frameRate),'-vcodec':'libx264','-crf':'17'})
        if self.dataFileIn is None:
            self.videoIn.set(cv2.CAP_PROP_POS_FRAMES,startFrame-1)
        for frame in range(startFrame,endFrame+1):
            if self.dataFileIn is None:
                isImage,image = self.videoIn.read()
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            else:
                image = self.dataFileIn['frames'][frame-1]
            vidOut.writeFrame(image[self.roiInd])
        vidOut.close()
        if self.dataFileIn is None:
            self.videoIn.set(cv2.CAP_PROP_POS_FRAMES,self.frameNum-1)
            
    def getFrameSaveRange(self):
        startFrame,endFrame = (self.frameNum-self.numDataPlotPts,self.frameNum) if self.frameNum>self.numDataPlotPts else (1,self.numDataPlotPts)
        text,ok = QtWidgets.QInputDialog.getText(self.mainWin,'Save Movie','Enter frames range:',text=str(startFrame)+'-'+str(endFrame))
        if not ok:
            return None,None
        startFrame,endFrame = [int(n) for n in text.split('-')]
        if startFrame<1:
            startFrame = 1
        if endFrame>self.numFrames:
            endFrame = self.numFrames
        return startFrame,endFrame
               
    def loadFrameData(self):
        filePath,fileType = QtWidgets.QFileDialog.getOpenFileName(self.mainWin,'Choose File',self.fileOpenSavePath,'')
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        if self.cam is not None:
            self.cameraMenuUseCam.setChecked(False)
            self.closeCamera()
        elif self.dataFileIn is not None:
            self.closeDataFileIn()
        elif self.videoIn is not None:
            self.closeVideo()
        self.mainWin.setWindowTitle('MouseEyeTracker'+'     '+filePath)
        fileName,fileExt = os.path.splitext(os.path.basename(filePath))
        if fileExt=='.hdf5':
            self.dataFileIn = h5py.File(filePath,'r')
            self.frameRate = self.dataFileIn.attrs.get('frameRate')
            self.mmPerPixel = self.dataFileIn.attrs.get('mmPerPixel')
            self.frameTimes = self.dataFileIn['frameTimes'][:]
            self.numFrames = self.dataFileIn['frames'].shape[0]
            if np.isnan(self.frameRate):
                self.frameRate = self.numFrames/self.frameTimes[-1]
        else:
            self.videoIn = cv2.VideoCapture(filePath)
            self.frameRate = self.videoIn.get(cv2.CAP_PROP_FPS)
            self.numFrames = int(round(self.videoIn.get(cv2.CAP_PROP_FRAME_COUNT)))
            dataFilePath = os.path.join(os.path.dirname(filePath),fileName+'.hdf5')
            if os.path.isfile(dataFilePath):
                dataFile = h5py.File(dataFilePath,'r')
                self.mmPerPixel = dataFile.attrs.get('mmPerPixel')
                self.frameTimes = dataFile['frameTimes'][:]
            else:
                self.mmPerPixel = np.nan
                self.frameTimes = np.nan
        if not np.all(np.isnan(self.frameTimes)):
            self.frameTimes -= self.frameTimes[0]
        self.fileMenuSave.setEnabled(True)
        self.frameNum = 1
        self.getVideoImage()
        self.resetROI()
        self.initDisplay()
        
    def loadTrackingData(self):
        filePath,fileType = QtWidgets.QFileDialog.getOpenFileName(self.mainWin,'Choose File',self.fileOpenSavePath,'Files (*.hdf5 *.npz *.mat)')
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        params = ('mmPerPixel','frameTimes','reflectCenter','pupilCenter','pupilArea','pupilX','pupilY','negSaccades','posSaccades')
        if fileType=='.hdf5':
            dataFile = h5py.File(filePath,'r')
            if 'mmPerPixel' in dataFile.attrs.keys():
                self.mmPerPixel = dataFile.attrs.get('mmPerPixel')
            for param in set(dataFile.keys()) & set(params[1:]):
                setattr(self,param,dataFile[param][:])
            dataFile.close()
        else:
            data = np.load(filePath) if fileType=='.npz' else scipy.io.loadmat(filePath,squeeze_me=True)
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
        
    def closeDataFileIn(self):
        self.closeFileCleanup()
        self.dataIsLoaded = False
        self.dataFileIn.close()
        self.dataFileIn = None
        
    def closeVideo(self):
        self.closeFileCleanup()
        self.videoIn.release()
        self.videoIn = None        
        
    def closeFileCleanup(self):
        self.turnOffButtons()
        self.frameNumSpinBox.setEnabled(False)
        self.frameNumSpinBox.blockSignals(True)
        self.frameNumSpinBox.setRange(0,1)
        self.frameNumSpinBox.setValue(0)
        self.frameNumSpinBox.setSuffix(' of 0')
        self.fileMenuOpenData.setEnabled(False)
        self.fileMenuSave.setEnabled(False)
        self.analysisMenu.setEnabled(False)
        self.image = None
        self.frameTimes = []
        if self.optionsMenuShowTracking.isChecked():
            self.removeFrameNumLines()
            for line in self.frameNumLines:
                line.setValue(0)
            self.resetPupilTracking()
            self.deleteAllSaccades()
        self.mainWin.setWindowTitle('MouseEyeTracker')
       
    def initCamera(self):
        if self.cameraMenuUseCam.isChecked():
            if self.dataFileIn is not None:
                self.closeDataFileIn()
            elif self.videoIn is not None:
                self.closeVideo()
            if self.camConfig:
                if self.camType=='vimba':
                    self.initVimba()
                if self.camSaveFileType=='.mp4' and self.skvideo is None:
                    self.initSkvideo()
                    if self.skvideo is None:
                        self.camSaveFileType = '.hdf5'
            else:
                self.getCamera()
            if self.camType=='vimba':
                self.cam = self.vimba.camera(self.camName)
                self.cam.open()
            elif self.camType=='webcam':
                self.cam = cv2.VideoCapture(int(self.camName[6:]))
            else:
                self.cameraMenuUseCam.setChecked(False)
                return
            self.initNidaq()
            self.mainWin.setWindowTitle('MouseEyeTracker'+'     '+'camera: '+self.camName+'     '+'nidaq: '+str(self.nidaq))
            self.setCamProps()
            self.frameNum = 0
            self.getCamImage()
            self.initDisplay()
            self.cameraMenuSettings.setEnabled(True)
            for item in self.cameraMenuSettingsItems:
                if self.camType=='vimba' or item in (self.cameraMenuSettingsBinning,self.cameraMenuSettingsExposure):
                    item.setEnabled(True)
                else:
                    item.setEnabled(False)
            self.cameraMenuLoadConfig.setEnabled(False)
            self.cameraMenuClearConfig.setEnabled(False)
            self.cameraMenuSaveConfig.setEnabled(True)
            self.trackMenuMmPerPixMeasure.setEnabled(True)
            if not self.cameraMenuNidaqIn.isChecked():
                self.saveCheckBox.setEnabled(True)
        else:
            self.closeCamera()
            
    def getCamera(self):
        self.initVimba()
        vimbaCams = [] if self.vimba is None else self.vimba.camera_ids()
        webcams = []
        i = 0
        while True:
            cam = cv2.VideoCapture(i)
            isImage,image = cam.read()
            if isImage:
                webcams.append('webcam'+str(i))
                i += 1
            else:
                break
        selectedCam,ok = QtWidgets.QInputDialog.getItem(self.mainWin,'Choose Camera','Camera IDs:',vimbaCams+webcams,editable=False)
        if ok:
            self.camName = selectedCam
            self.camType = 'vimba' if selectedCam in vimbaCams else 'webcam'
        else:
            self.camName = self.camType = None
            
    def initVimba(self):
        try:
            if self.vimba is None:
                import pymba
                self.vimba = pymba.Vimba()
            self.vimba.startup()
            self.vimba.system().run_feature_command("GeVDiscoveryAllOnce")
            time.sleep(0.2)
        except:
            if self.vimba is not None:
                self.vimba.shutdown()
                self.vimba = None
            print('Unable to initialize vimba')
    
    def initNidaq(self):
        try:
            import nidaqmx
            if not self.camConfig:
                deviceNames = nidaqmx.system._collections.device_collection.DeviceCollection().device_names
                selectedDevice,ok = QtWidgets.QInputDialog.getItem(self.mainWin,'Choose Nidaq Device','Nidaq Devices:',deviceNames,editable=False)
                if ok:
                    self.nidaq = selectedDevice
                    self.cameraMenuNidaqOut.setChecked(True)
                else:
                    return
            self.nidaqDigitalIn = nidaqmx.Task()
            self.nidaqDigitalIn.di_channels.add_di_chan(self.nidaq+'/port0/line0',line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self.nidaqDigitalOut = nidaqmx.Task()
            self.nidaqDigitalOut.do_channels.add_do_chan(self.nidaq+'/port1/line0',line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self.cameraMenuNidaq.setEnabled(True)
        except:
            self.nidaq = None
            print('Unable to initialize nidaq')
    
    def initSkvideo(self):
        try:
            if self.ffmpeg is None:
                ffmpegPath = QtWidgets.QFileDialog.getExistingDirectory(self.mainWin,'Select directory containing ffmpeg.exe','')
                if ffmpegPath!='':
                    self.ffmpeg = ffmpegPath
                else:
                    return
            import skvideo
            skvideo.setFFmpegPath(self.ffmpeg) # run this before importing skvideo.io
            import skvideo.io
            self.skvideo = skvideo
        except:
            self.ffmpeg = None
            print('Unable to initialize skvideo')
            
    def closeCamera(self):
        self.turnOffButtons()
        if self.camType=='vimba':
            self.cam.close()
            self.vimba.shutdown()
        else:
            self.cam.release()
        self.cam = None
        self.image = None
        if self.nidaq is not None:
            self.nidaqDigitalIn.close()
            self.nidaqDigitalOut.close()
            self.nidaq = None
        self.cameraMenuSettings.setEnabled(False)
        self.cameraMenuLoadConfig.setEnabled(True)
        self.cameraMenuClearConfig.setEnabled(True)
        self.cameraMenuSaveConfig.setEnabled(False)
        self.trackMenuMmPerPixMeasure.setEnabled(False)
        self.saveCheckBox.setEnabled(False)
        if self.optionsMenuShowTracking.isChecked():
            self.resetPupilTracking()
        self.mainWin.setWindowTitle('MouseEyeTracker')
        
    def startCamera(self,bufferSize=1):
        self.cameraMenuSettings.setEnabled(False)
        if self.nidaq is not None:
            self.nidaqDigitalIn.start()
            self.nidaqDigitalOut.start()
            self.nidaqDigitalOut.write(False)
        if self.camType=='vimba':
            for _ in range(bufferSize):
                frame = self.cam.new_frame()
                frame.announce()
                self.camFrames.append(frame)
            self.cam.start_capture()
        
    def stopCamera(self):
        if self.camType=='vimba':
            self.cam.AcquisitionStop()
            self.cam.end_capture()
            self.cam.flush_capture_queue()
            self.cam.revoke_all_frames()
            self.camFrames = []
        if self.dataFileOut is not None:
            self.closeDataFileOut()
        if self.nidaq is not None:
            self.nidaqDigitalIn.stop()
            self.nidaqDigitalOut.stop()
        self.cameraMenuSettings.setEnabled(True)
        
    def getCamImage(self):
        self.startCamera()
        if self.camType=='vimba':
            frame = self.camFrames[0]
            frame.queue_for_capture()
            self.cam.AcquisitionStart()
            frame.wait_for_capture()
            self.image = frame.buffer_data_numpy()
        else:
            isImage,image = self.cam.read()
            self.image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.stopCamera()
            
    def setCamProps(self):
        if self.camType=='vimba':
            if not self.camConfig:
                self.camBinning = 1
            self.cam.feature('BinningHorizontal').value = self.camBinning
            self.cam.feature('BinningVertical').value = self.camBinning
            self.fullRoiSize = (self.cam.feature('WidthMax').value,self.cam.feature('HeightMax').value)
            if not self.camConfig:
                self.camBufferSize = 60
                self.camExposure = 0.9
                self.frameRate = 60.0
                self.roiPos = (0,0)
                self.roiSize = self.fullRoiSize
            self.roiInd = np.s_[0:self.roiSize[1],0:self.roiSize[0]]
            self.cam.feature('PixelFormat').value ='Mono8'
            self.cam.feature('OffsetX').value = self.roiPos[0]
            self.cam.feature('OffsetY').value = self.roiPos[1]
            self.cam.feature('Width').value = self.roiSize[0]
            self.cam.feature('Height').value = self.roiSize[1]
            self.cam.feature('ExposureAuto').value = 'Off'
            self.cam.feature('ExposureTimeAbs').value = self.camExposure*1e6/self.frameRate
            self.cam.feature('AcquisitionFrameRateAbs').value = self.frameRate
            self.cam.feature('AcquisitionMode').value = 'Continuous'
            self.cam.feature('TriggerMode').value = 'Off'
            self.cam.feature('TriggerSource').value = 'FixedRate'
            self.cam.feature('SyncOutSelector').value = 'SyncOut2'
            self.cam.feature('SyncOutSource').value = 'Exposing'
            self.cam.feature('SyncOutPolarity').value = 'Normal'
        else:
            self.camBufferSize = None
            self.frameRate = np.nan
            self.webcamDefaultFrameShape = (int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)))
            if not self.camConfig:
                self.camBinning = 1
                self.camExposure = 1
                self.roiPos = (0,0)
                self.roiSize = self.webcamDefaultFrameShape[::-1]
            self.roiInd = np.s_[self.roiPos[1]:self.roiPos[1]+self.roiSize[1],self.roiPos[0]:self.roiPos[0]+self.roiSize[0]]
            if self.camBinning>1:
                h,w = [int(n/self.camBinning) for n in self.webcamDefaultFrameShape]
                self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
                self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,w)
                self.fullRoiSize = (w,h)
            else:
                self.fullRoiSize = self.webcamDefaultFrameShape[::-1]
            self.cam.set(cv2.CAP_PROP_EXPOSURE,math.log(self.camExposure/1000,2))
        
    def setCamBufferSize(self):
        val,ok = QtWidgets.QInputDialog.getInt(self.mainWin,'Set Camera Buffer Size','Frames:',value=self.camBufferSize,min=1)
        if not ok:
            return
        self.camBufferSize = val
            
    def setCamBinning(self):
        val,ok = QtWidgets.QInputDialog.getInt(self.mainWin,'Set Camera Spatial Binning','Pixels:',value=self.camBinning,min=1,max=8)
        if not ok:
            return
        scaleFactor = self.camBinning/val
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
        self.camBinning = val
        if self.camType=='vimba':
            self.cam.feature('BinningHorizontal').value = val
            self.cam.feature('BinningVertical').value = val
        else:
            h,w = [int(n/val) for n in self.webcamDefaultFrameShape]
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,w)
        self.resetROI()
        self.resetImage()
    
    def setCamExposure(self):
        if self.camType=='vimba':
            units = 'Fraction of frame interval:'
            minVal = 0.001
            maxVal = 0.99
        else:
            units = 'Exposure time (ms):'
            minVal = 0.001
            maxVal = 10000
        val,ok = QtWidgets.QInputDialog.getDouble(self.mainWin,'Set Camera Exposure',units,value=self.camExposure,min=minVal,max=maxVal,decimals=3)
        if not ok:
            return
        self.camExposure = val
        if self.camType=='vimba':
            self.cam.feature('ExposureTimeAbs').value = self.camExposure*1e6/self.frameRate
        else:
            self.cam.set(cv2.CAP_PROP_EXPOSURE,math.log(val/1000,2))
    
    def setCamFrameRate(self):
        val,ok = QtWidgets.QInputDialog.getDouble(self.mainWin,'Set Camera Frame Rate','Frames/s:',value=self.frameRate,min=0.01,max=119.30,decimals=2)
        if not ok:
            return
        self.frameRate = val
        self.cam.feature('AcquisitionFrameRateAbs').value = self.frameRate
        self.cam.feature('ExposureTimeAbs').value = self.camExposure*1e6/self.frameRate
        self.changePlotWindowDur()
        
    def setNidaqIO(self):
        if self.mainWin.sender() is self.cameraMenuNidaqIn:
            if self.cameraMenuNidaqIn.isChecked():
                self.saveCheckBox.setEnabled(False)
            else:
                if self.cam is not None:
                    self.saveCheckBox.setEnabled(True)
    
    def setCamSavePath(self):
        dirPath = QtWidgets.QFileDialog.getExistingDirectory(self.mainWin,'Choose Directory',self.camSavePath)
        if dirPath!='':
            self.camSavePath = dirPath
        
    def setCamSaveBaseName(self):
        val,ok = QtWidgets.QInputDialog.getText(self.mainWin,'Set File Base Name','',text=self.camSaveBaseName)
        if ok:
            self.camSaveBaseName = val
            
    def setCamSaveFileType(self):
        fileType,ok = QtWidgets.QInputDialog.getItem(self.mainWin,'Choose Save File Type','File Type:',('.hdf5','.mp4'),editable=False)
        if not ok:
            return
        if fileType=='.mp4' and self.skvideo is None:
            self.initSkvideo()
            if self.skvideo is not None:
                self.camSaveFileType = fileType
        else:
            self.camSaveFileType = fileType
    
    def loadCamConfig(self):
        filePath,fileType = QtWidgets.QFileDialog.getOpenFileName(self.mainWin,'Choose File',self.fileOpenSavePath,'*.json')
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        with open(filePath,'r') as file:
            config = json.load(file)
        for item in self.configItems:
            attr = getattr(self,item,None)
            if isinstance(attr,QtWidgets.QAction):
                attr.setChecked(config[item])
            else:
                setattr(self,item,config[item])
        self.camConfig = True
        self.cameraMenuUseCam.setChecked(True)
        self.initCamera()
    
    def clearCamConfig(self):
        self.camConfig = False
            
    def saveCamConfig(self):
        filePath,fileType = QtWidgets.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileOpenSavePath,'*.json')
        if filePath=='':
            return
        self.fileOpenSavePath = os.path.dirname(filePath)
        config = {}
        for item in self.configItems:
            attr = getattr(self,item)
            if isinstance(attr,QtWidgets.QAction):
                config[item] = attr.isChecked()
            else:
                config[item] = attr
        with open(filePath,'w') as file:
            json.dump(config,file)
                    
    def getVideoImage(self):
        if self.dataFileIn is not None:
            self.image = self.dataFileIn['frames'][self.frameNum-1]
        else:
            isImage,image = self.videoIn.read()
            self.image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if self.trackMenuBlurImage.isChecked():
            self.blurImage()
        if self.trackMenuExpImage.isChecked():
            self.exponentiateImage()
                
    def blurImage(self):
        self.image = cv2.GaussianBlur(self.image,(0,0),self.blurSigma)
                
    def setBlurImage(self):
        self.getVideoImage()
        self.updateDisplay()
        
    def setBlurSigma(self):
        val,ok = QtWidgets.QInputDialog.getDouble(self.mainWin,'Set Blur Sigma','',value=self.blurSigma,min=0.01,decimals=2)
        if not ok:
            return
        self.blurSigma = val
        if self.trackMenuBlurImage.isChecked():
            self.getVideoImage()
            self.updateDisplay()
            
    def exponentiateImage(self):
        self.image = self.image.astype(float)
        self.image /= self.image.max()
        self.image = 1-self.image
        self.image **= self.imgExponent
        self.image = 1-self.image
        self.image *= 255/self.image.max()
        self.image = self.image.astype(np.uint8)
        
    def setExponentiateImage(self):
        self.getVideoImage()
        self.updateDisplay()
        
    def setExponent(self):
        val,ok = QtWidgets.QInputDialog.getDouble(self.mainWin,'Set Exponent','',value=self.imgExponent,min=0.01,decimals=2)
        if not ok:
            return
        self.imgExponent = val
        if self.trackMenuExpImage.isChecked():
            self.getVideoImage()
            self.updateDisplay()
                    
    def initDisplay(self):
        self.resetImage()
        if self.cam is None:
            self.frameNumSpinBox.setRange(1,self.numFrames)
            self.frameNumSpinBox.setValue(self.frameNum)
            self.frameNumSpinBox.setSuffix(' of '+str(self.numFrames))
            self.frameNumSpinBox.blockSignals(False)
            self.frameNumSpinBox.setEnabled(True)
        if self.optionsMenuShowTracking.isChecked():
            for line in self.frameNumLines:
                line.setBounds((0,(self.numFrames-1)/self.frameRate))
            self.addFrameNumLines()
            self.setDataPlotDur()
            self.setDataPlotTime()
            self.resetPupilData()
            self.resetPupilDataPlot()        
            self.setDataPlotXRange()
            self.trackMenu.setEnabled(True)
            self.analysisMenu.setEnabled(True)
        else:
            self.trackMenu.setEnabled(False)
            self.analysisMenu.setEnabled(False)
            
    def resetImage(self):
        self.imageItem.setImage(self.image[self.roiInd].T,levels=(0,255))
        self.imageViewBox.autoRange(items=[self.imageItem])
        if self.pupilCenterSeed is not None:
            self.pupilCenterPlot.setData(x=[self.pupilCenterSeed[0]],y=[self.pupilCenterSeed[1]])
            self.pupilEllipsePlot.setData(x=[],y=[])
        if self.reflectCenterSeed is not None:
            self.reflectCenterPlot.setData(x=[self.reflectCenterSeed[0]],y=[self.reflectCenterSeed[1]])
        self.getRadialLines()
               
    def resetROI(self,keepPosAndSize=False):
        if self.cam is not None:
            if self.camType=='vimba':
                self.cam.feature('OffsetX').value = 0
                self.cam.feature('OffsetY').value = 0
                self.cam.feature('Width').value = self.cam.feature('WidthMax').value
                self.cam.feature('Height').value = self.cam.feature('HeightMax').value
            self.getCamImage()
        self.fullRoiSize = (self.image.shape[1],self.image.shape[0])
        self.roiInd = np.s_[0:self.fullRoiSize[1],0:self.fullRoiSize[0]]
        if not keepPosAndSize:
            self.roiPos = (0,0)
            self.roiSize = (self.image.shape[1],self.image.shape[0])
        
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
        if self.stopTracking:
            self.toggleStopTracking()
        if self.setDataNan:
            self.toggleSetDataNan()
        
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
        if self.image is None:
            return
        elif self.startVideoButton.isChecked():
            self.turnOffButtons(source=self.startVideoButton)
            self.startVideoButton.setText('Stop Video')
            if self.cam is None:
                self.frameNumSpinBox.blockSignals(True)
                if self.frameNum==self.numFrames:
                    self.frameNum = 0
                    if self.optionsMenuShowTracking.isChecked():
                        self.setDataPlotXRange()
                    if self.videoIn is not None:
                        self.videoIn.set(cv2.CAP_PROP_POS_FRAMES,0)
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
                self.startCamera(bufferSize=self.camBufferSize)
                if self.camType=='vimba':
                    if self.optionsMenuShowTracking.isChecked():
                        self.resetPupilData()
                    for frame in self.camFrames:
                        frame.queue_for_capture(frame_callback=camFrameCaptured)
                    self.cam.AcquisitionStart()
                else:
                    while self.startVideoButton.isChecked():
                        isImage,image = self.cam.read()
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        self.processCamFrame(image,time.perf_counter())
                        self.app.processEvents()
        else:
            if self.cam is not None:
                self.stopCamera()
            self.updateDisplay(showAll=True)
            if self.cam is None:
                self.frameNumSpinBox.blockSignals(False)
            self.startVideoButton.setText('Start Video')
            
    def updateDisplay(self,showAll=False,showNone=False):
        n = (self.frameNum-1)%self.displayUpdateInterval
        if showAll or (not showNone and n==0):
            self.imageItem.setImage(self.image[self.roiInd].T,levels=(0,255))
        if self.cam is None:
            self.selectedSaccade = None
        elif self.camType=='vimba':
            if self.dataPlotIndex==self.numDataPlotPts-1:
                self.dataPlotIndex = 0
            else:
                self.dataPlotIndex += 1
        if not self.stopTracking and (self.setDataNan or self.reflectCenterSeed is not None):
            self.trackReflect()
            if showAll or (not showNone and n==0):
                if self.reflectFound:
                    self.reflectCenterPlot.setData(x=[self.reflectCenterSeed[0]],y=[self.reflectCenterSeed[1]])
                else:
                    self.reflectCenterPlot.setData(x=[],y=[])
        if not self.stopTracking and (self.setDataNan or self.pupilCenterSeed is not None):
            self.trackPupil()
            if showAll or (not showNone and n==0):
                self.updatePupilPlot()
        if self.pupilCenterSeed is not None or self.dataIsLoaded:
            if showAll or (not showNone and n==1):
                self.updatePupilDataPlot()
        if self.trackMenuAdaptThresh.isChecked():
            self.meanImageIntensity = self.image.mean()
            
    def setDisplayUpdateInterval(self):
        val,ok = QtWidgets.QInputDialog.getInt(self.mainWin,'Set Display Update Interval','Frames:',value=self.displayUpdateInterval,min=1,max=10)
        if ok:
            self.displayUpdateInterval = val
        
    def processCamFrame(self,img,timestamp):
        self.frameNum += 1
        self.image = img
        showNone = False
        if self.saveCheckBox.isChecked() or (self.nidaq and self.cameraMenuNidaqIn.isChecked() and self.nidaqDigitalIn.read()):
            if self.dataFileOut is None:
                self.frameNum = 0
                if self.cameraMenuNidaqIn.isChecked():
                    self.saveCheckBox.setChecked(True)
                fileName = os.path.join(self.camSavePath,self.camSaveBaseName+'_'+time.strftime('%Y%m%d_%H%M%S'))
                self.dataFileOut = h5py.File(fileName+'.hdf5','w',libver='latest')
                self.dataFileOut.attrs.create('frameRate',self.frameRate)
                self.dataFileOut.attrs.create('mmPerPixel',self.mmPerPixel)
                self.frameTimeDataset = self.dataFileOut.create_dataset('frameTimes',(0,),maxshape=(None,),dtype=float)
                if self.camSaveFileType=='.hdf5':
                    imgShape = tuple(self.roiSize[::-1])
                    self.frameDataset = self.dataFileOut.create_dataset('frames',(0,)+imgShape,maxshape=(None,)+imgShape,dtype=img.dtype,chunks=(1,)+imgShape,compression='gzip',compression_opts=1)
                else:
                    self.videoOut = self.skvideo.io.FFmpegWriter(fileName+self.camSaveFileType,inputdict={'-r':str(self.frameRate)},outputdict={'-r':str(self.frameRate),'-vcodec':'libx264','-crf':'17'})
                showNone = True
            else:
                if self.nidaq and self.cameraMenuNidaqOut.isChecked():
                    self.nidaqDigitalOut.write(True)
                if self.camType=='vimba':
                    timestamp /= self.cam.GevTimestampTickFrequency
                self.frameTimeDataset.resize(self.frameNum,axis=0)
                self.frameTimeDataset[-1] = timestamp
                if self.camSaveFileType=='.hdf5':
                    self.frameDataset.resize(self.frameNum,axis=0)
                    self.frameDataset[-1] = img[self.roiInd]
                else:
                    self.videoOut.writeFrame(img[self.roiInd])
                if self.nidaq and self.cameraMenuNidaqOut.isChecked():
                    self.nidaqDigitalOut.write(False)
        elif self.dataFileOut is not None:
            self.closeDataFileOut()
            showNone = True
        self.updateDisplay(showNone)       
        
    def closeDataFileOut(self):
        self.dataFileOut.close()
        self.dataFileOut = None
        if self.videoOut is not None:
            self.videoOut.close()
            self.videoOut = None
        self.frameNum = 0
        if self.cameraMenuNidaqIn.isChecked():
            self.saveCheckBox.setChecked(False)
            
    def toggleStopTracking(self):
        self.stopTracking = not self.stopTracking
        self.trackMenuStopTracking.setChecked(self.stopTracking)
        if self.stopTracking:
            self.reflectCenterPlot.setData(x=[],y=[])
            self.pupilCenterPlot.setData(x=[],y=[])
            self.pupilEllipsePlot.setData(x=[],y=[])
            if self.setDataNan:
                self.toggleSetDataNan()
            
    def toggleSetDataNan(self):
        self.setDataNan = not self.setDataNan
        self.trackMenuSetDataNan.setChecked(self.setDataNan)
        if self.setDataNan and self.cam is None and not any([button.isChecked() for button in self.buttons]):
            if self.stopTracking:
                self.toggleStopTracking()
            self.setCurrentFrameDataNan()
            
    def setCurrentFrameDataNan(self):
        self.reflectCenterPlot.setData(x=[],y=[])
        self.pupilCenterPlot.setData(x=[],y=[])
        self.pupilEllipsePlot.setData(x=[],y=[])
        self.pupilArea[self.frameNum-1] = np.nan
        self.pupilX[self.frameNum-1] = np.nan
        self.pupilY[self.frameNum-1] = np.nan
        if self.pupilCenterSeed is not None or self.dataIsLoaded:
            self.updatePupilDataPlot()
            
    def mainWinKeyPressEvent(self,event):
        key = event.key()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
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
        elif key in (QtCore.Qt.Key_Left,QtCore.Qt.Key_Right,QtCore.Qt.Key_Up,QtCore.Qt.Key_Down,QtCore.Qt.Key_Minus,QtCore.Qt.Key_Equal):
            if key in (QtCore.Qt.Key_Left,QtCore.Qt.Key_Right,QtCore.Qt.Key_Up,QtCore.Qt.Key_Down) and self.cam is None and not any([button.isChecked() for button in self.buttons]) and self.selectedSaccade is not None:
                if key in (QtCore.Qt.Key_Left,QtCore.Qt.Key_Right):
                    saccades = self.negSaccades if self.selectedSaccade in self.negSaccades else self.posSaccades
                    move = -1 if key==QtCore.Qt.Key_Left else 1
                    saccades[saccades==self.selectedSaccade] += move
                    self.selectedSaccade += move
                elif key==QtCore.Qt.Key_Up and self.selectedSaccade in self.negSaccades:
                    self.negSaccades = self.negSaccades[self.negSaccades!=self.selectedSaccade]
                    self.posSaccades = np.unique(np.concatenate((self.posSaccades,[self.selectedSaccade])))
                elif key==QtCore.Qt.Key_Down and self.selectedSaccade in self.posSaccades:
                    self.posSaccades = self.posSaccades[self.posSaccades!=self.selectedSaccade]
                    self.negSaccades = np.unique(np.concatenate((self.negSaccades,[self.selectedSaccade])))
                else:
                    return
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
        elif self.optionsMenuShowTracking.isChecked():
            if key==QtCore.Qt.Key_Escape:
                self.toggleStopTracking()
            elif key==QtCore.Qt.Key_N:
                if int(modifiers & QtCore.Qt.ControlModifier)>0:
                    self.toggleSetDataNan()
                elif self.cam is None and not any([button.isChecked() for button in self.buttons]):
                    self.setCurrentFrameDataNan()
            elif key==QtCore.Qt.Key_Space:
                if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                    self.changePlotWindowDur(fullRange=True)
            elif key==QtCore.Qt.Key_Delete:
                if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                    if int(modifiers & QtCore.Qt.ControlModifier)>0:
                        self.deleteAllSaccades()
                    elif self.selectedSaccade is not None:
                        self.negSaccades = self.negSaccades[self.negSaccades!=self.selectedSaccade]
                        self.posSaccades = self.posSaccades[self.posSaccades!=self.selectedSaccade]
                        self.selectedSaccade = None
                        self.plotSaccades()
                elif self.setMaskButton.isChecked() and len(self.maskRoi)>0:
                    self.imageViewBox.removeItem(self.maskRoi[-1])
                    del(self.maskRoi[-1])
                    del(self.maskIndex[-1])
            elif key==QtCore.Qt.Key_F:
                if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                    self.findSaccades()
            elif key==QtCore.Qt.Key_S:
                if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                    self.posSaccades = np.unique(np.concatenate((self.posSaccades,[self.frameNum-1])))
                    self.selectedSaccade = self.frameNum-1
                    self.plotSaccades()
    
    def imageMouseClickEvent(self,event):
        if event.button()==QtCore.Qt.RightButton and not self.roiButton.isChecked() and not self.findReflectButton.isChecked() and self.reflectCenterSeed is not None:
            if self.stopTracking:
                self.toggleStopTracking()
            if self.setDataNan:
                self.toggleSetDataNan()
            x,y = event.pos().x(),event.pos().y()
            self.reflectRoiPos = [[int(x-self.reflectRoiSize[0][0]/2),int(y-self.reflectRoiSize[0][1]/2)]]
            if not self.startVideoButton.isChecked():
                self.trackReflect()
                if self.reflectFound:
                    self.reflectCenterPlot.setData(x=[self.reflectCenterSeed[0]],y=[self.reflectCenterSeed[1]])
                    if self.pupilCenterSeed is not None:
                        self.trackPupil()
                        self.updatePupilPlot()
                        if self.findPupilButton.isChecked():
                            self.updatePupilTrackParamPlots()
                        else:
                            self.updatePupilDataPlot()
                else:
                    self.reflectCenterPlot.setData(x=[],y=[])
            
    def imageDoubleClickEvent(self,event):
        x,y = event.pos().x(),event.pos().y()
        if self.findReflectButton.isChecked():
            n = len(self.reflectRoi)
            if n<1 or self.trackMenuReflectTypeRing.isChecked():
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
            if self.stopTracking:
                self.toggleStopTracking()
            if self.setDataNan:
                self.toggleSetDataNan()
            self.pupilCenterSeed = (x,y)
            if not self.startVideoButton.isChecked() and self.trackMenuPupilMethodStarburst.isChecked():
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
                self.selectedSaccade = saccades[np.argmin(np.absolute(saccades-frame))]
            
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
                self.selectedSaccade = frame
                self.plotSaccades()
        
    def setROI(self):
        if self.image is None:
            return
        elif self.roiButton.isChecked():
            self.turnOffButtons(source=self.roiButton)
            self.resetROI(keepPosAndSize=True)
            self.resetImage()
            if self.roi is None:
                self.roi = pg.ROI(self.roiPos,self.roiSize,maxBounds=None,pen='r')
                self.roi.addScaleHandle(pos=(1,1),center=(0.5,0.5))
                self.imageViewBox.addItem(self.roi)
            else:
                self.roi.setVisible(True)
        else:
            newPos = []
            for p,maxSize in zip(self.roi.pos(),self.fullRoiSize):
                if p<0:
                    newPos.append(0)
                elif p>maxSize-1:
                    newPos.append(maxSize-1)
                else:
                    newPos.append(int(p))
            newSize = []
            for s,p,maxSize in zip(self.roi.size(),newPos,self.fullRoiSize):
                if s<1:
                    newSize.append(1)
                elif p+s>maxSize:
                    newSize.append(maxSize-p)
                else:
                    newSize.append(int(s))    
            self.roi.setPos(newPos)
            self.roi.setSize(newSize)
            deltaPos = [newPos[i]-self.roiPos[i] for i in (0,1)]
            self.roiPos = newPos
            self.roiSize = newSize
            self.pupilCenter -= deltaPos
            self.reflectCenter -= deltaPos
            if self.pupilCenterSeed is not None:
                self.pupilCenterSeed = (self.pupilCenterSeed[0]-deltaPos[0],self.pupilCenterSeed[1]-deltaPos[1])
            if self.reflectCenterSeed is not None:
                self.reflectCenterSeed = (self.reflectCenterSeed[0]-deltaPos[0],self.reflectCenterSeed[1]-deltaPos[1])
            if self.pupilRoi is not None:
                self.pupilRoiPos[0] -= deltaPos[0]
                self.pupilRoiPos[1] -= deltaPos[1]
            for i,roi in enumerate(self.reflectRoi):
                self.reflectRoiPos[i][0] -= deltaPos[0]
                self.reflectRoiPos[i][1] -= deltaPos[1]
            if len(self.maskRoi)>0:
                for roi in self.maskRoi:
                    roi.setPos((roi.pos()[0]-deltaPos[0],roi.pos()[1]-deltaPos[1]))
                self.updateMaskIndex()       
            if self.cam is None or self.camType=='webcam':
                self.roiInd = np.s_[self.roiPos[1]:self.roiPos[1]+self.roiSize[1],self.roiPos[0]:self.roiPos[0]+self.roiSize[0]]
            else:
                self.roiInd = np.s_[0:self.roiSize[1],0:self.roiSize[0]]
            if self.cam is not None:
                if self.camType=='vimba':
                    self.cam.feature('OffsetX').value = self.roiPos[0]
                    self.cam.feature('OffsetY').value = self.roiPos[1]
                    self.cam.feature('Width').value = self.roiSize[0]
                    self.cam.feature('Height').value = self.roiSize[1]
                self.getCamImage()
            self.roi.setVisible(False)
            self.resetImage()
            
    def setMask(self):
        if self.image is None:
            return
        elif self.setMaskButton.isChecked():
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
            
    def setUseMask(self):
        if self.cam is None and not any([button.isChecked() for button in self.buttons]) and self.pupilCenterSeed is not None:
            self.trackPupil()
            self.updatePupilPlot()
            self.updatePupilDataPlot()
     
    def findPupil(self):
        if self.image is None:
            return
        elif self.findPupilButton.isChecked():
            self.turnOffButtons(source=self.findPupilButton)
            if self.trackMenuPupilMethodStarburst.isChecked() or self.trackMenuPupilMethodLine.isChecked():
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
                    self.pupilRoiSize = [int(math.ceil(0.2*max(self.roiSize)))]*2
                    self.pupilRoiPos = [int(0.5*(self.roiSize[i]-self.pupilRoiSize[i])) for i in (0,1)]
                    self.pupilRoi = pg.ROI(self.pupilRoiPos,self.pupilRoiSize,maxBounds=maxBoundsRect,pen='r')
                    self.pupilRoi.addScaleHandle(pos=(1,1),center=(0.5,0.5))
                    self.pupilRoi.sigRegionChangeFinished.connect(self.pupilRoiRegionChanged)
                    self.imageViewBox.addItem(self.pupilRoi)
                else:
                    self.pupilRoi.setPos(self.pupilRoiPos)
                    self.pupilRoi.setSize(self.pupilRoiSize)
                    self.pupilRoi.setVisible(True)
        else:
            if self.trackMenuPupilMethodStarburst.isChecked() or self.trackMenuPupilMethodLine.isChecked():
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
        if not self.setDataNan and (self.reflectCenterSeed is None or self.reflectFound):
            img = self.image[self.roiInd].copy()
            if self.trackMenuPupilSignPos.isChecked():
                img = 255-img
            if self.useMaskCheckBox.isChecked() and len(self.maskRoi)>0:
                for ind in self.maskIndex:
                    img[ind] = 0
            if self.trackMenuAdaptThresh.isChecked():
                self.pupilEdgeThresh += self.image.mean()-self.meanImageIntensity
            if self.trackMenuPupilMethodStarburst.isChecked():
                self.findPupilWithStarburst(img)
            elif self.trackMenuPupilMethodLine.isChecked():
                self.findPupilWithLine(img)
            elif self.trackMenuPupilMethodGradients.isChecked():
                self.findPupilWithGradients(img)
            else:
                self.findPupilWithIntensity(img)
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
        img = cv2.GaussianBlur(img,(0,0),0.005*img.shape[1])
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
        
    def findPupilWithIntensity(self,img):
        imgRoi = img[self.pupilRoiPos[1]:self.pupilRoiPos[1]+self.pupilRoiSize[1],self.pupilRoiPos[0]:self.pupilRoiPos[0]+self.pupilRoiSize[0]]
        maxInd = np.unravel_index(np.argmax(imgRoi),imgRoi.shape)
        self.pupilCenterSeed = [self.pupilRoiPos[0]+maxInd[1],self.pupilRoiPos[1]+maxInd[0]]
        self.pupilRoiIntensity = imgRoi.mean()
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
        methods = (self.trackMenuPupilMethodStarburst,self.trackMenuPupilMethodLine,self.trackMenuPupilMethodGradients,self.trackMenuPupilMethodIntensity) 
        params = (self.trackMenuCircularity,self.trackMenuLineOrigin,self.trackMenuGradientDownsamp,None)
        for method,param in zip(methods,params):
            isSelected = method is self.mainWin.sender()
            method.setChecked(isSelected)
            if param is not None:
                param.setEnabled(isSelected)
        self.pupilCenterSeed = None
        self.pupilCenterPlot.setData(x=[],y=[])
        self.pupilEllipsePlot.setData(x=[],y=[])
        
    def setAdaptiveThreshold(self):
        self.meanImageIntensity = self.image.mean()
        
    def setCircularityThresh(self):
        val,ok = QtWidgets.QInputDialog.getDouble(self.mainWin,'Set pupil circularity threshold','ellipse axis length ratio:',value=self.pupilCircularityThresh,min=0.01,max=0.99,decimals=2)
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
        val,ok = QtWidgets.QInputDialog.getDouble(self.mainWin,'Set pupil gradient downsample','fraction of pixels:',value=self.pupilGradientDownsample,min=0.1,max=1,decimals=2)
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
            elif self.trackMenuPupilMethodIntensity.isChecked():
                self.pupilArea[self.dataPlotIndex] = self.pupilRoiIntensity
            if self.reflectCenterSeed is None:
                self.pupilX[self.dataPlotIndex] = self.pupilCenterSeed[0]
                self.pupilY[self.dataPlotIndex] = self.pupilCenterSeed[1] 
            elif self.trackMenuReflectTypeSpot.isChecked() or not self.trackMenuPupilMethodStarburst.isChecked() or np.isnan(self.mmPerPixel):
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
        self.minNumPixAboveThresh = int(round(self.numPixAboveThreshLine.value()))
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
        if self.image is None:
            return
        elif self.findReflectButton.isChecked():
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
            if self.trackMenuReflectTypeSpot.isChecked() or len(self.reflectRoi)>1:
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
        spotCenters = np.zeros((len(self.reflectRoi),2))
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
        val,ok = QtWidgets.QInputDialog.getInt(self.mainWin,'Set Reflection Threshold','Pixel intensity:',value=self.reflectThresh,min=0,max=254)
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
        if self.videoIn is not None:
            self.videoIn.set(cv2.CAP_PROP_POS_FRAMES,self.frameNum-1)
        self.getVideoImage()
        self.updateDisplay(showAll=True)
    
    def changePlotWindowDur(self,fullRange=False):
        if fullRange:
            newVal = self.numFrames/self.frameRate
        else:
            newVal = float(self.plotDurEdit.text())
            if self.cam is None:
                if newVal<3/self.frameRate:
                    newVal = 3/self.frameRate
                elif newVal>self.numFrames/self.frameRate:
                    newVal = self.numFrames/self.frameRate
            else:
                if np.isnan(self.frameRate):
                    newVal = 3 if newVal<3 else int(newVal)
                elif newVal<3/self.frameRate:
                    newVal = 3/self.frameRate
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

    def setDataPlotDur(self):
        self.dataPlotDur = self.defaultDataPlotDur
        self.pupilYPlotItem.setLabel('bottom','Time (s)')
        if np.isnan(self.frameRate):
            self.dataPlotDur = 60
            self.pupilYPlotItem.setLabel('bottom','Frame')
        elif self.cam is None and self.defaultDataPlotDur>self.numFrames/self.frameRate:
            self.dataPlotDur = self.numFrames/self.frameRate   
        self.plotDurEdit.setText(str(round(self.dataPlotDur,3)))
            
    def setDataPlotTime(self):
        if np.isnan(self.frameRate):
            self.dataPlotTime = np.arange(self.dataPlotDur)
        else:
            self.dataPlotTime = np.arange(0,self.dataPlotDur-0.5/self.frameRate,1/self.frameRate)
        self.numDataPlotPts = self.dataPlotTime.size
        
    def setMmPerPix(self):
        val = 0 if np.isnan(self.mmPerPixel) else self.mmPerPixel
        val,ok = QtWidgets.QInputDialog.getDouble(self.mainWin,'Set mm/pixel','mm/pixel:',value=val,min=0,decimals=4)
        if ok:
            self.mmPerPixel = val if val>0 else np.nan
    
    def measureMmPerPix(self):
        if self.reflectCenterSeed is None:
            QtWidgets.QMessageBox.about(self.mainWin,'Set mm/pixel','First find reflection')
        else:
            p = int(0.5*self.frameRate)
            avgPts = p if self.numDataPlotPts>=p else self.numDataPlotPts
            initialReflectCenter = self.getAvgReflectCenter(avgPts)
            QtWidgets.QMessageBox.about(self.mainWin,'Set mm/pixel','Move camera 0.5 mm; then press ok')
            finalReflectCenter = self.getAvgReflectCenter(avgPts)
            self.mmPerPixel = 0.5/np.sqrt(np.sum((finalReflectCenter-initialReflectCenter)**2))
            
    def getAvgReflectCenter(self,avgPts):
        return np.mean(np.tile(self.reflectCenter,(2,1))[:self.numDataPlotPts+self.dataPlotIndex,:][-avgPts:,:],axis=0)
            
    def analyzeAllFrames(self):
        while self.frameNum<self.numFrames:
            self.frameNum += 1
            self.getVideoImage()
            if self.frameNum==self.numFrames:
                self.updateDisplay(showAll=True)
                self.changePlotWindowDur(fullRange=True)
            else:
                self.updateDisplay(showNone=True)
        
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
        if np.all(np.isnan(self.frameTimes)):
            return
        frameIntervals = np.diff(self.frameTimes)*1e3
        plt.figure()
        plt.plot(range(2,self.numFrames+1),frameIntervals)
        plt.axis([1,self.numFrames,0,max(frameIntervals)*1.1])
        plt.xlabel('Frame Number')
        plt.ylabel('Frame Interval (ms)')
        plt.show()
        
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
        if np.all(np.isnan(frameTimes)):
            t = np.arange(self.numFrames)/self.frameRate
        else:
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
        val,ok = QtWidgets.QInputDialog.getInt(self.mainWin,'Set Saccade Smoothing','number of points:',value=self.saccadeSmoothPts,min=1)
        if ok:
            self.saccadeSmoothPts = val
        
    def setSaccadeThresh(self):
        val,ok = QtWidgets.QInputDialog.getDouble(self.mainWin,'Set Saccade Threshold','standard devations from baseline:',value=self.saccadeThresh,min=0.1,decimals=1)
        if ok:
            self.saccadeThresh = val
            
    def setSaccadeRefractoryPeriod(self):
        val,ok = QtWidgets.QInputDialog.getDouble(self.mainWin,'Set Saccade Refractory Period','seconds:',value=self.saccadeRefractoryPeriod,min=0.001,decimals=3)
        if ok:
            self.saccadeRefractoryPeriod = val
        

if __name__=="__main__":
    start()