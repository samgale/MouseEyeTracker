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
from scipy import signal
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
from matplotlib import pyplot as plt


def start():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    w = MouseEyeTracker(app)
    app.exec_()


class MouseEyeTracker():
    
    def __init__(self,app):
        self.app = app
        self.fileOpenPath = os.path.dirname(os.path.realpath(__file__))
        self.fileSavePath = copy.copy(self.fileOpenPath)
        self.camSavePath = copy.copy(self.fileOpenPath)
        self.camSaveBaseName = 'MouseEyeTracker'
        self.nidaq = False
        self.vimba = None
        self.cam = None
        self.frame = None
        self.video = None
        self.dataFileIn = None
        self.dataFileOut = None
        self.image = None
        self.roi = None
        self.pupilCenter = None
        self.reflectCenter = None
        self.reflectRoi = []
        self.reflectThresh = 254
        self.maskRoi = []
        self.mmPerPixel = np.nan
        self.lensRotRadius = 1.25
        self.lensOffset = 0.1
        self.corneaOffset = 0.2
        self.defaultDataPlotDur = 2.0
        self.dataIsLoaded = False
        
        # signal generator object
        self.sigGen = SignalGenerator()
        self.sigGen.camFrameCapturedSignal.connect(self.processCamFrame)
        
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
        self.fileMenuOpen.triggered.connect(self.openFile)
        self.fileMenu.addAction(self.fileMenuOpen)
        
        self.fileMenuSave = self.fileMenu.addMenu('Save')
        self.fileMenuSave.setEnabled(False)
        self.fileMenuSaveMovie = QtGui.QAction('Movie',self.mainWin)
        self.fileMenuSaveMovie.triggered.connect(self.saveMovie)
        self.fileMenuSaveAnnotatedMovie = QtGui.QAction('Annotated Movie',self.mainWin,enabled=False)
        self.fileMenuSaveAnnotatedMovie.triggered.connect(self.saveAnnotatedMovie)
        self.fileMenuSaveHDF5 = QtGui.QAction('HDF5',self.mainWin)
        self.fileMenuSaveHDF5.triggered.connect(self.saveHDF5)
        self.fileMenuSave.addActions([self.fileMenuSaveMovie,self.fileMenuSaveAnnotatedMovie,self.fileMenuSaveHDF5])
        
        # camera menu
        self.cameraMenu = self.menuBar.addMenu('Camera')         
        self.cameraMenuUseCam = QtGui.QAction('Use Camera',self.mainWin,checkable=True)
        self.cameraMenuUseCam.triggered.connect(self.initCamera)
        self.cameraMenu.addAction(self.cameraMenuUseCam)
        
        self.cameraMenuSettings = self.cameraMenu.addMenu('Settings')
        self.cameraMenuSettings.setEnabled(False)
        self.cameraMenuSettingsBinning = QtGui.QAction('Spatial Binning',self.mainWin)
        self.cameraMenuSettingsBinning.triggered.connect(self.setCamBinning)
        self.cameraMenuSettingsExposure = QtGui.QAction('Exposure',self.mainWin)
        self.cameraMenuSettingsExposure.triggered.connect(self.setCamExposure)
        self.cameraMenuSettingsFrameRate = QtGui.QAction('Frame Rate',self.mainWin)
        self.cameraMenuSettingsFrameRate.triggered.connect(self.setCamFrameRate)
        self.cameraMenuSettings.addActions([self.cameraMenuSettingsBinning,self.cameraMenuSettingsExposure,self.cameraMenuSettingsFrameRate])
        
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
        
        # tools menu
        self.toolsMenu = self.menuBar.addMenu('Tools')
        self.toolsMenuReflect = self.toolsMenu.addMenu('Reflection')
        self.toolsMenuReflectType = self.toolsMenuReflect.addMenu('Set Type')
        self.toolsMenuReflectTypeSpot = QtGui.QAction('Spot',self.mainWin,checkable=True)
        self.toolsMenuReflectTypeSpot.setChecked(True)
        self.toolsMenuReflectTypeSpot.triggered.connect(self.setReflectType)
        self.toolsMenuReflectTypeRing = QtGui.QAction('Ring',self.mainWin,checkable=True)
        self.toolsMenuReflectTypeRing.triggered.connect(self.setReflectType)
        self.toolsMenuReflectType.addActions([self.toolsMenuReflectTypeSpot,self.toolsMenuReflectTypeRing])
        self.toolsMenuReflectThresh = QtGui.QAction('Set Threshold',self.mainWin)
        self.toolsMenuReflectThresh.triggered.connect(self.setReflectThresh)
        self.toolsMenuReflect.addAction(self.toolsMenuReflectThresh)
        
        self.toolsMenuMmPerPix = self.toolsMenu.addMenu('mm/pixel')
        self.toolsMenuMmPerPixSet = QtGui.QAction('Set',self.mainWin)
        self.toolsMenuMmPerPixSet.triggered.connect(self.setMmPerPix)
        self.toolsMenuMmPerPixMeasure = QtGui.QAction('Measure',self.mainWin,enabled=False)
        self.toolsMenuMmPerPixMeasure.triggered.connect(self.measureMmPerPix)
        self.toolsMenuMmPerPix.addActions([self.toolsMenuMmPerPixSet,self.toolsMenuMmPerPixMeasure])
        
        self.toolsMenuAnalyzeAll = QtGui.QAction('Analyze All Frames',self.mainWin,enabled=False)
        self.toolsMenuAnalyzeAll.triggered.connect(self.analyzeAllFrames)
        self.toolsMenuLoadData = QtGui.QAction('Load Analyzed Data',self.mainWin,enabled=False)
        self.toolsMenuLoadData.triggered.connect(self.loadAnalyzedData)
        self.toolsMenuFrameIntervals = QtGui.QAction('Plot Frame Intervals',self.mainWin,enabled=False)
        self.toolsMenuFrameIntervals.triggered.connect(self.plotFrameIntervals)
        self.toolsMenu.addActions([self.toolsMenuAnalyzeAll,self.toolsMenuLoadData,self.toolsMenuFrameIntervals])
        
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
        
        # start video button
        self.startVideoButton = QtGui.QPushButton('Start Video',checkable=True)
        self.startVideoButton.clicked.connect(self.startVideo)
        
        # set roi button
        self.roiButton = QtGui.QPushButton('Set ROI',checkable=True)
        self.roiButton.clicked.connect(self.setROI)
        
        # find pupil button
        self.findPupilButton = QtGui.QPushButton('Find Pupil',checkable=True)
        self.findPupilButton.clicked.connect(self.findPupil)
        
        # find reflection button
        self.findReflectButton = QtGui.QPushButton('Find Reflection',checkable=True)
        self.findReflectButton.clicked.connect(self.findReflect)
        
        # set mask button
        self.setMaskButton = QtGui.QPushButton('Set Masks',checkable=True)
        self.setMaskButton.clicked.connect(self.setMask)
        
        self.buttons = (self.startVideoButton,self.roiButton,self.findPupilButton,self.findReflectButton,self.setMaskButton)
        
        # plot windows
        self.dataPlotLayout = pg.GraphicsLayoutWidget()
        self.pupilAreaPlotItem = self.dataPlotLayout.addPlot(row=0,col=0,enableMenu=False)
        self.pupilAreaPlotItem.setMouseEnabled(x=False,y=False)
        self.pupilAreaPlotItem.hideButtons()
        self.pupilAreaPlotItem.setLabel('left','Pupil Area')
        self.pupilAreaPlotItem.mouseClickEvent = self.dataPlotMouseClickEvent
        self.pupilAreaPlotItem.mouseDoubleClickEvent = self.dataPlotDoubleClickEvent
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
        self.pupilYPlotItem.mouseClickEvent = self.dataPlotMouseClickEvent
        self.pupilYPlotItem.mouseDoubleClickEvent = self.dataPlotDoubleClickEvent
        self.pupilYPlot = self.pupilYPlotItem.plot(x=[0,self.defaultDataPlotDur],y=[0,0])
        self.pupilYPlotItem.disableAutoRange()
        
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
        
        # save checkbox
        self.saveCheckBox = QtGui.QCheckBox('Save Video Data',enabled=False)
        
        # use mask checkbox
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
        
        # plot window duration edit
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
        
    def saveMovie(self):
        filePath = QtGui.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileSavePath,'*.avi')
        if filePath=='':
            return
        self.fileSavePath = os.path.dirname(filePath)
        vidOut = cv2.VideoWriter(filePath,-1,self.frameRate,self.roiSize)
        if self.dataFileIn is None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES,0)
        else:
            frame = 0
        while True:
            if self.dataFileIn is None:
                isImage,image = self.video.read()
                if not isImage:
                    break
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            else:
                frame += 1
                if frame==len(self.dataFileIn.keys()):
                    break
                image = self.dataFileIn[str(frame)][:,:]
            vidOut.write(image[self.roiInd])
        vidOut.release()
        if self.dataFileIn is None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES,self.frameNum-1)
        
    def saveAnnotatedMovie(self):
        filePath = QtGui.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileSavePath,'*.avi')
        if filePath=='':
            return
        self.fileSavePath = os.path.dirname(filePath)
            
    def saveHDF5(self):
        filePath = QtGui.QFileDialog.getSaveFileName(self.mainWin,'Save As',self.fileSavePath,'*.hdf5')
        if filePath=='':
            return
        self.fileSavePath = os.path.dirname(filePath)
        dataFile = h5py.File(filePath,'w',libver='latest')
        for param in ('pupilArea','pupilX','pupilY'):
            dataFile.create_dataset(param,data=getattr(self,param),compression='gzip',compression_opts=1)
        if self.dataFileIn is not None:
            self.getFrameTimes()
            dataFile.create_dataset('frameTimes',data=self.frameTimes,compression='gzip',compression_opts=1)
        dataFile.close()
               
    def openFile(self):
        filePath = QtGui.QFileDialog.getOpenFileName(self.mainWin,'Choose File',self.fileOpenPath,'*.avi *.mov *.hdf5')
        if filePath=='':
            return
        self.fileOpenPath = os.path.dirname(filePath)
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
                self.numFrames = sum(1 for _ in self.dataFileIn.iterkeys())
            self.frameTimes = np.full(self.numFrames,np.nan)
            self.mmPerPixel = self.dataFileIn.attrs.get('mmPerPixel')
            self.toolsMenuFrameIntervals.setEnabled(True)
        else:
            self.video = cv2.VideoCapture(filePath)
            self.frameRate = self.video.get(cv2.CAP_PROP_FPS)
            self.numFrames = int(round(self.video.get(cv2.CAP_PROP_FRAME_COUNT)))
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
        self.toolsMenuAnalyzeAll.setEnabled(True)
        self.toolsMenuLoadData.setEnabled(True)
        for line in self.frameNumLines:
            line.setBounds((0,(self.numFrames-1)/self.frameRate))
        self.addFrameNumLines()
        self.plotDurEdit.setText(str(self.dataPlotDur))
        self.pupilArea = np.full(self.numFrames,np.nan)
        self.pupilX = np.full(self.numFrames,np.nan)
        self.pupilY = np.full(self.numFrames,np.nan)
        self.frameNum = 1
        self.getVideoImage()
        self.initDisplay()
        
    def closeDataFileIn(self):
        self.closeFileCleanup()
        self.toolsMenuFrameIntervals.setEnabled(False)
        self.dataIsLoaded = False
        self.frameTimes = []
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
        self.toolsMenuAnalyzeAll.setEnabled(False)
        self.toolsMenuLoadData.setEnabled(False)
        self.removeFrameNumLines()
        for line in self.frameNumLines:
            line.setValue(0)
        self.resetPupilTracking()
       
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
            self.resetPupilData()
            self.cameraMenuSettings.setEnabled(True)
            self.toolsMenuMmPerPixMeasure.setEnabled(True)
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
        self.toolsMenuMmPerPixMeasure.setEnabled(False)
        self.saveCheckBox.setEnabled(False)
        self.resetPupilTracking()
        
    def getCamImage(self):
        self.startCamera()
        self.frame.queueFrameCapture()
        self.cam.runFeatureCommand("AcquisitionStart")
        self.frame.waitFrameCapture()
        self.image = np.ndarray(buffer=self.frame.getBufferByteData(),dtype=np.uint8,shape=(self.frame.height,self.frame.width,1)).squeeze()
        self.stopCamera()
        
    def startCamera(self):
        self.cameraMenu.setEnabled(False)
        if self.nidaq:
            self.nidaqDigInputs.StartTask()
            self.nidaqDigOutputs.StartTask()
            self.nidaqDigOutputs.Write(np.zeros(self.nidaqDigOutputs.deviceLines,dtype=np.uint8))
        self.frame = self.cam.getFrame()
        self.frame.announceFrame()
        self.cam.startCapture()
        
    def stopCamera(self):
        self.cam.runFeatureCommand("AcquisitionStop")
        self.cam.endCapture()
        self.cam.revokeAllFrames()
        self.frame = None
        if self.dataFileOut is not None:
            self.closeDataFileOut()
        if self.nidaq:
            self.nidaqDigInputs.StopTask()
            self.nidaqDigOutputs.StopTask()
        self.cameraMenu.setEnabled(True)
            
    def setCamProps(self):
        self.frameRate = 60.0
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
            
    def setCamBinning(self):
        val,ok = QtGui.QInputDialog.getInt(self.mainWin,'Set Camera Spatial Binning','Pixels:',value=self.cam.BinningHorizontal,min=1,max=8,step=1)
        if not ok:
            return
        scaleFactor = self.cam.BinningHorizontal/val
        if self.pupilCenter is not None:
            self.pupilCenter = [int(n*scaleFactor) for n in self.pupilCenter]
        if self.reflectCenter is not None:
            self.reflectCenter = [int(n*scaleFactor) for n in self.reflectCenter]
        for i,roi in enumerate(self.reflectRoi):
            self.reflectRoiPos[i] = [int(n*scaleFactor) for n in roi.pos()]
            self.reflectRoiSize[i] = [int(n*scaleFactor) for n in roi.size()]
            roi.setPos(self.reflectRoiPos[i])
            roi.setSize(self.reflectRoiSize[i])
        if len(self.maskRoi)>0:
            for roi in self.maskRoi:
                roi.setPos([int(n*scaleFactor) for n in roi.pos()])
                roi.setSize([int(n*scaleFactor) for n in roi.size()])
            self.updateMaskIndex()
        self.camExposure *= self.cam.BinningHorizontal/val
        if self.camExposure>0.99:
            self.camExposure = 0.99
        elif self.camExposure<0.01:
            self.camExposure = 0.01
        self.cam.ExposureTimeAbs = self.camExposure*1e6/self.frameRate
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
        self.resetPupilDataPlot()        
        self.setDataPlotTime()
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
        if self.pupilCenter is not None:
            self.pupilCenterPlot.setData(x=[self.pupilCenter[0]],y=[self.pupilCenter[1]])
            self.pupilEllipsePlot.setData(x=[],y=[])
        if self.reflectCenter is not None:
            self.reflectCenterPlot.setData(x=[self.reflectCenter[0]],y=[self.reflectCenter[1]])
        self.getRadialLines()
        
    def resetPupilTracking(self):
        self.pupilCenterPlot.setData(x=[],y=[])
        self.pupilEllipsePlot.setData(x=[],y=[])
        self.pupilCenter = None
        self.reflectCenterPlot.setData(x=[],y=[])
        self.reflectCenter = None
        for roi in self.reflectRoi:
            self.imageViewBox.removeItem(roi)
        self.reflectRoi = []
        for roi in self.maskRoi:
            self.imageViewBox.removeItem(roi)
        self.maskRoi = []
        
    def resetPupilData(self):
        self.dataPlotFrameInd = 0
        self.pupilArea = np.full(self.numDataPlotPts,np.nan)
        self.pupilX = np.full(self.numDataPlotPts,np.nan)
        self.pupilY = np.full(self.numDataPlotPts,np.nan)
            
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
                self.startCamera()
                self.frame.queueFrameCapture(frameCallback=self.camFrameCaptured)
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
        if self.reflectCenter is not None:
            self.trackReflect()
            if updateAll or (not updateNone and n==1):
                if self.reflectFound:
                    self.reflectCenterPlot.setData(x=[self.reflectCenter[0]],y=[self.reflectCenter[1]])
                else:
                    self.reflectCenterPlot.setData(x=[],y=[])
        if self.pupilCenter is not None:
            if self.cam is not None:
                self.dataPlotFrameInd += 1
            self.trackPupil()
            if updateAll or (not updateNone and n==1):
                self.updatePupilPlot()
        if self.pupilCenter is not None or self.dataIsLoaded:
            if updateAll:
                self.updatePupilDataPlot()
            elif not updateNone and n>1:
                updatePlotN = [False,False,False]
                updatePlotN[n-2] = True
                self.updatePupilDataPlot(updatePlotN)
                
    def updateDisplayWithNan(self):
        self.reflectCenterPlot.setData(x=[],y=[])
        self.pupilCenterPlot.setData(x=[],y=[])
        self.pupilEllipsePlot.setData(x=[],y=[])
        self.pupilArea[self.frameNum-1] = np.nan
        self.pupilX[self.frameNum-1] = np.nan
        self.pupilY[self.frameNum-1] = np.nan
        self.updatePupilDataPlot()
            
    def camFrameCaptured(self,frame):
        img = np.ndarray(buffer=frame.getBufferByteData(),dtype=np.uint8,shape=(frame.height,frame.width,1)).squeeze()
        self.sigGen.camFrameCapturedSignal.emit(img,frame._frame.timestamp)
        frame.queueFrameCapture(frameCallback=self.camFrameCaptured)
        
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
            
    def mainWinKeyPressEvent(self,event):
        key = event.key()
        if key in (QtCore.Qt.Key_Comma,QtCore.Qt.Key_Period):
            if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                if key==QtCore.Qt.Key_Comma:
                    if self.frameNum==1:
                        return
                    self.frameNum -= 1
                else:
                    if self.frameNum==self.numFrames:
                        return
                    self.frameNum += 1
                self.frameNumSpinBox.setValue(self.frameNum)
        elif key==QtCore.Qt.Key_N:
            if self.cam is None and not any([button.isChecked() for button in self.buttons]):
                self.updateDisplayWithNan()
        elif key in (QtCore.Qt.Key_Left,QtCore.Qt.Key_Right,QtCore.Qt.Key_Up,QtCore.Qt.Key_Down,QtCore.Qt.Key_Minus,QtCore.Qt.Key_Equal):
            if self.roiButton.isChecked():
                roi = self.roi
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
        elif key==QtCore.Qt.Key_Delete:
            if self.setMaskButton.isChecked() and len(self.maskRoi)>0:
                self.imageViewBox.removeItem(self.maskRoi[-1])
                del(self.maskRoi[-1])
                del(self.maskIndex[-1])
                
    def imageMouseClickEvent(self,event):
        if event.button() == QtCore.Qt.RightButton and not self.roiButton.isChecked() and not self.findReflectButton.isChecked() and self.reflectCenter is not None:
            x,y = event.pos().x(),event.pos().y()
            self.reflectRoiPos = [[int(x-self.reflectRoiSize[0][0]/2),int(y-self.reflectRoiSize[0][1]/2)]]
            if not self.startVideoButton.isChecked():
                self.trackReflect()
                if self.reflectFound:
                    self.reflectCenterPlot.setData(x=[self.reflectCenter[0]],y=[self.reflectCenter[1]])
                else:
                    self.reflectCenterPlot.setData(x=[],y=[])
            
    def imageDoubleClickEvent(self,event):
        x,y = event.pos().x(),event.pos().y()
        if self.findReflectButton.isChecked():
            n = len(self.reflectRoi)
            if n<1 or (self.toolsMenuReflectTypeRing.isChecked() and n<4):
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
        elif not self.roiButton.isChecked() and (self.findPupilButton.isChecked() or self.pupilCenter is not None):
            self.pupilCenter = (x,y)
            if self.findPupilButton.isChecked() or not self.startVideoButton.isChecked():
                self.trackPupil()
                self.updatePupilPlot()
                if self.findPupilButton.isChecked():
                    self.updatePupilTrackParamPlots()
                else:
                    self.updatePupilDataPlot()
                    
    def dataPlotMouseClickEvent(self,event):
        if event.button() == QtCore.Qt.RightButton and self.cam is None and not any([button.isChecked() for button in self.buttons]):
            validData = np.logical_not(np.isnan(self.pupilArea))
            if any(validData):
                self.frameNumSpinBox.setValue(np.where(validData)[0][-1]+1)
            
    def dataPlotDoubleClickEvent(self,event):
        if self.cam is None and not any([button.isChecked() for button in self.buttons]):
            self.changePlotWindowDur(fullRange=True)
        
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
                # self.roi.setMaxBounds ?
                self.roi.setVisible(True)
            if self.pupilCenter is not None:
                self.pupilCenter = (self.pupilCenter[0]+self.roiPos[0],self.pupilCenter[1]+self.roiPos[1])
            if self.reflectCenter is not None:
                self.reflectCenter = (self.reflectCenter[0]+self.roiPos[0],self.reflectCenter[1]+self.roiPos[1])
            for i,roi in enumerate(self.reflectRoi):
                roi.setPos((roi.pos()[0]+self.roiPos[0],roi.pos()[1]+self.roiPos[1]))
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
            if self.pupilCenter is not None:
                self.pupilCenter = (self.pupilCenter[0]-self.roiPos[0],self.pupilCenter[1]-self.roiPos[1])
            if self.reflectCenter is not None:
                self.reflectCenter = (self.reflectCenter[0]-self.roiPos[0],self.reflectCenter[1]-self.roiPos[1])
            for i,roi in enumerate(self.reflectRoi):
                roi.setPos((roi.pos()[0]-self.roiPos[0],roi.pos()[1]-self.roiPos[1]))
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
            if self.pupilCenter is not None:
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
            if self.pupilCenter is None:
                self.pupilEdgeThresh = 2*self.image[self.roiInd][self.image[self.roiInd]>0].min()
                self.minNumPixAboveThresh = 2
                self.edgeFilt = np.ones(self.minNumPixAboveThresh)
                self.edgeDistThreshOffset = 0
                self.edgeDistThreshFactor = 6
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
            if self.pupilCenter is not None:
                self.updatePupilTrackParamPlots()
        else:
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
            if self.pupilCenter is not None:
                self.pupilAreaRange = [self.pupilArea[self.dataPlotFrameInd]]*2
                self.pupilXRange = [self.pupilX[self.dataPlotFrameInd]]*2
                self.pupilYRange = [self.pupilY[self.dataPlotFrameInd]]*2
                self.setDataPlotXRange()
                self.updatePupilDataPlot()
                
    def trackPupil(self):
        self.pupilFound = False
        if 0<self.pupilCenter[0]<self.roiSize[0]-1 and 0<self.pupilCenter[1]<self.roiSize[1]-1 and (self.reflectCenter is None or self.reflectFound):
            # get radial profiles and find pupil edges
            # radial profile must cross edge thresh for min num of consecutive pix
            # radial profile = 0 for masked pixels
            img = self.image[self.roiInd]
            if self.useMaskCheckBox.isChecked() and len(self.maskRoi)>0:
                for ind in self.maskIndex:
                    img[ind] = 0
            x = self.radialLinesX+int(self.pupilCenter[0])
            y = self.radialLinesY+int(self.pupilCenter[1])
            inFrame = np.logical_and(np.logical_and(x>=0,x<self.roiSize[0]),np.logical_and(y>=0,y<self.roiSize[1]))
            self.radialProfiles[:,:] = 0
            self.pupilEdges = np.zeros((self.numRadialLines*2,2),dtype=np.float32)
            for i in range(self.numRadialLines):
                xInFrame = x[i,inFrame[i,:]]
                yInFrame = y[i,inFrame[i,:]]
                lineProfile = img[yInFrame,xInFrame]
                centerInd = np.where(np.logical_and(xInFrame==int(self.pupilCenter[0]),yInFrame==int(self.pupilCenter[1])))[0][0]
                self.radialProfiles[i,0:lineProfile.size-centerInd] = lineProfile[centerInd:]
                self.radialProfiles[i+self.numRadialLines,:centerInd+1] = lineProfile[centerInd::-1]
                if self.minNumPixAboveThresh>1:
                    edgeInd1 = np.where(np.correlate(self.radialProfiles[i,:]>self.pupilEdgeThresh,self.edgeFilt,mode='valid')==self.minNumPixAboveThresh)[0]
                    edgeInd2 = np.where(np.correlate(self.radialProfiles[i+self.numRadialLines,:]>self.pupilEdgeThresh,self.edgeFilt,mode='valid')==self.minNumPixAboveThresh)[0]
                else:
                    edgeInd1 = np.where(self.radialProfiles[i,:]>self.pupilEdgeThresh)[0]
                    edgeInd2 = np.where(self.radialProfiles[i+self.numRadialLines,:]>self.pupilEdgeThresh)[0]
                if edgeInd1.size>0:
                    self.pupilEdges[i,0] = xInFrame[centerInd+edgeInd1[0]]
                    self.pupilEdges[i,1] = yInFrame[centerInd+edgeInd1[0]]
                if edgeInd2.size>0:
                    self.pupilEdges[i+self.numRadialLines,0] = xInFrame[centerInd-edgeInd2[0]]
                    self.pupilEdges[i+self.numRadialLines,1] = yInFrame[centerInd-edgeInd2[0]]
            # fit ellipse to edge points
            # throw out edge points with outlier distances from center
            self.pupilEdges = self.pupilEdges[self.pupilEdges.any(axis=1),:]
            if self.pupilEdges.shape[0]>0:
                self.pupilEdgeDist = np.sqrt(np.sum((self.pupilEdges-self.pupilCenter)**2,axis=1))
                meanEdgeDist = self.pupilEdgeDist.sum()/self.pupilEdgeDist.size
                self.pupilEdges = self.pupilEdges[np.absolute(self.pupilEdgeDist-(meanEdgeDist+self.edgeDistThreshOffset))<self.edgeDistThreshFactor*np.sqrt(np.sum((self.pupilEdgeDist-meanEdgeDist)**2)/self.pupilEdgeDist.size)]
                if self.pupilEdges.shape[0]>4:
                    center,halfLength,angle = cv2.fitEllipse(self.pupilEdges)
                    if 0<center[0]<self.roiSize[0]-1 and 0<center[1]<self.roiSize[1]-1:
                        self.pupilCenter,self.pupilEllipseHalfLength,self.pupilEllipseAngle = center,halfLength,angle
                        self.pupilFound = True
        self.updatePupilData()
        
    def getRadialLines(self):
        angles = np.arange(0,90,20)
        slopes = np.append(np.nan,1/np.tan(np.radians(angles[1:])))
        self.numRadialLines = 2*angles.size-1
        maxLength = max(self.roiSize)
        self.radialLinesX = np.zeros((self.numRadialLines,maxLength*2),dtype=np.int16)
        self.radialLinesY = np.zeros((self.numRadialLines,maxLength*2),dtype=np.int16)
        for i,angle in enumerate(angles):
            if angle==0:
                self.radialLinesY[i,:] = np.arange(-maxLength,maxLength)+1
            elif angle==90:
                self.radialLinesX[i,:] = np.arange(-maxLength,maxLength)+1
            elif angle==45:
                self.radialLinesX[i,:] = np.arange(-maxLength,maxLength)+1
                self.radialLinesY[i,:] = np.arange(-maxLength,maxLength)+1
            elif angle<45:
                self.radialLinesY[i,:] = np.arange(-maxLength,maxLength)+1
                self.radialLinesX[i,:] = self.radialLinesY[i,:]/slopes[i] # x = y/m
            elif angle>45:
                self.radialLinesX[i,:] = np.arange(-maxLength,maxLength)+1
                self.radialLinesY[i,:] = slopes[i]*self.radialLinesX[i,:] # y = mx
        self.radialLinesX[angles.size:,:] = self.radialLinesX[1:angles.size,:]
        self.radialLinesY[angles.size:,:] = -self.radialLinesY[1:angles.size,:]
        self.radialProfiles = np.zeros((self.numRadialLines*2,max(self.roiSize)),dtype=np.uint8)
        
    def updatePupilPlot(self):
        if self.pupilFound and (self.reflectCenter is None or self.reflectFound):
            self.pupilCenterPlot.setData(x=[self.pupilCenter[0]],y=[self.pupilCenter[1]])
            angle = self.pupilEllipseAngle*math.pi/180
            sinx = np.sin(np.arange(0,370,10)*math.pi/180)
            cosx = np.cos(np.arange(0,370,10)*math.pi/180)
            self.pupilEllipsePlot.setData(x=self.pupilCenter[0]+self.pupilEllipseHalfLength[0]/2*cosx*math.cos(angle)-self.pupilEllipseHalfLength[1]/2*sinx*math.sin(angle),
                                          y=self.pupilCenter[1]+self.pupilEllipseHalfLength[0]/2*cosx*math.sin(angle)+self.pupilEllipseHalfLength[1]/2*sinx*math.cos(angle))
        else:
            self.pupilCenterPlot.setData(x=[],y=[])
            self.pupilEllipsePlot.setData(x=[],y=[])
            
    def updatePupilData(self):
        if self.cam is None:
            self.dataPlotFrameInd = self.frameNum-1
        else:
            if self.dataPlotFrameInd==self.numDataPlotPts:
                self.dataPlotFrameInd = 0
            leadingPtsInd = np.s_[self.dataPlotFrameInd+1:self.dataPlotFrameInd+math.ceil(self.numDataPlotPts/10)]
            self.pupilArea[leadingPtsInd] = np.nan
            self.pupilX[leadingPtsInd] = np.nan
            self.pupilY[leadingPtsInd] = np.nan
        if self.pupilFound and (self.reflectCenter is None or self.reflectFound):
            self.pupilArea[self.dataPlotFrameInd] = math.pi*self.pupilEllipseHalfLength[1]**2
            if not np.isnan(self.mmPerPixel):
                self.pupilArea[self.dataPlotFrameInd] *= self.mmPerPixel**2
            if self.reflectCenter is None:
                self.pupilX[self.dataPlotFrameInd] = self.pupilCenter[0]
                self.pupilY[self.dataPlotFrameInd] = self.pupilCenter[1] 
            elif np.isnan(self.mmPerPixel) or self.toolsMenuReflectTypeSpot.isChecked():
                self.pupilX[self.dataPlotFrameInd] = self.pupilCenter[0]-self.reflectCenter[0]
                self.pupilY[self.dataPlotFrameInd] = self.pupilCenter[1]-self.reflectCenter[1]
            else:
                pupilRadius = (self.lensRotRadius**2-(self.pupilEllipseHalfLength[1]*self.mmPerPixel)**2)**0.5-self.lensOffset
                self.pupilX[self.dataPlotFrameInd],self.pupilY[self.dataPlotFrameInd] = [180/math.pi*math.asin((pupilRadius*self.mmPerPixel*(self.reflectCenter[i]-self.pupilCenter[i])/(pupilRadius-self.corneaOffset))/pupilRadius) for i in (0,1)]
        else:
            self.pupilArea[self.dataPlotFrameInd] = np.nan
            self.pupilX[self.dataPlotFrameInd] = np.nan
            self.pupilY[self.dataPlotFrameInd] = np.nan
        
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
        connectPts = np.logical_not(np.isnan(self.pupilArea[dataPlotInd])).astype(np.uint32)
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
        self.pupilEdgePtsPlot.setData(self.pupilEdges)
        xmax = 0
        for i in range(len(self.radialProfilePlot)):
            xmax = max([xmax,np.where(self.radialProfiles[i])[0][-1]])
            self.radialProfilePlot[i].setData(self.radialProfiles[i,:])
            self.radialProfilePixAboveThreshPlot[i].setData(np.correlate(self.radialProfiles[i,:]>self.pupilEdgeThresh,np.ones(self.minNumPixAboveThresh)))
        xTickSpacing = self.getTickSpacing(xmax)
        self.pupilAreaPlotItem.setRange(xRange=(0,xmax),yRange=(0,2*self.pupilEdgeThresh))
        self.pupilAreaPlotItem.getAxis('left').setTickSpacing(levels=[(self.getTickSpacing(self.pupilEdgeThresh*2),0)])
        self.pupilAreaPlotItem.getAxis('bottom').setTickSpacing(levels=[(xTickSpacing,0)])
        self.pupilXPlotItem.setRange(xRange=(0,xmax),yRange=(0,2*self.minNumPixAboveThresh))
        self.pupilXPlotItem.getAxis('left').setTickSpacing(levels=[(round(self.minNumPixAboveThresh/2),0)])
        self.pupilXPlotItem.getAxis('bottom').setTickSpacing(levels=[(xTickSpacing,0)])
        if self.pupilEdges.shape[0]>0:
            self.edgeDistPlot.setData(x=np.arange(self.pupilEdgeDist.size)+1,y=self.pupilEdgeDist)
            self.edgeDistUpperThreshLine.setValue(self.pupilEdgeDist.mean()+self.edgeDistThreshOffset+self.edgeDistThreshFactor*self.pupilEdgeDist.std())
            self.edgeDistLowerThreshLine.setValue(self.pupilEdgeDist.mean()+self.edgeDistThreshOffset-self.edgeDistThreshFactor*self.pupilEdgeDist.std())
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
        self.edgeDistThreshOffset = lowerThresh+(upperThresh-lowerThresh)/2-meanEdgeDist
        self.edgeDistThreshFactor = (upperThresh-lowerThresh)/2/self.pupilEdgeDist.std()
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
            if self.toolsMenuReflectTypeSpot.isChecked() or len(self.reflectRoi)==4:
                if self.toolsMenuReflectTypeRing.isChecked():
                    self.getReflectTemplate()
                    if not self.reflectFound:
                        return
                self.trackReflect()
                if self.reflectFound:
                    self.reflectCenterPlot.setData(x=[self.reflectCenter[0]],y=[self.reflectCenter[1]])
                    if self.pupilCenter is not None:
                        self.updatePupilData()
                        self.updatePupilDataPlot()
            
    def trackReflect(self):
        roiPos,roiSize = self.reflectRoiPos[0],self.reflectRoiSize[0]
        if self.toolsMenuReflectTypeSpot.isChecked():
            y,x = np.where(self.image[self.roiInd][roiPos[1]:roiPos[1]+roiSize[1],roiPos[0]:roiPos[0]+roiSize[0]]>self.reflectThresh)
            if any(y):
                self.reflectCenter = (roiPos[0]+x.mean(),roiPos[1]+y.mean())
            else:
                self.reflectFound = False
                return
        else:
            y,x = np.unravel_index(np.argmax(signal.fftconvolve(self.image[self.roiInd][roiPos[1]:roiPos[1]+roiSize[1],roiPos[0]:roiPos[0]+roiSize[0]],self.reflectTemplate,mode='same')),roiSize)          
            center = (roiPos[0]+x,roiPos[1]+y)
            if any((center[i]-roiSize[i]<0 or center[i]+roiSize[i]>self.roiSize[i]-1 for i in [0,1])):
                self.reflectFound = False
                return
            self.reflectCenter = center
        self.reflectRoiPos = [[int(self.reflectCenter[0]-roiSize[0]/2),int(self.reflectCenter[1]-roiSize[1]/2)]]
        self.reflectFound = True
        
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
        self.reflectCenter = spotCenters.mean(axis=0)
        roiSize = 4*int(max(spotCenters.max(axis=0)-spotCenters.min(axis=0)))
        self.reflectRoiSize = [[roiSize]*2]
        self.reflectTemplate = np.zeros((roiSize,)*2,dtype=bool)
        self.reflectRoiPos = [[int(self.reflectCenter[0]-roiSize/2),int(self.reflectCenter[1]-roiSize/2)]]
        spotCenters = (spotCenters-(self.reflectCenter-roiSize/2)).astype(int)
        m,n = int(ptsAboveThresh/2),int(round(ptsAboveThresh/2))
        for center in spotCenters:     
            self.reflectTemplate[center[1]-m:center[1]+n,center[0]-m:center[0]+n] = True
            
    def setReflectType(self):
        if self.mainWin.sender() is self.toolsMenuReflectTypeSpot:
            self.toolsMenuReflectTypeSpot.setChecked(True)
            self.toolsMenuReflectTypeRing.setChecked(False)
        else:
            self.toolsMenuReflectTypeSpot.setChecked(False)
            self.toolsMenuReflectTypeRing.setChecked(True)
        if len(self.reflectRoi)>0:
            for roi in self.reflectRoi:
                self.imageViewBox.removeItem(roi)
            self.reflectRoi = []
            self.reflectCenter = None
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
        self.plotDurEdit.setText(str(round(newVal,2)))
        self.dataPlotDur = newVal
        if self.cam is not None:
            self.resetPupilData()
        self.setDataPlotTime()
        self.setDataPlotXRange()
        if all(np.isnan(self.pupilArea)):
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
        if self.reflectCenter is None:
            QtGui.QMessageBox.about(self.mainWin,'Set mm/pixel','First find reflection')
        else:
            initialReflectCenter = self.reflectCenter
            QtGui.QMessageBox.about(self.mainWin,'Set mm/pixel','Move camera 0.5 mm; then press ok')
            self.mmPerPixel = 0.5/(((self.reflectCenter[0]-initialReflectCenter[0])**2+(self.reflectCenter[1]-initialReflectCenter[1])**2)**0.5)
            
    def analyzeAllFrames(self):
        while self.frameNum<self.numFrames:
            self.frameNum += 1
            self.getVideoImage()
            if self.frameNum==self.numFrames:
                self.updateDisplay(updateAll=True)
                self.changePlotWindowDur(fullRange=True)
            else:
                self.updateDisplay(updateNone=True)
                
    def loadAnalyzedData(self):
        filePath = QtGui.QFileDialog.getOpenFileName(self.mainWin,'Choose File',self.fileOpenPath,'*.hdf5')
        if filePath=='':
            return
        self.fileOpenPath = os.path.dirname(filePath)
        dataFile = h5py.File(filePath,'r')
        for param in ('pupilArea','pupilX','pupilY'):
            setattr(self,param,dataFile[param][:])
        if self.dataFileIn is not None:
            self.frameTimes = dataFile['frameTimes'][:]
        self.dataIsLoaded = True
        self.pupilAreaRange = [self.pupilArea[self.frameNum-1]]*2
        self.pupilXRange = [self.pupilX[self.frameNum-1]]*2
        self.pupilYRange = [self.pupilY[self.frameNum-1]]*2
        self.setDataPlotXRange()
        self.updatePupilDataPlot()
    
    def plotFrameIntervals(self):
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
        
        
class SignalGenerator(QtCore.QObject):

    camFrameCapturedSignal = QtCore.pyqtSignal(np.ndarray,float)
    
    def __init__(self):
        QtCore.QObject.__init__(self)
        

if __name__=="__main__":
    start()