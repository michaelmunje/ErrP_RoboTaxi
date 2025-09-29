# import cnbiloop
import sys
import time
import os
import numpy as np
import math
from math import *
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget
from python_client import Trigger

class visualInterface(QWidget):
    def __init__(self):
        super(visualInterface, self).__init__()
        self.initialization()

    def initialization(self):
        """Put configuration setting here"""

        "screen"
        self.screenWidth = QDesktopWidget().availableGeometry().width()
        self.screenHeight = QDesktopWidget().availableGeometry().height()
        self.screenHalf = QtCore.QPoint(self.screenWidth/2, self.screenHeight/2) 
        self.setGeometry(0, 0, self.screenWidth, self.screenHeight)
        
        "text"
        self.fontColor = QtGui.QColor(255, 255, 255)
        self.font = QtGui.QFont()
        self.fontSize = 60
        self.font.setPixelSize(self.fontSize)
        self.textWindowWidth = 2000
        self.textWindowHeight = 300

        "window"
        self.setWindowTitle('Calibration')
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setFocus()

        "time"
        self.recordingLength = 90
        self.timestamp = QtCore.QTime()
        self.timestamp.start()
        
        "serial port (should be commented out during the development)"
        self.isParallel = True
        if (self.isParallel is True):
            self.parallel = Trigger('ARDUINO')
            self.parallel.init(50)
            self.TriggerSend(1)

        QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.flush()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.UpdateLoop)
        self.paintCycleTime = 20  # in [ms]
        timer.start(self.paintCycleTime)

        self.show()

    def UpdateLoop(self):
        self.update()

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.PaintInterface(qp)
        qp.end()

    def PaintInterface(self, qp):
        """Main Drawing Function"""
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)

        "fill background"
        qp.fillRect(0, 0, self.screenWidth, self.screenHeight, QtGui.QColor(0, 0, 0))

        "show comments on window"
        qp.setPen(self.fontColor)
        qp.setFont(self.font)
        secElapsed = round(self.timestamp.elapsed()/1000)
        secLeft = self.recordingLength - secElapsed
        text = ' EOG Calibration : ' + str(secLeft)
        if (60 < secLeft <= 90):
            text = text + '\n\n Repeated Eye Blinks'
        elif (45 < secLeft <= 60):
            text = text + '\n\n Horizontal Eye Movements'
        elif (30 < secLeft <= 45):
            text = text + '\n\n Vertical Eye Movements'
        elif (15 < secLeft <= 30):
            text = text + '\n\n Clockwise Eye Movements'
        elif (0 < secLeft <= 15):
            text = text + '\n\n Counter-Clockwise Eye Movements'

        qp.drawText(QtCore.QRectF(self.screenHalf.x() - self.textWindowWidth/2, self.screenHalf.y()- self.textWindowHeight/2, self.textWindowWidth, self.textWindowHeight), QtCore.Qt.AlignCenter, text)

        timeElapsed = self.timestamp.elapsed()/1000
        if (timeElapsed > self.recordingLength):
            self.TriggerSend(2)
            exit()

    def TriggerSend(self, value):
        if (self.isParallel is True):
            self.parallel.signal(value)

def main():
    app = QApplication(sys.argv)
    ex = visualInterface()
    sys.exit(app.exec_())

if __name__ == '__main__': 
    main()
