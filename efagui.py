from PyQt5 import QtWidgets as Qw   
import guigenerator as gui
import sys


def startGUI():
	app = Qw.QApplication(sys.argv)         
    wmain = gui.MyForm()                        
    wmain.show()                            
    sys.exit(app.exec_())

