from PyQt5 import QtWidgets as Qw   
import guigenerator as gui
import sys

####----------------------------------------
####   Execute this file to open GUI
####----------------------------------------
#   main
if __name__ == '__main__':

    app = Qw.QApplication(sys.argv)         
    wmain = gui.MyForm()                        
    wmain.show()                            
    sys.exit(app.exec_())