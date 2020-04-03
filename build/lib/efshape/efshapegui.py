from PyQt5 import QtWidgets as Qw   
import guigenerator as gui
import sys

def startGUI(argv=sys.argv[1:]):
    app = Qw.QApplication(sys.argv)         
    wmain = gui.MyForm()                        
    wmain.show()                            
    sys.exit(app.exec_())
####----------------------------------------
####   Execute this file to open GUI
####----------------------------------------
#   main
if __name__ == '__main__':
    startGUI()
#     app = Qw.QApplication(sys.argv)         
#     wmain = gui.MyForm()                        
#     wmain.show()                            
#     sys.exit(app.exec_())
