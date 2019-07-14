python-based GUI for shape analysis for non-directional particles.
© 2019 Sojiro Fukuda All Rightss Reserved.

About
This is the python-based GUI software for elliptic Fourier and Principal component analysis.
You can analyze binarized images by this software.
To open the GUI, please open the fpsanalyzer.py and build it.
Free to distribute but without any warranty.
The paper of this method is still not published (I am trying hard).

NOTE: The following python packages are required in this GUI software
Python 3.7.3 
pandas ver. 0.24.2
matplotlib ver. 3.0.3
numpy ver. 1.16.3
PyQt5 ver. 5.9.2
OpenCV: 4.1.0

How to buid
There are mainly two way to build this GUI.
First one is the way to use command tool and another is the way using other tools such as Anaconda.
In both cases, firstly, you have to make a clone of this Git ropository in your laptop.
For instance, using command tool, type like bellow

> git clone https://github.com/SojiroFukuda/FPS-PCA-GUI

In the case of the way to use command tools, move to the cloned folda by

> cd FPS-PCA-GUI

and then, execute following command

> python setup.py install

Finally, you can buid GUI by following command. 

> python efshapegui.py



