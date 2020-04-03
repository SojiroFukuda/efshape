# ef-shape

efshape is a python package for shape analysis of 2D image.
The method is based on the combination between 'Elliptic Fourier Analysis (EFA)' and 'Principal Component Analysis (PCA)' and called EF-PCA method.
The basic idea is to convert the 2D closed contour into a numeric dataset by EFA and then, using multivariate analysis (PCA) , detect the major shape variation of the dataset. 
EFA enables you to describe the complicated shape complehensively as a number of dataset called elliptic Fourire descriptors.
One of the merits of this method is that you can easily reconstruct the shape from the descriptors, which also make it easier to interpret the shape variation of detected principal component axes.

In this package, you can choose three different types of EF-PCA methods: one of which, EFD-based EF-PCA, is suitable for directional object like bio-forms and the others, Amplitude- and FPS-based EF-PCA, are suitable for non-directional object such as sedimentary grain.
This package provides you both CUI- and GUI-based package.

# About license
© 2019 Sojiro Fukuda All Rightss Reserved.
Free to modify and redistribute by your own responsibility.