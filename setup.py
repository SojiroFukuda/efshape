from setuptools import setup, find_packages

setup(
    name="fps_pca_gui_analyzer",
    version="0.0.1",
    description='package for bulding the GUI software for grain shape analysis based on elliptic Fourier analysis',
    author='Sojiro Fukuda',
    author_email='s.fukuda-2018@hull.ac.uk',
    url="https://github.com/SojiroFukuda/FPS-PCA-GUI",
    license='Sojiro FUkuda'
    install_requires=['numpy','PyQt5','matplotlib','os','re','glob','pandas','pathlib','datatime','cv2','scipy','sklearn'],
    packages=find_packages(exclude=('tests', 'docs'))
)
