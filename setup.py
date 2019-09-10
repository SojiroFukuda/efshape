from setuptools import setup,find_packages

setup(
    name="efshapegui",
    version="1.1.0",
    packages=find_packages(),
    install_requires=["PyQt5", "matplotlib","numpy","pandas","scipy","sklearn"],
    author="Sojiro Fukuda",
    author_email="S.Fukuda-2018@hull.ac.uk",
    entry_points={
        "console_scripts": [
            "efshape = main"
        ],
        "gui_scripts": [
            "efshapegui = main"
        ]
    }
)
