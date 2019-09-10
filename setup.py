from setuptools import setup

setup(
    name="efshapegui",
    version="1.1.0)",
    install_requires=["PyQt5", "matplotlib","numpy","pandas","scipy","sklearn"],
    extras_require={
#         "develop": ["dev-packageA", "dev-packageB"]
    },
    entry_points={
        "console_scripts": [
            "efshape = efshapegui:startGUI"
        ],
        "gui_scripts": [
            "efshapeGUI = efshapegui:startGUI"
        ]
    }
)
