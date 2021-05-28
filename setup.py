from setuptools import setup, find_packages

from datetime import datetime

now = datetime.now()
VERSION = 'git'+now.strftime("%Y%m%d")
VERSION = '0.0.3'
DESCRIPTION = 'Measurements with Python'
LONG_DESCRIPTION = 'Classes and methods to do data acquisition and processing'

# Setting up
setup(
    name="measpy",
    version=VERSION,
    author="Olivier Doaré",
    author_email="<olivier.doare@ensta-paris.fr>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy','matplotlib','unyt','csaps'],
    keywords=['Python', 'Measurements', 'Data acquisition', 'Signal processing'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3"
        ]
)
