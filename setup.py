# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import versioneer

DESCRIPTION = 'Measurements with Python'
LONG_DESCRIPTION = """
    measpy is a set of classes and methods to help the data acquisition and analysis of signals. It is mainly acoustics and vibrations oriented. This package is very recent, it is still incomplete and many bugs may appear.

    The base classes are:

    - ```Signal```: It is basically a class that contain a 1D numpy array, an a few other properties to describe the data as: sampling frequency, calibration and unit.
    - ```Spectral```: A spectral data class that contains the complex amplitudes as a 1D numpy array for frequencies up to the Nyquist frequency, and some properties as sampling frequency, unit, description
    - ```Measurement``` : A class that describe a data acquisition process, its outputs (Signal objects), its inputs (Signal objects)...
    - ```Weighting``` : Weighting spectral functions (Not yet fully test/functionnal)

    For now, these daq devices can be used :

    - Audio cards, via the ```sounddevice``` package,
    - NI DAQ cards, via the ```nidaqmx``` package.
    - Picoscope scopes, via the ```picosdk-python-wrappers``` package.
    
    """

# Setting up
setup(
    name="measpy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
