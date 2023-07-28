---
title: 'measpy: a Python package for data acquisition and signal processing'
tags:
  - Python
  - vibrations
  - acoustics
  - data acquisition
  - signal processing
authors:
  - name: Olivier Doaré
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Caroline Pascal
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2"
  - name: Clément Savaro
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: ENSTA Paris, Unité de Mécanique
   index: 1
 - name: ENSTA Paris, U2IS
   index: 2
date: 1 August 2023
bibliography: paper.bib

---

# Summary

The `measpy` package is a Python library for signal processing and data acquisition. It provides a set of classes and methods that allow users to quickly and easily perform signal processing tasks, such as data acquisition, filtering, transfer function calculation, correlations, spectrogram plots, etc. `measpy` is written in a functional programming style, which makes it easy to write concise and efficient code.

At the core of the package, there is the Signal class, which describes a physical sampled time series as a 1D numpy array, a sampling frequency, a physical unit, thanks to the `unyt` package, and eventual calibration informations, plus any user properties. The spectral class describe signals in the spectral domain. Many implemented signal processing methods of these two base classes return either a Signal or a Spectral object, in a functionnal programming style. Most basic signal processing methods consist in an encapsulation of signal processing functions of the `scipy` package.

Additionnally `measpy` implements the Measurement class, which defines a digital acquitition process (inputs, outputs, units, calibrations) and interfaces with data acquisition libraries supplied by hardware manufacturers.

# Statement of need

In experimental acoustics, vibrations, more generally for dynamical systems analysis, in academic research, teaching or industry, one of the main tasks consists of outputing and recording analog voltage signals using DAQ cards. One has then to manage sampled data as well as sampling frequency, calibration data, measurement units. In Python, the `scipy.signal` package covers most common signal processing tools, whereas DAQ card manufacturers often provide a python package compatible with their drivers.

The `measpy` package is built on top of these packages to provide a unified and quick way to do the data acquisition, data file operations, signal processing and analysis.

To 
Functionnal programmming paradigm

# Usage examples

## Signal processing

The creation of an empty `Signal` object consists 


## Measurement task

# Future developments


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

