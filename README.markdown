# facerec #

## Overview ##

This project implements a face recognition framework for Python with:

* Preprocessing
    * Histogram Equalization
    * Local Binary Patterns
    * TanTriggsPreprocessing (Tan, X., and Triggs, B. "Enhanced local texture feature sets for face recognition under difficult lighting conditions.". IEEE Transactions on Image Processing 19 (2010), 1635–650.)
* Feature Extraction
    * Eigenfaces (Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of Cognitive Neuroscience 3 (1991), 71–86.)
    * Fisherfaces (Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection.". IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997), 711–720.)
    * Local Binary Patterns Histograms (Ahonen, T., Hadid, A., and Pietikainen, M. "Face Recognition with Local Binary Patterns.". Computer Vision - ECCV 2004 (2004), 469–481.)
        * Original LBP
        * Extended LBP
    * Local Phase Quantization (Ojansivu V & Heikkilä J. "Blur insensitive texture classification using local phase quantization." Proc. Image and Signal Processing (ICISP 2008), 5099:236-243.)
* Classifier
    * k-Nearest Neighbor; available distance metrics
        * Euclidean Distance
        * Cosine Distance
        * ChiSquare Distance
        * Bin Ratio Distance
    * Support Vector Machines; using libsvm bindings. (Vapnik, V. "Statistical Learning Theory.". John Wiley and Sons, New York, 1998.)
* Cross Validation
    * k-fold Cross Validation
    * Leave-One-Out Cross Validation
    * Leave-One-Class-Out Cross Validation

## Documentation ##

You can find the documentation in the `doc` folder coming with this project. I have added the build html folder to 
the repository, so you don't need to build it by yourself. But if you want to build the documentation by yourself,
here is how to do it.

### Sphinx ###

You need to install [Sphinx](http://sphinx-doc.org) in order to build the project, which can be obtained with `pip`:

```
pip install sphinx
```

### Build the Documentation ###

Windows:

```
make.cmd html
```

Linux:

```
make html
```

Or if you only have `sphinx` installed then simply run:

```
sphinx-build -b html source build
```
