# facerec/py #

## Contents ##

* [Introduction](#introduction)
* [Installation](#installation)
	* [setup.py](#setup_py)
	* [Dependencies](#dependencies)
	* 
## <a name="introduction"></a>Introduction ##

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

## <a name="installation"></a>Installation ##

### <a name="setup_py"></a>setup.py ###

[facerec](https://github.com/bytefish/facerec) comes with setup.py script in the ``py`` folder, so installing it is as simple as running:

```
python setup.py install
```

### <a name="dependencies"></a>Dependencies ###

[facerec](https://github.com/bytefish/facerec) has dependencies to:

* [PIL](http://www.pythonware.com/products/pil/)
* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [matplotlib](http://matplotlib.org/)

All of these packages can be obtained with [pip](http://pip.readthedocs.org/en/latest/)

My current Windows setup uses PIL 1.1.7, NumPy 1.8.1, SciPy 0.14.0, matplotlib 1.2.0.