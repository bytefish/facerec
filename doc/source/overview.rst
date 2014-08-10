Overview
========

This project implements a face recognition framework for Python (and MATLAB/GNU Octave) with:

* Preprocessing
    * Histogram Equalization
    * Local Binary Patterns
    * TanTriggsPreprocessing [TT2010]_
* Feature Extraction
    * Eigenfaces [TP1991]_
    * Fisherfaces [BHK1997]_
    * Local Binary Patterns Histograms [AHP2004]_
        * Original LBP
        * Extended LBP
    * Local Phase Quantization [HO2008]_
* Classifier
    * k-Nearest Neighbor; available distance metrics
        * Euclidean Distance
        * Cosine Distance
        * ChiSquare Distance
        * Bin Ratio Distance
    * Support Vector Machines; using libsvm bindings. [Vapnik1998]_
* Cross Validation
    * k-fold Cross Validation
    * Leave-One-Out Cross Validation
    * Leave-One-Class-Out Cross Validation
    
References
**********

.. [TT2010] Tan, X., and Triggs, B. *"Enhanced local texture feature sets for face recognition under difficult lighting conditions."*. IEEE Transactions on Image Processing 19 (2010), 1635–650.
.. [TP1991] Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of Cognitive Neuroscience 3 (1991), 71–86.
.. [BHK1997] Belhumeur, P. N., Hespanha, J., and Kriegman, D. *"Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection."*. IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997), 711–720.
.. [AHP2004] Ahonen, T., Hadid, A., and Pietikainen, M. *"Face Recognition with Local Binary Patterns."*. Computer Vision - ECCV 2004 (2004), 469–481.
.. [HO2008] Ojansivu V & Heikkilä J. *"Blur insensitive texture classification using local phase quantization."* Proc. Image and Signal Processing (ICISP 2008), 5099:236-243.
.. [Vapnik1998] Vapnik, V. *"Statistical Learning Theory."*. John Wiley and Sons, New York, 1998.