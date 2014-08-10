Features
========

Available Feature Extraction Algorithms
***************************************

TODO

facerec.feature
---------------

* Identity
* PCA
* LDA
* Fisherfaces
* SpatialHistogram

facerec.preprocessing
---------------------

* Resize
* HistogramEqualization
* TanTriggsPreprocessing
* LBPPreprocessing (with a ``LocalDescriptor`` from ``facerec.lbp``)
    * ExtendedLBP
    * VarLBP
    * LPQ
* MinMaxNormalizePreprocessing
* ZScoreNormalizePreprocessing

facerec.operators
-----------------

* ChainOperator
* CombineOperator
* CombineOperatorND

Image processing chains
***********************

Sometimes it's also necessary to perform preprocessing on your images. The framework makes it easy to experiment with algorithms. You can achieve image processing chains by using the `ChainOperator <https://github.com/bytefish/facerec/blob/master/py/facerec/operators.py>`_. The ``ChainOperator`` computes a ``feature1`` and passes its output to a ``feature2``. Any Operator coming with the framework basically is an ``AbstractFeature``, so it can be used like any other feature extraction algorithm.

.. code-block:: python

    class FeatureOperator(AbstractFeature):
        """
        A FeatureOperator operates on two feature models.
        
        Args:
            model1 [AbstractFeature]
            model2 [AbstractFeature]
        """
        def __init__(self,model1,model2):
            if (not isinstance(model1,AbstractFeature)) or (not isinstance(model2,AbstractFeature)):
                raise Exception("A FeatureOperator only works on classes implementing an AbstractFeature!")
            self.model1 = model1
            self.model2 = model2
        
        def __repr__(self):
            return "FeatureOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
            
    class ChainOperator(FeatureOperator):
        """
        The ChainOperator chains two feature extraction modules:
            model2.compute(model1.compute(X,y),y)
        Where X can be generic input data.
        
        Args:
            model1 [AbstractFeature]
            model2 [AbstractFeature]
        """
        def __init__(self,model1,model2):
            FeatureOperator.__init__(self,model1,model2)
            
        def compute(self,X,y):
            X = self.model1.compute(X,y)
            return self.model2.compute(X,y)
            
        def extract(self,X):
            X = self.model1.extract(X)
            return self.model2.extract(X)
        
        def __repr__(self):
            return "ChainOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"

Imagine we want to perform a TanTriggs preprocessing, before applying a Fisherfaces algorithm. The TanTriggs Preprocessing is a simple illumination normalization algorithm, which was first proposed in [TanTriggs]_. Now what you would do is using the output of the ``TanTriggsPreprocessing`` feature extraction as input for the ``Fisherfaces`` feature extraction. We can express this with a `ChainOperator` in facerec:

.. code-block:: python

    from facerec.preprocessing import TanTriggsPreprocessing
    from facerec.feature import Fisherfaces
    from facerec.operators import ChainOperator
    from facerec.model import PredictableModel


    feature = ChainOperator(TanTriggsPreprocessing(), Fisherfaces())
    classifier = NearestNeighbor()
    model = PredictableModel(feature, classifier)

    
References
**********

.. [TanTriggs]  Tan, X., and Triggs, B. *"Enhanced local texture feature sets for face recognition under difficult lighting conditions."*. IEEE Transactions on Image Processing 19 (2010), 1635–650.