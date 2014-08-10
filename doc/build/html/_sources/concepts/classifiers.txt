Classifiers
===========

Prediction
**********

Since a standard classifier output can't be assumed, a classifier always outputs a list with:

::

    [ predicted_label, generic_classifier_output ]


Take the k-Nearest Neighbor for example. Imagine I have a 3-Nearest Neighbor classifier, then your PredictableModel is going to return something similar to:

.. code-block:: python

    >>> model.predict(X)
    [ 0, 
      { 'labels'    : [ 0,      0,      1      ],
        'distances' : [ 10.132, 10.341, 13.314 ]
      }
    ]


In this example the predicted label is ``0``, because two of three nearest neighbors were of label ``0`` and only one neighbor was ``1``. The generic output is given in a dict for these classifiers, so you are given some semantic information. I prefer this over plain Python lists, because it is probably hard to read through some code, if you are accessing stuff by indices only.

If you only want to know the predicted label for a query image ``X`` you would write:

.. code-block:: python

    predicted_label = model.predict(X)[0]

And if you want to make your `PredictableModel` more sophisticated, by rejecting examples based on the classifier output for example, then you'll need to access the generic classifier output:

.. code-block:: python

    prediction = model.predict(X)
    predicted_label = prediction[0]
    generic_classifier_output = prediction[1]

You have to read up the classifier output in the help section of each classifers predict method.
    
References
**********

.. [TanTriggs]  Tan, X., and Triggs, B. *"Enhanced local texture feature sets for face recognition under difficult lighting conditions."*. IEEE Transactions on Image Processing 19 (2010), 1635–650.