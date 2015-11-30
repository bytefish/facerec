# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

from distutils.core import setup

long_description = '''facerec is a framework for experimenting
with face recognition algorithms. It provides a set of classifiers,
features, algorithms, operators, examples and a basic webservice.'''

setup(name='facerec',
    version='1.0',
    description='facerec framework',
    long_description=long_description,
    author='Philipp Wagner',
    author_email='bytefish@gmx.de',
    url='http://www.github.com/bytefish/facerec',
    license='BSD',
    classifiers= [
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    packages=['facerec'],
    package_dir={'facerec': 'facerec'},
)
