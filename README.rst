========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |ci| |codecov|
    * - package
      - | |version| |supported-implementations|

.. |docs| image:: https://readthedocs.org/projects/distdl/badge/?style=flat
    :target: https://readthedocs.org/projects/distdl
    :alt: Documentation Status

.. |ci| image:: https://github.com/distdl/distdl/workflows/package%20tests/badge.svg
    :alt: DistDL Github Actions build status
    :target: https://github.com/distdl/distdl/actions

.. |travis| image:: https://api.travis-ci.com/distdl/distdl.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/distdl/distdl

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/distdl/distdl?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/distdl/distdl

.. |requires| image:: https://requires.io/github/distdl/distdl/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/distdl/distdl/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/distdl/distdl/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/distdl/distdl

.. |version| image:: https://img.shields.io/pypi/v/distdl.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/distdl

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/distdl.svg
    :alt: Supported versions
    :target: https://pypi.org/project/distdl

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/distdl.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/distdl



.. end-badges

A Distributed Deep Learning package for PyTorch.

* Free software: BSD 2-Clause License

Installation
============

You can install the in-development version with:

    pip install https://github.com/distdl/distdl/archive/master.zip

DistDL will be available on pip shortly.


Documentation
=============


https://distdl.readthedocs.io/


Development
===========

To run the all tests run::

    mpirun -np 20 python -m mpi4py -m pytest --with-mpi -rsa tests

Substitute ``mpiexec`` or ``srun`` as correct for your system.

.. Note, to combine the coverage data from all the tox environments run:

.. .. list-table::
..     :widths: 10 90
..     :stub-columns: 1

..     - - Windows
..       - ::

..             set PYTEST_ADDOPTS=--cov-append
..             tox

..     - - Other
..       - ::

..             PYTEST_ADDOPTS=--cov-append tox
