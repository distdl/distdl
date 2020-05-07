========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/distdl/badge/?style=flat
    :target: https://readthedocs.org/projects/distdl
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/distdl/distdl.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/distdl/distdl

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

.. |wheel| image:: https://img.shields.io/pypi/wheel/distdl.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/distdl

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/distdl.svg
    :alt: Supported versions
    :target: https://pypi.org/project/distdl

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/distdl.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/distdl

.. |commits-since| image:: https://img.shields.io/github/commits-since/distdl/distdl/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/distdl/distdl/compare/v0.0.0...master



.. end-badges

A Distributed Deep Learning package for PyTorch.

* Free software: BSD 2-Clause License

Installation
============

::

    pip install distdl

You can also install the in-development version with::

    pip install https://github.com/distdl/distdl/archive/master.zip


Documentation
=============


https://distdl.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
