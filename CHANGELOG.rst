
Changelog
=========

0.0.0 (2020-05-07)
------------------

* Package initialized.


0.0.1 (2020-06-09)
------------------

* Initial release.


0.2.0 (2020-08-18)
------------------

* Dramatically improved documentation.
* Added channel-distributed convolutional layer.
* Abstracted convolutional layer interface.  It now auto-selects
  implementation.
* Added pre-forward hooks so that communication buffers are only allocated
  when the shape of the input tensor changes.
* Improved general consistency of layer structure and member names.

0.3.0 (2020-12-01)
------------------
* Corrected use of dtype in internal buffers.
* Cleaned up partition API.
* Fixed a bug where MPI resources were not released.
* Removed assumption that transpose requires load-balanced input.
* Added smarter buffer re-use.
* Added distributed batch normalization layer.
* Added distributed upsampling interpolation layer.