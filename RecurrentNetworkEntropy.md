# Change Log
All notable changes to RecurrentNetworkEntropy.py project will be documented in this file.

## [0.8.8]
## Added
- Included option to show final printed text in train().
- Included option to randomize sequence in train().
- Ability to change spectral radius when initializing the network.

## Changed
- Created function for update rule in update() to improve legibility. 
- Moved caches and eps values from update() into \_\_init\_\_().

### Fixed
- Fixed problem where only the last sequence was being processed by the 
  network after being split into smaller sequences.


## [0.8.7]
### Changed
- Changed diag() computation in nabla_h L from 3 lines to one line since
  it is the same as a Hadamard product.


## [Unreleased] - 01/09/2016
### Added
- Ability to evaluate cost and accuracy.
- Exponential decay on update rate.

### Fixed
- removed second instance of feedforward() from sgd().


## [Unreleased] - 31/08/2016
### Added
- Added ability to alter spectral radius of hidden weights
- Added cross-entropy loss function.

### Fixed
- \nabla_h L computation step by removing the iterative addition. 
