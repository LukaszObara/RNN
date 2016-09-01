# Change Log
All notable changes to RecurrentNetworkEntropy.py project will be documented in this file.

## [Unreleased] - 01/09/2016
### Added
- Ability to evaluate cost and accuracy
- exponential decay on update rate

### Fixed
- removed second instance of feedforward() from sgd()


## [Unreleased] - 31/08/2016
### Added
- Added ability to alter spectral radius of hidden weights
- Added cross-entropy loss function

### Fixed
- \nabla_h L computation step by remove the iterative addition 
