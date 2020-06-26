## Fixed Diagonal Matrices 
A python package to calculate dispersion C6 coefficients according to the method presented in [1] and used in [2].

## Requirements:
- Python 3.7 or higher
- Numpy
- Scipy
- Cython
- Optional: pyscf, HORTON 2 gbasis and iodata (for using the pyscf_horton submodule)
- Optional: numba (for using the numba submodule)

## Setup:

If required, install HORTON 2 gbasis and iodata via conda:
- conda install -c theochem gbasis iodata libint=2.0.3

Install pyscf via conda:
- conda install -c pyscf pyscf

If required, get numba via pip or conda.

Then:
- Clone repository
- cd dispersion
- pip install .

Want to use parallel? Then use:
- python setup.py install parallel

## Usage
See the examples folder of the repository for how to use the package.

- example_He.py can be run without arguments and calculates isotropic C6 for two helium atoms with HF, MP2 and CCSD.
- example_H2.py can be run without arguments and calculates isotropic C6 and anisotropies for two H2 molecules with HF, MP2 and CCSD.
- example_xyz.py has as arguments the xyz file of the first system, then xyz file of the second system, charge of first system, spin of first system (note, pyscf convention is 2S, instead of 2S+1, singlet is zero), charge of second system, spin of second system.
- For example: example_xyz.py He.xyz He.xyz 0 0 0 0

## Removal
- pip uninstall .

or manually remove package if installed via setup.py.


## References
1. D.P. Kooi and P. Gori-Giorgi, 2019. A variational approach to London dispersion interactions without density distortion. J. Phys. Chem. Lett., 10(7), pp. 1537-1541.
2. D.P. Kooi, T. Weckman, and P. Gori-Giorgi, 2020. Dispersion without many-body density distortion: Assessment on atoms and small molecules. arXiv preprint arXiv:2002.08708.

## License
Fixed Diagonal Matrices (FDM), a python package to calculate dispersion C6 coefficients.

Copyright (C) 2020  Derk Kooi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License 
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Credits
The Continuous integration pipeline is an adaptation from the **[NLESC-JCER/ci_for_science](https://github.com/NLESC-JCER/ci_for_science)**.
