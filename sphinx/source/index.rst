.. pyscf-ita documentation master file, created by
   sphinx-quickstart on Wed Jul 19 00:23:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PYSCF-ITA
#########

The [information-theoretic quantities](../developer/ita.ipynb) developed by Shubin Liu et al. is 
becoming more and more popular in predicting and understanding many chemical relevant problems, 
such as reactivity, regioselectivity, aromaticity, pKa and so on. See Acta Phys. -Chim. Sin., 32, 
98 (2016) and WIREs Comput Mol Sci., e1461 (2019) for reviews.

**PYSCF-ITA** is a Python extension library of PyScf for post-processing molecular quantum chemistry 
calculations. So, like any other library, it can be directly imported and used in Python scripts 
and codes. All the information-theoretic quantities can be easily calculated by pyscf-ita, which 
can be freely download at https://zhaoyilin.github.io/pyscf-ita/. This document will briefly illustrate how 
to calculate various information-theoretic and related quantities. 

Contents
========

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: USER DOCUMENTATION:

   user/installation
   user/quickstart
   user/examples

.. toctree::
   :maxdepth: 10
   :numbered:
   :caption: DEVELOPER DOCUMENTATION:

   developer/aim
   developer/ita
   developer/api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
