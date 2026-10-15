Linear algebra
==============

All these names are available from ``numanalysis`` and from ``numanalysis.linalg``.

Factorizations
--------------

.. autofunction:: numanalysis.lu
.. autofunction:: numanalysis.cholesky
.. autofunction:: numanalysis.ldlt

.. autoclass:: numanalysis.LUResult
.. autoclass:: numanalysis.CholeskyResult
.. autoclass:: numanalysis.LDLResult

Linear systems
--------------

.. autofunction:: numanalysis.solve
.. autofunction:: numanalysis.solve_tridiagonal
.. autofunction:: numanalysis.forward_substitution
.. autofunction:: numanalysis.back_substitution
.. autofunction:: numanalysis.det
.. autofunction:: numanalysis.inv

.. autoclass:: numanalysis.Traced

Norms and conditioning
----------------------

.. autofunction:: numanalysis.norm
.. autofunction:: numanalysis.cond

Properties
----------

.. autofunction:: numanalysis.is_square
.. autofunction:: numanalysis.is_symmetric
.. autofunction:: numanalysis.is_hermitian
.. autofunction:: numanalysis.is_positive_definite

Test matrices
-------------

.. autofunction:: numanalysis.hilbert
.. autofunction:: numanalysis.vandermonde
.. autofunction:: numanalysis.random_spd
