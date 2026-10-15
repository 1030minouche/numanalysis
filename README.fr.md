# numanalysis

[![CI](https://github.com/1030minouche/numanalysis/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/1030minouche/numanalysis/actions/workflows/ci.yml)
[![Documentation](https://github.com/1030minouche/numanalysis/actions/workflows/docs.yml/badge.svg?branch=main)](https://1030minouche.github.io/numanalysis/)
[![PyPI](https://img.shields.io/pypi/v/numanalysis.svg)](https://pypi.org/project/numanalysis/)
[![Versions de Python](https://img.shields.io/pypi/pyversions/numanalysis.svg)](https://pypi.org/project/numanalysis/)
[![Couverture](https://codecov.io/gh/1030minouche/numanalysis/branch/main/graph/badge.svg)](https://codecov.io/gh/1030minouche/numanalysis)
[![Licence : MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](https://github.com/1030minouche/numanalysis/blob/main/LICENSE)

*L'analyse numérique en boîte de verre, en pur Python.* [English version](https://github.com/1030minouche/numanalysis/blob/main/README.md)
· [Documentation](https://1030minouche.github.io/numanalysis/) (en anglais)

`numanalysis` est une bibliothèque d'analyse numérique pensée pour l'enseignement.
Chaque algorithme est écrit pour être lu, fonctionne en arithmétique exacte ou en
multiprécision aussi bien qu'en flottants, montre ses étapes à la demande, et SciPy
sert de référence dans les tests. Elle ne cherche pas à concurrencer NumPy ou SciPy
sur la vitesse.

- **Aucune dépendance** : le cœur n'utilise que la bibliothèque standard.
- **Tous les types de nombres** : `float`, `complex`, `Fraction` (exact), `Decimal` et
  mpmath (précision arbitraire) ; rien n'est converti en `float` dans votre dos.
- **Pas à pas** : `steps=True` montre les pivots, les opérations sur les lignes et les
  matrices intermédiaires, en texte, Markdown ou LaTeX, en français ou en anglais.
- **Exercices** : des matrices dont les factorisations tombent juste, avec énoncé et
  corrigé détaillé en LaTeX.
- **Typée et testée** : annotations de type, `py.typed`, tests par propriétés, couverture à 100 %.

## Installation

```bash
pip install numanalysis
```

Python 3.11 ou plus récent est requis. Extras optionnels : `mp` (mpmath) et `plot`
(matplotlib).

## Prise en main

Factorisation LU avec pivot partiel, `P A = L U` :

```pycon
>>> from numanalysis import lu
>>> P, L, U = lu([[0, 1], [1, 0]])
>>> P
[[0.0, 1.0], [1.0, 0.0]]
>>> U
[[1.0, 0.0], [0.0, 1.0]]

```

La même factorisation en arithmétique exacte :

```pycon
>>> from fractions import Fraction
>>> A = [[Fraction(2), Fraction(1)], [Fraction(4), Fraction(3)]]
>>> res = lu(A)
>>> res.L[1][0], res.U[1][1]
(Fraction(1, 2), Fraction(-1, 2))
>>> res.perm
[1, 0]

```

Résoudre un système et voir comment c'est fait :

```pycon
>>> from numanalysis import solve
>>> x, etapes = solve([[Fraction(2), 1], [1, 3]], [3, 5], steps=True, lang="fr")
>>> x
[Fraction(4, 5), Fraction(7, 5)]
>>> print(etapes.to_text().splitlines()[3])
  L₂ ← L₂ − (1/2)·L₁

```

Factorisation de Cholesky d'une matrice symétrique définie positive, `A = L Lᵀ` :

```pycon
>>> from numanalysis import cholesky, is_positive_definite
>>> cholesky([[4, 2], [2, 5]]).L
[[2.0, 0.0], [1.0, 2.0]]
>>> is_positive_definite([[1, 100], [0, 1]])
False

```

Les erreurs sont explicites et héritent de `ValueError` :

```pycon
>>> cholesky([[1, 2], [2, 1]])
Traceback (most recent call last):
    ...
numanalysis.exceptions.NotPositiveDefiniteError: A is not positive definite: the leading minor of order 2 is not positive

```

> **Remarque.** `scipy.linalg.lu` suit la convention `A = P L U` : sa matrice de
> permutation est la transposée de celle que renvoie `numanalysis.lu`.

## Fonctions disponibles

| Fonction | Description |
| --- | --- |
| `lu(A, pivot=...)` | LU avec pivot partiel ou total (`P A = L U`, `P A Q = L U`) |
| `cholesky(A)` | Factorisation de Cholesky `A = L Lᵀ` (`L Lᴴ` en complexe) |
| `ldlt(A)` | `A = L D Lᵀ` sans racine carrée, exacte en `Fraction` |
| `solve(A, b)` | Résout `A x = b` (second membre vecteur ou matrice) |
| `solve_tridiagonal(...)` | Algorithme de Thomas, en temps linéaire |
| `forward_substitution`, `back_substitution` | Systèmes triangulaires (descente, remontée) |
| `det(A)`, `inv(A)` | Déterminant, inverse |
| `norm(x, ord)`, `cond(A, ord)` | Normes 1, 2 (vecteurs), ∞, Frobenius ; conditionnement |
| `is_square`, `is_symmetric`, `is_hermitian`, `is_positive_definite` | Propriétés |
| `hilbert(n)`, `vandermonde(x)`, `random_spd(n)` | Matrices de test (`exact=True` pour des fractions) |
| `numanalysis.exercises` | Générateur d'exercices avec énoncés et corrigés en LaTeX |

Vous migrez depuis la 0.1.x ? Consultez le
[guide de migration](https://github.com/1030minouche/numanalysis/blob/main/CHANGELOG.md).

## Feuille de route

Sont prévus : QR et moindres carrés, valeurs propres, méthodes itératives et matrices
creuses, équations non linéaires, interpolation et quadrature, EDO, EDP et une galerie
d'exemples de mécanique et d'aéro.

## Contribuer

Les contributions sont les bienvenues : voir
[CONTRIBUTING.md](https://github.com/1030minouche/numanalysis/blob/main/CONTRIBUTING.md).

## Licence

[MIT](https://github.com/1030minouche/numanalysis/blob/main/LICENSE)
