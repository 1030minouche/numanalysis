# numanalysis

[![CI](https://github.com/1030minouche/numanalysis/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/1030minouche/numanalysis/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/numanalysis.svg)](https://pypi.org/project/numanalysis/)
[![Versions de Python](https://img.shields.io/pypi/pyversions/numanalysis.svg)](https://pypi.org/project/numanalysis/)
[![Couverture](https://codecov.io/gh/1030minouche/numanalysis/branch/main/graph/badge.svg)](https://codecov.io/gh/1030minouche/numanalysis)
[![Licence : MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](https://github.com/1030minouche/numanalysis/blob/main/LICENSE)

*L'analyse numérique en boîte de verre, en pur Python.* [English version](https://github.com/1030minouche/numanalysis/blob/main/README.md)

`numanalysis` est une bibliothèque d'analyse numérique pensée pour l'enseignement.
Chaque algorithme est écrit pour être lu et fonctionne aussi bien en arithmétique
exacte (`fractions.Fraction`) qu'en flottants. Dans les tests, SciPy sert de référence.
Elle ne cherche pas à concurrencer NumPy ou SciPy sur la vitesse.

- **Aucune dépendance** : le cœur n'utilise que la bibliothèque standard.
- **Toute entrée matricielle** : listes, tuples ou tableaux NumPy. L'entrée n'est jamais modifiée.
- **Exacte si vous le voulez** : avec des coefficients `Fraction`, les résultats sont exacts.
- **Typée et testée** : annotations de type, `py.typed`, tests par propriétés, couverture à 100 %.

## Installation

```bash
pip install numanalysis
```

Python 3.11 ou plus récent est requis.

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

| Fonction                  | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `lu(A)`                   | LU avec pivot partiel, renvoie `P, L, U` (`P A = L U`)        |
| `cholesky(A)`             | Factorisation de Cholesky, renvoie `.L` (`A = L Lᵀ`)         |
| `is_square(A)`            | La matrice est-elle carrée ?                                  |
| `is_symmetric(A)`         | La matrice est-elle symétrique (avec tolérance en flottants) ? |
| `is_positive_definite(A)` | La matrice est-elle symétrique définie positive ?             |

Vous migrez depuis la 0.1.x ? Consultez le
[guide de migration](https://github.com/1030minouche/numanalysis/blob/main/CHANGELOG.md).

## Feuille de route

Sont prévus : solveurs triangulaires, déterminants, conditionnement, traces pas à pas
avec rendu LaTeX, QR, valeurs propres, méthodes itératives, EDO, EDP et une galerie
d'exemples de mécanique et d'aéro. Voir
[ROADMAP.md](https://github.com/1030minouche/numanalysis/blob/main/ROADMAP.md).

## Contribuer

Les contributions sont les bienvenues : voir
[CONTRIBUTING.md](https://github.com/1030minouche/numanalysis/blob/main/CONTRIBUTING.md).

## Licence

[MIT](https://github.com/1030minouche/numanalysis/blob/main/LICENSE)
