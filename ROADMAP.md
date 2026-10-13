# Feuille de route

### v0.2.0 — Réparer et poser les fondations

Bugs de 0.1.1 à corriger, chacun avec un test de régression :
1. Après `pip install`, `from numanalysis import cholesky` lève `ModuleNotFoundError: No module named 'lu'` (import absolu `from lu import verifCarre`).
2. Du code s'exécute à l'import : `print` dans `__init__.py`, test et `print` en bas de `cholesky.py`.
3. Pas de pivotage : `[[0, 1], [1, 0]]` lève `ZeroDivisionError` ; pour `[[1e-20, 1], [1, 1]]`, L·U redonne `[[1e-20, 1], [1, 0]]`.
4. `est_definie_positive([[1, 100], [0, 1]])` renvoie `True` alors que xᵀAx = −98 pour x = (1, −1) : seul le triangle inférieur est lu.
5. Calculs en double : `llu_unit` et `ulu_unit` refont chacune la factorisation ; `cholesky` factorise deux fois.
6. Types incohérents : `ulu_unit([[4, 3], [6, 3]])` renvoie `[[4, 3], [0.0, -1.5]]` ; les doctests échouent.
7. Métadonnées fausses : `Requires-Python >=2.7`, classifier Python 2, description « Package for LU decomp ».

- [ ] `.gitignore` Python complet ; structure `src/numanalysis/` (avec un sous-paquet `linalg/`) via `git mv` ; `readme.md` → `README.md`.
- [ ] `pyproject.toml` (backend hatchling) : `requires-python = ">=3.11"`, classifiers Python 3, description, licence MIT, URLs (dépôt, doc, issues), mots-clés, extras ; version unique dans `src/numanalysis/__init__.py`.
- [ ] Import relatif corrigé ; plus aucun code exécuté à l'import.
- [ ] Nouvelle API : `is_square`, `is_symmetric`, `is_positive_definite` (qui vérifie aussi la symétrie), `lu` (une seule factorisation, pivot partiel, renvoie `P, L, U` avec PA = LU ; documenter la différence avec `scipy.linalg.lu`, qui renvoie P tel que A = PLU), `cholesky` (une seule factorisation, lève `NotPositiveDefiniteError`).
- [ ] Exceptions dédiées, validation des entrées, types de sortie cohérents.
- [ ] Tests : régression des 7 bugs, propriétés `hypothesis` (PA = LU, LLᵀ = A), SciPy comme oracle, doctests ; couverture ≥ 90 %.
- [ ] Outillage : ruff, mypy, pre-commit, `py.typed`.
- [ ] Workflows `ci.yml` et `publish.yml`, plus mes réglages manuels.
- [ ] README en anglais et `README.fr.md` avec badges (CI, PyPI, couverture, licence) ; `CHANGELOG.md` au format Keep a Changelog, avec une section « Migration depuis 0.1.x » (anciens noms → nouveaux) ; `CONTRIBUTING.md`.

### v0.3.0 — Le cœur « boîte de verre »
- [ ] `_scalar.py` : `float`, `Fraction`, `Decimal`, `complex` et mpmath (extra `mp`) dans tous les algorithmes existants.
- [ ] Solveurs directs : substitutions avant et arrière, `solve`, `det` (produit des pivots × signe de la permutation), `inv`, pivot total (`pivot="complete"`), `ldlt` (sans racine carrée, donc exact en `Fraction`), algorithme de Thomas (`solve_tridiagonal`).
- [ ] Conditionnement : normes 1, ∞ et de Frobenius, `cond` ; générateurs `hilbert`, `vandermonde`, `random_spd` avec option `exact=True`.
- [ ] Mode `steps=True` pour les méthodes directes : pivot choisi, permutations, opérations du type « L₂ ← L₂ − 2·L₁ », matrices intermédiaires ; rendu texte, Markdown et LaTeX (`_repr_latex_` pour Jupyter), en français ou en anglais (`lang="fr"`).
- [ ] Générateur d'exercices : matrices entières dont la LU, la Cholesky et la solution « tombent juste », avec énoncé et corrigé détaillé en LaTeX.
- [ ] Documentation Sphinx (thème Furo, myst-nb, docstrings NumPy) publiée sur GitHub Pages. Notebook phare : Hilbert en `float` contre `Fraction` (l'erreur en flottants passe d'environ 1e-10 à n = 6 à des dizaines vers n = 13, alors que le calcul en fractions est exact).

### v0.4.0 — QR, moindres carrés, valeurs propres
- [ ] Gram-Schmidt classique et modifié, Householder, Givens ; moindres carrés par équations normales et par QR ; démo de la perte d'orthogonalité.
- [ ] Puissance itérée, puissance inverse avec shift, itération du quotient de Rayleigh, réduction de Hessenberg, algorithme QR avec shifts (cas symétrique d'abord), disques de Gershgorin.
- [ ] SVD pédagogique (Jacobi unilatéral) et conditionnement en norme 2.
- [ ] Visualisations (extra `plot`) : disques de Gershgorin, courbes de convergence.

### v0.5.0 — Méthodes itératives et matrices creuses
- [ ] Format CSR codé à la main (construction COO → CSR, produit matrice-vecteur, transposée) ; assembleur du laplacien 1D et 2D (stencil à 5 points).
- [ ] Jacobi, Gauss-Seidel, SOR (ω optimal théorique pour Poisson), gradient conjugué, gradient conjugué préconditionné (Jacobi, SSOR), GMRES redémarré.
- [ ] Objet résultat commun : solution, historique des résidus, nombre d'itérations, statut de convergence.
- [ ] Notebook : comparaison des méthodes sur Poisson 2D, effet de ω.

### v0.6.0 — Équations non linéaires, interpolation, quadrature
- [ ] Dichotomie, point fixe, Newton, sécante, Newton multidimensionnel (avec la LU de la bibliothèque), Broyden ; historique et ordre de convergence estimé.
- [ ] Lagrange, différences divisées, nœuds de Chebyshev, splines cubiques (via Thomas).
- [ ] Trapèzes, Simpson, Romberg, Gauss-Legendre (nœuds par Golub-Welsch avec le module de valeurs propres), Simpson adaptative.
- [ ] Démos : phénomène de Runge, bassins d'attraction de Newton dans le plan complexe.

### v0.7.0 — Équations différentielles ordinaires
- [ ] Euler explicite et implicite, Heun, RK4, Dormand-Prince adaptatif, BDF2, Störmer-Verlet.
- [ ] Interface commune inspirée de `solve_ivp` ; tracé des régions de stabilité.
- [ ] Test automatique de l'ordre de convergence de chaque schéma.
- [ ] Démos : énergie d'un pendule (Euler contre Verlet) ; couche limite de Blasius par méthode de tir (vérifier f''(0) ≈ 0,332).

### v0.8.0 — EDP et vérification façon CFD
- [ ] Différences finies : Poisson 1D et 2D, chaleur (θ-schéma, Crank-Nicolson), advection (upwind, Lax-Wendroff) avec contrôle CFL, Burgers.
- [ ] Volumes finis 1D : Godunov et HLL pour Burgers et Euler ; tube à choc de Sod comparé à la solution exacte du problème de Riemann.
- [ ] Multigrille géométrique (V-cycle) pour Poisson.
- [ ] Solutions manufacturées : des tests CI échouent si un schéma perd son ordre théorique.

### v0.9.0 — Continuation et galerie mécanique/aéro
- [ ] Continuation par paramètre naturel et par pseudo-longueur d'arc ; détection des points limites par le signe de det(K_T), calculé avec la LU.
- [ ] Treillis de von Mises (snap-through) : échec de Newton piloté en force, succès de la continuation.
- [ ] Méthode des panneaux sur un profil NACA 4 chiffres, Cl comparé à 2πα.
- [ ] Cavité entraînée (vorticité–fonction de courant, Re = 100, maillage modéré) comparée aux données de Ghia, Ghia & Shin (1982).
- [ ] Labo HPC (notebook ; Numba n'est requis que par ce notebook) : la même LU en pur Python, NumPy vectorisé, Numba, par blocs et LAPACK, avec benchmarks et comptage des flops (~2n³/3).

### v1.0.0 — Stabilisation
- [ ] Revue complète de l'API (noms, signatures, exceptions), politique de dépréciation documentée, API figée.
- [ ] Documentation complète, tutoriels FR et EN, galerie, démo JupyterLite.
- [ ] `CITATION.cff`, archivage Zenodo avec DOI (réglage manuel de mon côté), `paper.md` et `paper.bib` pour le Journal of Open Source Education (~1 000 mots, « statement of need », usage en enseignement).
