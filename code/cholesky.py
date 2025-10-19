from lu import verifCarre


def est_symetrique(A: list[list[float]], tol: float = 1e-9) -> bool:
    """
    Vérifie si une matrice carrée A est symétrique.

    Paramètres
    ----------
    A : list[list[float]]
        Matrice à vérifier (liste de listes).
    tol : float, optionnel
        Tolérance numérique pour comparer les éléments symétriques.

    Retour
    ------
    bool
        True si A est symétrique, False sinon.
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("La matrice doit être carrée.")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


def est_definie_positive(A: list[list[float]]) -> bool:
    """
    Vérifie si une matrice A est définie positive à l'aide de la méthode de Cholesky.

    Paramètres
    ----------
    A : list[list[float]]
        Matrice à tester (liste de listes, carrée et réelle).

    Retour
    ------
    bool
        True si A est définie positive, False sinon.
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("La matrice doit être carrée.")

    # Test via décomposition de Cholesky
    try:
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                somme = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - somme
                    if val <= 0:
                        return False
                    L[i][j] = val ** 0.5
                else:
                    L[i][j] = (A[i][j] - somme) / L[j][j]
        return True
    except ZeroDivisionError:
        return False
    except ValueError:
        return False


def cholesky(A: list[list[float]]) -> list[list[float]]:
    """
    Effectue la décomposition de Cholesky d'une matrice symétrique définie positive.

    Paramètres
    ----------
    A : list[list[float]]
        Matrice à décomposer (carrée, symétrique et définie positive).

    Retour
    ------
    list[list[float]]
        Matrice triangulaire inférieure L telle que A = L * L^T.

    Exceptions
    ----------
    ValueError :
        Si la matrice n'est pas carrée, symétrique ou définie positive.
    """
    if not verifCarre(A):
        raise ValueError("La matrice doit être carrée.")
    if not est_symetrique(A):
        raise ValueError("La matrice doit être symétrique.")
    if not est_definie_positive(A):
        raise ValueError("La matrice n'est pas définie positive (Cholesky impossible).")

    n = len(A)
    L = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            somme = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                val = A[i][i] - somme
                if val <= 0:
                    raise ValueError("La matrice n'est pas définie positive.")
                L[i][j] = val ** 0.5
            else:
                L[i][j] = (A[i][j] - somme) / L[j][j]

    return L


test = [[1, 2, 6, 8], [2, 5, 15, 23], [6, 15, 46, 73], [8, 23, 73, 130]]
print(cholesky(test))
