def verifCarre(A: list[list[float]]) -> bool:
    """
    Vérifie si une matrice est carrée.

    Args:
        A (list[list[float]]): Matrice à vérifier.

    Returns:
        bool: True si la matrice est carrée, False sinon.

    Examples:
        >>> verifCarre([[1, 2], [3, 4]])
        True
        >>> verifCarre([[1, 2, 3], [4, 5, 6]])
        False
    """
    n = len(A)
    return all(len(row) == n for row in A)


def llu_unit(A: list[list[float]]) -> list[list[float]]:
    """
    Calcule la matrice L de la décomposition LU d'une matrice carrée A,
    en supposant que la matrice L a une diagonale unité.

    Cette fonction réalise une décomposition LU sans pivot, telle que :
        A = L * U

    où :
    - L est une matrice triangulaire inférieure avec diagonale unité,
    - U est une matrice triangulaire supérieure.

    Args:
        A (list[list[float]]): Matrice carrée à décomposer.

    Returns:
        list[list[float]]: Matrice L de la décomposition LU.

    Raises:
        ValueError: Si la matrice n'est pas carrée.
        ZeroDivisionError: Si un pivot nul est rencontré lors du calcul.

    Examples:
        >>> A = [[4, 3], [6, 3]]
        >>> L = llu_unit(A)
        >>> L
        [[1.0, 0.0],
         [1.5, 1.0]]
    """
    if not verifCarre(A):
        raise ValueError("Matrice non carrée")

    n = len(A)
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for j in range(n):
        # Diagonale unité pour L
        L[j][j] = 1.0

        # Calcul des coefficients de U (ligne j)
        for i in range(j + 1):
            somme = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - somme

        # Calcul des coefficients de L (colonne j)
        for i in range(j + 1, n):
            somme = sum(L[i][k] * U[k][j] for k in range(j))
            if U[j][j] == 0:
                raise ZeroDivisionError(f"U[{j}][{j}] = 0, décomposition impossible sans pivot")
            L[i][j] = (A[i][j] - somme) / U[j][j]

    return L


def ulu_unit(A: list[list[float]]) -> list[list[float]]:
    """
    Calcule la matrice U de la décomposition LU d'une matrice carrée A,
    en supposant que la matrice L a une diagonale unité.

    Cette fonction réalise la même décomposition que `llu_unit`, mais
    renvoie la matrice U au lieu de L.

    Args:
        A (list[list[float]]): Matrice carrée à décomposer.

    Returns:
        list[list[float]]: Matrice U de la décomposition LU.

    Raises:
        ValueError: Si la matrice n'est pas carrée.
        ZeroDivisionError: Si un pivot nul est rencontré lors du calcul.

    Examples:
        >>> A = [[4, 3], [6, 3]]
        >>> U = ulu_unit(A)
        >>> U
        [[4.0, 3.0],
         [0.0, -1.5]]
    """
    if not verifCarre(A):
        raise ValueError("Matrice non carrée")

    n = len(A)
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for j in range(n):
        # Diagonale unité pour L
        L[j][j] = 1.0

        # Calcul des coefficients de U (ligne j)
        for i in range(j + 1):
            somme = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - somme

        # Calcul des coefficients de L (colonne j)
        for i in range(j + 1, n):
            somme = sum(L[i][k] * U[k][j] for k in range(j))
            if U[j][j] == 0:
                raise ZeroDivisionError(f"U[{j}][{j}] = 0, décomposition impossible sans pivot")
            L[i][j] = (A[i][j] - somme) / U[j][j]

    return U
