# Installation

numanalysis needs Python 3.11 or later and has no runtime dependency:

```bash
pip install numanalysis
```

Optional extras:

| Extra | Installs | For |
|---|---|---|
| `mp` | mpmath | multiprecision `mpf` / `mpc` numbers |
| `plot` | matplotlib, NumPy | the plots of the examples |

```bash
pip install "numanalysis[mp,plot]"
```

Inputs can be lists, tuples or NumPy arrays; outputs are always plain lists of lists,
so NumPy is never required.

## Unicode output on Windows

The text traces use Unicode symbols (`←`, `↔`, `√`, subscripts). If a Windows console
prints `UnicodeEncodeError`, switch Python to UTF-8 output, for instance with
`set PYTHONUTF8=1` (cmd) or `$env:PYTHONUTF8 = 1` (PowerShell). Jupyter is not
affected.
