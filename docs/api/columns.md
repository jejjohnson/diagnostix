# Columns

A column is a field on levels along one axis. The levels are shared by every column (a 1-D coordinate), or each column has its own (a coordinate of the field's shape), as on terrain-following levels.

Functions here take plain arrays. The output is the field without the axis of its levels, in its dtype, and NaN where the question has no answer: a column on land has no mixed layer.

`first_crossing` is the rule the others stand on: the first level that reaches a value, and the position between that level and the one before it.

```python
import numpy as np

from diagnostix.columns import mixed_layer_depth

# ρ: (Ny, Nx, Nz) → h: (Ny, Nx)
h = mixed_layer_depth(rho, depth, valid=~np.isnan(rho))
```

::: diagnostix.columns
    options:
      show_root_heading: false
      show_source: false
