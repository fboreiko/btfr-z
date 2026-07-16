import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from functools import partial

jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=("order",))
def jax_contra_interpolator(grid, positions, grid_axes, order=1):
    """
    A thin wrapper that converts the contra grid and query points into JAX arrays,
    performs interpolation, and applies a validity mask to handle out-of-bounds points
    by setting the log(mhi) value to NaNs.

    Args:
        grid (jnp.ndarray): The grid of values to interpolate.
        positions (jnp.ndarray): The positions to query in the grid.
        grid_axes (list of jnp.ndarray): The axes defining the grid.
        order (int, optional): The order of interpolation. Defaults to 1.

    Returns:
        jnp.ndarray: Interpolated values with out-of-bounds positions set to NaN.
    """
    indices = []
    valid_mask = jnp.ones(positions.shape[0], dtype=bool)  # Start with all points valid
    for i in range(positions.shape[1]):
        axis = grid_axes[i]
        grid_min = axis[0]
        grid_max = axis[-1]
        n_points = axis.shape[0]
        index_coord = (positions[:, i] - grid_min) / (grid_max - grid_min) * (n_points - 1)
        valid_mask &= (positions[:, i] >= grid_min) & (positions[:, i] <= grid_max)
        indices.append(index_coord)

    coords = jnp.stack(indices, axis=0)  # shape: (dimensions, n_points)
    interpolated_values = map_coordinates(grid, coords, order=order, mode='constant', cval=0)
    return jnp.where(valid_mask, interpolated_values, jnp.nan)