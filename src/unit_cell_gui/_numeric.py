"""Shared input checks before conversion to the renderer's float32 storage."""
import math
import operator
import numpy as np

GPU_MAX = float(np.finfo(np.float32).max)


def gpu_values(value, label):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > GPU_MAX):
        raise ValueError(f'{label} must be finite and float32-representable')
    return array


def integer(value, label='index'):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{label} must be an integer, not bool')
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f'{label} must be an integer') from exc


def index_array(value, shape, label):
    # Object conversion preserves mixed bool/string/float values until checked.
    if isinstance(value, np.ndarray) and value.dtype.kind in 'iu':
        array = value.reshape(shape)
        if array.size and (np.any(array < 0) or np.any(array > np.iinfo(np.dtype(int)).max)):
            raise ValueError(f'{label} outside the supported index range')
        result = array.astype(int, copy=True)
        result.setflags(write=False)
        return result
    raw = np.asarray(value, dtype=object).reshape(shape)
    result = np.empty(raw.shape, dtype=int)
    for index in np.ndindex(raw.shape):
        number = integer(raw[index], label)
        if number < 0 or number > np.iinfo(np.dtype(int)).max:
            raise ValueError(f'{label} outside the supported index range')
        result[index] = number
    result.setflags(write=False)
    return result


def fractions(weights):
    """Preserve partial occupancy; normalize large finite positive weights."""
    values = tuple(max(0., float(value)) for value in weights)
    total = sum(values)
    if math.isfinite(total):
        scale = 1. if total <= 1. else 1. / total
        return tuple(value * scale for value in values)
    maximum = max(values, default=0.)
    scaled = tuple(value / maximum for value in values)
    total = sum(scaled)
    return tuple(value / total for value in scaled)


def atom_radius(atom, scale=1.):
    radius = getattr(atom, '_base_radius', None)
    if radius is None:
        weights = [max(0., component.occupancy) for component in atom.components]
        # Preserve ordinary arithmetic; rescale before extreme products.
        if max(weights, default=0.) > 1e250 / max(1, len(weights)):
            maximum = max(weights)
            weights = [weight / maximum for weight in weights]
        total = sum(weights)
        radius = (sum(weight * component.radius for weight, component in zip(weights, atom.components))
                  / total if total > 1e-12 else .22)
    value = radius * scale
    if not math.isfinite(value) or value > GPU_MAX:
        raise ValueError('scaled atom radius must be finite and float32-representable')
    return value


def orientation_matrix(value):
    matrix = gpu_values(value, 'orientation').reshape(3, 3).copy()
    # The actual uploaded matrix must remain invertible and well conditioned.
    rounded = matrix.astype(np.float32).astype(float)
    singular = np.linalg.svd(rounded, compute_uv=False)
    if singular[-1] <= 0 or singular[-1] <= singular[0] * 1e-6:
        raise ValueError('orientation must be invertible and numerically stable (condition < 1e6)')
    gpu_values(np.linalg.inv(matrix), 'inverse orientation')
    return matrix


def scene_numeric_bounds(scene):
    """Immutable, camera/scale-independent summary, prepared with the model."""
    static = np.zeros(3)
    if len(scene.cell_vertices):
        static = np.maximum(static, np.max(np.abs(scene.cell_vertices),axis=0))
    lift = max(1e-4,.001*scene.base_radius)
    for poly in scene.polyhedra:
        if len(poly.vertices):
            static = np.maximum(static,np.max(np.abs(poly.vertices),axis=0)+lift)
    centres = (np.max(np.abs(scene.atom_centers),axis=0)
               if len(scene.atoms) else np.zeros(3))
    maximum = max((atom._base_radius if hasattr(atom,'_base_radius') else atom_radius(atom)
                   for atom in scene.atoms),default=0.)
    bond_radius = max(.018,.020*scene.base_radius)/3. if len(scene.bonds) else 0.
    static.setflags(write=False);centres.setflags(write=False)
    return static,centres,maximum,bond_radius


def scene_extent(scene, scale):
    """Conservative generated bounds, reusing the immutable model summary."""
    bounds = getattr(scene,'_numeric_bounds',None)
    if bounds is None:  # compatibility with previously pickled model instances
        bounds = scene_numeric_bounds(scene)
    static,centres,maximum,bond_radius = bounds
    radius = maximum*scale
    gpu_values(radius,'scaled atom radius')
    padding = .012*scene.base_radius
    extent = static.copy()
    if len(scene.atoms):
        extent = np.maximum(extent,centres+max(radius+padding,bond_radius))
    for index,point in getattr(scene,'position_overrides',{}).items():
        radius = atom_radius(scene.atoms[index],scale)
        extent = np.maximum(extent,np.abs(point)+max(radius+padding,bond_radius))
    lift = max(1e-4,.001*scene.base_radius)
    for poly in getattr(scene,'polyhedron_overrides',{}).values():
        if len(poly.vertices):
            extent = np.maximum(extent,np.max(np.abs(poly.vertices),axis=0)+lift)
    gpu_values(extent,'generated scene coordinates/radii')
    return extent


def camera_extent(extent, orientation, base_radius=None):
    camera = np.abs(orientation) @ extent
    gpu_values(camera, 'transformed scene coordinates/radii')
    if base_radius is not None:
        coefficient = 3.96/base_radius  # maximum XY projection coefficient at zoom 4.5
        gpu_values(coefficient, 'projection scale')
        # Outlines use an unscaled screen-facing radius. Keep a conservative
        # bound for that path too, including very small general transforms.
        gpu_values(np.maximum(camera,extent)[:2]*coefficient, 'projected scene coordinates/radii')
        lift = max(1e-4,.001*base_radius)
        gpu_values(extent + np.abs(np.linalg.inv(orientation)) @ np.full(3,lift),
                   'inverse-transformed hatch coordinates')
