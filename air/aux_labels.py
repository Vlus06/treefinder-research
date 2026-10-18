# air/aux_labels.py
# Auxiliary (structure / edge) targets computed on the fly from the TreeFinder label raster.
#   mode "fixed"    : positives are dead-tree pixels (label == 1, valid pixels only)  -- what the paper describes
#   mode "released" : positives are the NO-DATA pixels (label raster == 255), edge stored as uint8 0/255 and
#                     boundary_dist = edt(1 - edge) on the uint8 array -- bug-for-bug replica of the public
#                     RONGHUA-LI/TreeMortSeg `aux_label_gen.py` + `random_split.py`, kept to measure its effect.
import numpy as np
from scipy import ndimage

_S8 = np.ones((3, 3), dtype=bool)


def _edge(binary: np.ndarray) -> np.ndarray:
    return binary & ~ndimage.binary_erosion(binary, structure=_S8)


def _instance_distance(binary: np.ndarray) -> np.ndarray:
    """Per-instance Euclidean distance to the instance boundary, divided by that instance's maximum (centre = 1)."""
    out = np.zeros(binary.shape, np.float32)
    lab, n = ndimage.label(binary, structure=_S8)
    for i in range(1, n + 1):
        m = lab == i
        d = ndimage.distance_transform_edt(m)
        mx = d.max()
        if mx > 0:
            out[m] = d[m] / mx
    return out


def make_aux(label: np.ndarray, no_data: np.ndarray, mode: str):
    """label: [H,W] int (0/1, no-data already mapped to 0); no_data: [H,W] bool. Returns edge, dist, boundary_dist."""
    if mode == "fixed":
        pos = (label == 1) & ~no_data
        edge = _edge(pos)
        dist = _instance_distance(pos)
        if edge.any():
            bd = ndimage.distance_transform_edt(~edge).astype(np.float32)
        else:
            bd = np.full(label.shape, 10.0, np.float32)
        return edge.astype(np.float32), dist, bd
    if mode == "released":
        pos = no_data.astype(np.uint8)
        erosion = ndimage.binary_erosion(pos, structure=_S8)
        edge_u8 = np.where((pos - erosion) == 1, 255, 0).astype(np.uint8)
        dist = _instance_distance(pos.astype(bool))
        bd = ndimage.distance_transform_edt(1 - edge_u8).astype(np.float32)   # uint8 arithmetic: 1-255 == 2
        return edge_u8.astype(np.float32), dist, bd
    raise ValueError(mode)
