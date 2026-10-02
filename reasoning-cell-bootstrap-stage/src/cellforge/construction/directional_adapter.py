"""Typed authored adapter: directional count cell -> grid plus activation."""
import numpy as np
from cellforge.construction.typed_population import repaired_count
def directional_packet(tensors,request):
    if not isinstance(request,dict) or set(request)!={'grid'}:raise ValueError('Grid request only; labels and foreign fields forbidden')
    grid,activation=repaired_count(tensors,request['grid'])
    return {'grid':grid,'activation':activation.tolist()}
def project_grid(packet):
    if not isinstance(packet,dict) or set(packet)!={'grid','activation'}:raise ValueError('complete directional packet required')
    grid=np.asarray(packet['grid']);activation=np.asarray(packet['activation'],float)
    if grid.ndim!=2 or not grid.size or grid.dtype.kind not in 'iu' or np.any(grid<0) or np.any(grid>9):
        raise ValueError('invalid grid output')
    if activation.shape!=grid.shape or not np.isfinite(activation).all():raise ValueError('aligned finite activation required')
    return grid.tolist()

