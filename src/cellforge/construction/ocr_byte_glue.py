"""Explicit numerical projection of legacy OCR categories to canonical byte ports.

This supplies no language parser and no missing OCR recognition capacity.
"""
import numpy as np

SOURCE_VOCAB = ['<pad>','<bos>','<eos>','0','1','2','3','4','5','6','7','8','9']
DESTINATION_INDICES = [0,1,2,51,52,53,54,55,56,57,58,59,60]

def project_ocr_category(params, source, *, failed=False):
    if failed is not False:
        raise ValueError('failed OCR cannot become a token observation')
    if set(params) != {'projection.gain','projection.bias'}:
        raise ValueError('exact explicit projection tensors required')
    gain=np.asarray(params['projection.gain']);bias=np.asarray(params['projection.bias'])
    x=np.asarray(source)
    for a in (gain,bias):
        if a.dtype.kind not in 'iuf' or a.shape!=(13,) or not np.isfinite(a).all():
            raise ValueError('thirteen finite numeric projection values required')
    if x.dtype.kind not in 'iuf' or x.shape!=(13,) or not np.isfinite(x).all() or not np.isin(x,[0,1]).all() or x.sum()!=1:
        raise ValueError('one actual OCR category onehot required')
    y=np.zeros(259,dtype=float)
    y[DESTINATION_INDICES]=gain*x+bias
    if not np.isin(y,[0,1]).all() or y.sum()!=1:
        raise ValueError('bridge output violates canonical category contract')
    return y
