# encoding: utf-8
#cython: profile=False

import cython
import numpy as np
cimport numpy as cnp

cpdef voxmm_to_vox(double x, double y, double z, float[:] dim, float[:] voxres):
    return np.array(voxmm_to_vox_c(x,y,z,dim,voxres))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef (float, float, float) voxmm_to_vox_c(double x, double y, double z, float[:] dim, float[:] voxres) nogil:
    cdef:
        float i = x / voxres[0]
        float j = y / voxres[1]
        float k = z / voxres[2]
        int out_bound
    out_bound = (0 <= i <= (dim[0] - 1) and
                    0 <= j <= (dim[1] - 1) and
                    0 <= k <= (dim[2] - 1))
    if not out_bound:
        eps = float(1e-8)  # Epsilon to exclude upper borders
        i = max(0,
                min(voxres[0] * (dim[0] - eps), i))
        j = max(0,
                min(voxres[1] * (dim[1] - eps), j))
        k = max(0,
                min(voxres[2] * (dim[2] - eps), k))

    i= i / voxres[0] - 0.5
    j= j / voxres[1] - 0.5
    k= k / voxres[2] - 0.5
    return (i,j,k)
    
