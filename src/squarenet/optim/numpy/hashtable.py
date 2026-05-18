import numpy as np

class HashTable():
    def __init__(self, grid, dims):
        self.shape = grid.shape
        self.gtable = grid
        self.htable = np.full(self.shape,  fill_value = -1, dtype=grid.dtype)
        self.dims = dims
        self.D = len(dims)
        self.iter = 0

        self.dslice = []
        self.hslice = []
        self.gslice = []

        self.gslice, self.hslice, self.dslice = self.init_slices()

    def init_slices(self):
        D = self.D
        dslice = []
        hslice = []
        gslice = []

        for it in range(D):
            d = self.dims[it]
            Ndim = self.gtable.ndim

            dsl = [slice(0, -1)] * Ndim
            dsl[d] = slice(None)
            dslice.append(tuple(dsl))

            hslice.append([None] * (2 * D))
            gslice.append([None] * (2 * D))

            for i in range(2 * D):
                hsl = [slice(None)] * Ndim
                gsl = [slice(None)] * Ndim

                hsl[d] = slice(i, None, 2 * D)
                gsl[d] = slice(i, None, 2 * D)

                #==================
                # Haching the table
                #==================
                if i < D:
                    for j in range(i):
                        if j != it:
                            hsl[self.dims[j]] = slice(None, -1)
                            gsl[self.dims[j]] = slice(1, None)
                if i >= D:
                    for j in range(2 * D - i):
                        if j != it:
                            hsl[self.dims[j]] = slice(None, -1)
                            gsl[self.dims[j]] = slice(1, None)

                gslice[it][i] = tuple(gsl)
                hslice[it][i] = tuple(hsl)

        return gslice, hslice, dslice

    def sort(self):
        it = self.iter
        d = self.dims[it]
        D = self.D
    
        for hsl, gsl in zip(self.hslice[it], self.gslice[it]):
            self.htable[hsl] = self.gtable[gsl]

        self.htable[self.dslice[it]].sort(axis = d)

        for hsl, gsl in zip(self.hslice[it], self.gslice[it]):
            self.gtable[gsl] = self.htable[hsl]
        
        self.iter_all()

    def iter_all(self):
        it = self.iter
        d = self.dims[it]
        D = self.D

        for sls in [self.hslice[it], self.gslice[it]]:
            for isl, sl in enumerate(sls):
                sl_list = list(sl)
                start = sl_list[d].start
                sl_list[d] = slice((start + 1) % (2 * D), None, 2 * D)
                sls[isl] = tuple(sl_list)

        self.iter = (it + 1) % D