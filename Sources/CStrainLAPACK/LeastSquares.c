#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include "CStrainLAPACK.h"
int cubo_strain_least_squares(int rows, int columns, double *a, double *b, double rcond) {
    __LAPACK_int m=rows, n=columns, nrhs=1, lda=m, ldb=m>n?m:n, rank=0, info=0, lwork=-1;
    double *s=calloc(m<n?m:n,sizeof(double)), query=0;
    if (!s) return -100;
    dgelss_(&m,&n,&nrhs,a,&lda,b,&ldb,s,&rcond,&rank,&query,&lwork,&info);
    if (info || query < 1 || query > 2147483647) { free(s); return info?info:-101; }
    lwork=(__LAPACK_int)query;
    double *work=calloc(lwork,sizeof(double));
    if (!work) { free(s); return -100; }
    dgelss_(&m,&n,&nrhs,a,&lda,b,&ldb,s,&rcond,&rank,work,&lwork,&info);
    free(s); free(work); return info;
}
