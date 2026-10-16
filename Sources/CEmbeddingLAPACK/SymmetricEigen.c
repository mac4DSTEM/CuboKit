#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include "CEmbeddingLAPACK.h"
int cubo_embedding_symmetric_eigen(int n_, double *a, double *w) {
    char jobz = 'V', uplo = 'U';
    __LAPACK_int n = n_, lda = n_, info = 0, lwork = -1, liwork = -1, iquery = 0;
    double query = 0;
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, &query, &lwork, &iquery, &liwork, &info);
    if (info || query < 1 || query > 2147483647 || iquery < 1) return info ? info : -101;
    lwork = (__LAPACK_int)query; liwork = iquery;
    double *work = calloc(lwork, sizeof(double));
    __LAPACK_int *iwork = calloc(liwork, sizeof(__LAPACK_int));
    if (!work || !iwork) { free(work); free(iwork); return -100; }
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
    free(work); free(iwork);
    return info;
}

void cubo_embedding_syrk_upper(int n, int k, double alpha, const double *a, double beta, double *c) {
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, n, k, alpha, a, n, beta, c, n);
}
