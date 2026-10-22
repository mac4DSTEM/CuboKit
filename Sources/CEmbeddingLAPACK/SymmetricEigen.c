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

void cubo_embedding_gemm_nt(int m, int n, int k, const double *a, const double *b, double *c) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, a, k, b, k, 0, c, n);
}

int cubo_embedding_svd(int m_, int n_, double *a, double *s, double *u, double *vt) {
    char jobz = 'S';
    __LAPACK_int m = m_, n = n_, lda = m_, r = m_ < n_ ? m_ : n_, ldu = m_, ldvt = r, info = 0, lwork = -1;
    double query = 0;
    __LAPACK_int *iwork = calloc(8 * (size_t)r, sizeof(__LAPACK_int));
    if (!iwork) return -100;
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &query, &lwork, iwork, &info);
    if (info || query < 1 || query > 2147483647) { free(iwork); return info ? info : -101; }
    lwork = (__LAPACK_int)query;
    double *work = calloc(lwork, sizeof(double));
    if (!work) { free(iwork); return -100; }
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
    free(work); free(iwork);
    return info;
}
