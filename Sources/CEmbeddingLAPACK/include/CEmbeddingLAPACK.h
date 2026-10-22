/// Symmetric eigendecomposition (LAPACK dsyevd, upper triangle). `a` is n×n, column-major, overwritten with the
/// eigenvectors (column j belongs to `w[j]`); `w` receives the eigenvalues in ascending order. Returns LAPACK info.
int cubo_embedding_symmetric_eigen(int n, double *a, double *w);

/// c (n×n, row-major, upper triangle only) = alpha · aᵀa + beta · c, for a of `k × n`, row-major (BLAS dsyrk).
void cubo_embedding_syrk_upper(int n, int k, double alpha, const double *a, double beta, double *c);

/// c (m×n, row-major) = a · bᵀ, for a of `m × k` and b of `n × k`, both row-major (BLAS dgemm).
void cubo_embedding_gemm_nt(int m, int n, int k, const double *a, const double *b, double *c);

/// Thin SVD (LAPACK dgesdd, jobz 'S') of the m×n matrix `a`, column-major, overwritten. With r = min(m, n): `s` receives the r
/// singular values in descending order, `u` the m×r left vectors (column-major) and `vt` the r×n transposed right vectors
/// (column-major, leading dimension r). Returns LAPACK info.
int cubo_embedding_svd(int m, int n, double *a, double *s, double *u, double *vt);
