#include <stdlib.h>
#include "../../shared.h"

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

// Initialises the CG solver
void cg_init(
    const int x,
    const int y,
    const int halo_depth,
    const int coefficient,
    double rx,
    double ry,
    double* rro,
    double* density,
    double* energy,
    double* u,
    double* p,
    double* r,
    double* w,
    double* kx,
    double* ky)
{
  if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
  {
    die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
  }

#pragma omp target teams distribute parallel for
  for (int index = 0; index < x*y; index++) {
      p[index] = 0.0;
      r[index] = 0.0;
      u[index] = energy[index]*density[index];
  }

#pragma omp target teams distribute parallel for
  for (int index = 1+x; index < (x-1)*(y-1); ++index)
  {
      w[index] = (coefficient == CONDUCTIVITY) 
        ? density[index] : 1.0/density[index];
  }

#pragma omp target teams distribute parallel for
  for (int index = halo_depth + halo_depth*x; index < (x-1)*(y-1); ++index)
  {
      kx[index] = rx*(w[index-1]+w[index]) /
        (2.0*w[index-1]*w[index]);
      ky[index] = ry*(w[index-x]+w[index]) /
        (2.0*w[index-x]*w[index]);
  }

  double rro_temp = 0.0;

#pragma omp target teams distribute parallel for reduction(+:rro_temp) map(tofrom: rro_temp)
  for (int c = 0; c < (y - 2 * halo_depth)*(x - 2 * halo_depth); ++c)
  {
    int jj = c / (x - 2 * halo_depth) + halo_depth; int kk = c % (x - 2 * halo_depth) + halo_depth;
      const int index = kk + jj*x;
      const double smvp = SMVP(u);
      w[index] = smvp;
      r[index] = u[index]-w[index];
      p[index] = r[index];
      rro_temp += r[index]*p[index];
  }

  // Sum locally
  *rro += rro_temp;
}

// Calculates w
void cg_calc_w(
    const int x,
    const int y,
    const int halo_depth,
    double* pw,
    double* p,
    double* w,
    double* kx,
    double* ky)
{
  double pw_temp = 0.0;

#pragma omp target teams distribute parallel for reduction(+:pw_temp) map(tofrom: pw_temp)
  for (int c = 0; c < (y - 2 * halo_depth)*(x - 2 * halo_depth); ++c)
  {
    int jj = c / (x - 2 * halo_depth) + halo_depth; int kk = c % (x - 2 * halo_depth) + halo_depth;
      const int index = kk + jj*x;
      const double smvp = SMVP(p);
      w[index] = smvp;
      pw_temp += w[index]*p[index];
  }

  *pw += pw_temp;
}

// Calculates u and r
void cg_calc_ur(
    const int x,
    const int y,
    const int halo_depth,
    const double alpha,
    double* rrn,
    double* u,
    double* p,
    double* r,
    double* w)
{
  double rrn_temp = 0.0;

#pragma omp target teams distribute parallel for reduction(+:rrn_temp) map(tofrom: rrn_temp)
  for (int c = 0; c < (y - 2 * halo_depth)*(x - 2 * halo_depth); ++c)
    {
      int jj = c / (x - 2 * halo_depth) + halo_depth; int kk = c % (x - 2 * halo_depth) + halo_depth;
      const int index = kk + jj*x;

      u[index] += alpha*p[index];
      r[index] -= alpha*w[index];
      rrn_temp += r[index]*r[index];
  }

  *rrn += rrn_temp;
}

// Calculates p
void cg_calc_p(
    const int x,
    const int y,
    const int halo_depth,
    const double beta,
    double* p,
    double* r)
{
#pragma omp target teams distribute parallel for
  for (int c = 0; c < (y - 2 * halo_depth)*(x - 2 * halo_depth); ++c)
  {
      int jj = c / (x - 2 * halo_depth) + halo_depth; int kk = c % (x - 2 * halo_depth) + halo_depth;
      const int index = kk + jj*x;

      p[index] = beta*p[index] + r[index];
  }
}

