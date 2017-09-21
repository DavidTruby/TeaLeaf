#include <cstdlib>
#include <RAJA/RAJA.hpp>
using ExecPolicy = RAJA::omp_target_parallel_for_exec<0>;
using ReducePolicy = RAJA::omp_target_reduce<0>;

template <typename T, typename F>
T reduce_sum(int start, int end, F loop_body) {
  T reduction_variable {};
#pragma omp target teams distribute parallel for reduction(+ : reduction_variable) map(tofrom: reduction_variable)
  for (int i = start; i < end; ++i) {
    reduction_variable += loop_body(i);
  }
  return reduction_variable;
}

extern "C" {
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

  RAJA::forall<ExecPolicy>(0, x*y, [=](int index) {
      p[index] = 0.0;
      r[index] = 0.0;
      u[index] = energy[index]*density[index];
    });

  RAJA::forall<ExecPolicy>(1+x, (x-1)*(y-1), [=](int index) {
      w[index] = (coefficient == CONDUCTIVITY) 
        ? density[index] : 1.0/density[index];
    });

  RAJA::forall<ExecPolicy>(1+x, (x-1)*(y-1), [=](int index) {
      kx[index] = rx*(w[index-1]+w[index]) /
        (2.0*w[index-1]*w[index]);
      ky[index] = ry*(w[index-x]+w[index]) /
        (2.0*w[index-x]*w[index]);
    });

  double rro_temp = reduce_sum<double>(0, (y-2*halo_depth)*(x-2*halo_depth), [=](int c){
    int jj = c / (x - 2 * halo_depth) + halo_depth; int kk = c % (x - 2 * halo_depth) + halo_depth;
      const int index = kk + jj*x;
      const double smvp = SMVP(u);
      w[index] = smvp;
      r[index] = u[index]-w[index];
      p[index] = r[index];
      return r[index]*p[index];
    });

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
  double pw_temp = reduce_sum<double>(0, (y-2*halo_depth)*(x-2*halo_depth), [=](int c)  {
      int jj = c / (x - 2 * halo_depth) + halo_depth; int kk = c % (x - 2 * halo_depth) + halo_depth;
      const int index = kk + jj*x;
      const double smvp = SMVP(p);
      w[index] = smvp;
      return w[index]*p[index];
    });

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
  double rrn_temp = reduce_sum<double>(0, (y-2*halo_depth)*(x-2*halo_depth), [=](int c) {
      int jj = c / (x - 2 * halo_depth) + halo_depth; int kk = c % (x - 2 * halo_depth) + halo_depth;
      const int index = kk + jj*x;

      u[index] += alpha*p[index];
      r[index] -= alpha*w[index];
      return r[index]*r[index];
    });

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
  RAJA::forall<ExecPolicy>(0, (y-2*halo_depth)*(x-2*halo_depth), [=](int c) {
      int jj = c / (x - 2 * halo_depth) + halo_depth; int kk = c % (x - 2 * halo_depth) + halo_depth;
      const int index = kk + jj*x;

      p[index] = beta*p[index] + r[index];
  });
}

}
