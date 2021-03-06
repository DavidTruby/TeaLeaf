#include "kokkos_shared.hpp"
#include "../../shared.h"

using namespace Kokkos;

// Initialises p,r,u,w
void cg_init_u(
            const int x, const int y, const int coefficient, 
            KView p, KView r, KView u, KView w, KView density, 
            KView energy) 
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    {
        p(index) = 0.0;
        r(index) = 0.0;
        u(index) = energy(index)*density(index);
        w(index) = (coefficient == CONDUCTIVITY) 
            ? density(index) : 1.0/density(index);
    });
}

// Initialises kx,ky
void cg_init_k(
        const int x, const int y, const int halo_depth, KView w, 
        KView kx, KView ky, const double rx, const double ry)
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    {
        const size_t kk = index % x; 
        const size_t jj = index / x; 

        if(jj >= halo_depth && jj < y-1 && 
           kk >= halo_depth && kk < x-1)
        {
            kx(index) = rx*(w(index-1)+w(index)) /
                (2.0*w(index-1)*w(index));
            ky(index) = ry*(w(index-x)+w(index)) /
                (2.0*w(index-x)*w(index));
        }
    });
}

// Initialises w,r,p and calculates rro
void cg_init_others(
        const int x, const int y, const int halo_depth, KView kx, 
        KView ky, KView p, KView r, KView u, KView w, double* rro) 
{
    parallel_reduce(x*y, KOKKOS_LAMBDA (const int index, double& rro_temp)
    {
        const size_t kk = index % x; 
        const size_t jj = index / x; 

        if(kk >= halo_depth && kk < x - halo_depth &&
           jj >= halo_depth && jj < y - halo_depth)
        {
            const double smvp = SMVP(u);
            w(index) = smvp;
            r(index) = u(index)-w(index);
            p(index) = r(index);
            rro_temp += r(index)*p(index);
        }
    }, *rro);
}

// Calculates the value for w
void cg_calc_w(
        const int x, const int y, const int halo_depth, KView w, 
        KView p, KView kx, KView ky, double* pw) 
{

    parallel_reduce(x*y, KOKKOS_LAMBDA (const int& index, double& pw_temp)
    {
      const size_t kk = index % x;
      const size_t jj = index / x;

      if(kk >= halo_depth && kk < x - halo_depth &&
         jj >= halo_depth && jj < y - halo_depth)
      {
          const double smvp = SMVP(p);
          w(index) = smvp;
          pw_temp += w(index)*p(index);
      }
    }, *pw);
}

// Calculates the value of u and r
void cg_calc_ur(
        const int x, const int y, const int halo_depth, KView u, 
        KView r, KView p, KView w, const double alpha, double* rrn) 
{
    const int x_inner = x - 2*halo_depth;
    const int y_inner = y - 2*halo_depth;
    const int off0 = halo_depth*(x + 1);

    parallel_reduce(x_inner*y_inner, KOKKOS_LAMBDA (const int& i, double& rrn_temp)
    {
      const int col = i % x_inner;
      const int row = i / x_inner;
      const int index = off0 + col + row*x;

      u(index) += alpha*p(index);
      const double r_idx = r(index) - alpha*w(index);
      rrn_temp += r_idx*r_idx;
    }, *rrn);
}

// Calculates a value for p
void cg_calc_p(
        const int x, const int y, const int halo_depth, const double beta, 
        KView p, KView r) 
{
    const int x_inner = x - 2*halo_depth;
    const int y_inner = y - 2*halo_depth;
    const int off0 = halo_depth*(x + 1);

    parallel_for(x_inner*y_inner, KOKKOS_LAMBDA (const int& i)
    {
      const int col = i % x_inner;
      const int row = i / x_inner;
      const int index = off0 + col + row*x;

      p(index) = beta*p(index) + r(index);
    });
}

