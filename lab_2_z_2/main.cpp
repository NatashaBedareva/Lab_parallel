#include <iostream>
#include "omp.h"
#include <cstdlib>
#include <cmath>

const double a = -4.0;
const double b = 4.0;
const int nsteps = 80000000;

const double PI = 3.14159265358979323846;

#define FREAD 40

double func(double x)
{
    return exp(-x * x);
}
double integrate(double (*func)(double),double a, double b, int n)
{
    double h = (b-a)/n;
    double sum = 0.0;

    for(int i = 0; i<n;i++)
    {
        sum+= func(a+h*(i+0.5));
    }
    sum*=h;
    return sum;
}

double integrate_omp(double (*func)(double),double a, double b, int n)
{
    double h = (b-a)/n;
    double sum = 0.0;

    #pragma omp parallel num_threads(FREAD)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n-1):(lb + items_per_thread - 1);
        
        double sumloc = 0.0;
        for(int i = lb; i<=ub;i++)
        {
            sumloc += func(a+h*(i+0.5));     
        }
        #pragma omp atomatic
        sum+= sumloc;
    }
    
    sum*=h;
    return sum;
}

void run_serial()
{
    double t = omp_get_wtime();
    double res = integrate(func,a,b,nsteps);
    t = omp_get_wtime()-t;
    std::cout<<"Time serial = "<<t<<std::endl;
    std::cout<<"RESULT = "<<res<<std::endl;
}

void run_parallel()
{
    double t = omp_get_wtime();
    double res = integrate_omp(func,a,b,nsteps);
    t = omp_get_wtime()-t;
    std::cout<<"Time parallel = "<<t<<std::endl;
    std::cout<<"RESULT = "<<res<<std::endl;
}


int main()
{
    
    //run_serial();
    run_parallel();
    
    return 0;
}
