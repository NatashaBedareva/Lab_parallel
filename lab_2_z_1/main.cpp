#include <iostream>
#include "omp.h"
#include <cstdlib>
#include <time.h>

#define COUNT 40000
#define FREAD 60

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    #pragma omp parallel num_threads(FREAD)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m-1):(lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) 
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
                
        }
    }
}

void matrix_vector_serial(double *a, double *b, double *c, int m, int n)
{
    for(int i =0;i<m; i++)
    {
        c[i] = 0.0;
        for(int j = 0; j<n;j++)
        {
            c[i]+=a[i*n + j] * b[j];
        }

    }
}

void run_parallel(int m)
{
    double *a, *b, *c;
    int n=m;

    a = (double *) malloc(sizeof(*a) * m *n);
    b = (double *) malloc(sizeof(*b) * n);
    c = (double *) malloc(sizeof(*c) * m);
    #pragma omp parallel num_threads(FREAD)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m-1):(lb + items_per_thread - 1);
        
        for(int i=lb;i<ub;i++)
        {
            for(int j = 0;j<n;j++)
            {
                a[ i * n + j] = i + j;
            }
            c[i] = 0.0;
        }
    }
    
    for( int j = 0; j<n;j++)
    {
        b[j] = j;
    }
    double t = omp_get_wtime();
    matrix_vector_product(a,b,c,m,n);
    t = omp_get_wtime()-t;
    std::cout<<"time parallel = "<<t<<std::endl;
    free(a);
    free(b);
    free(c);
}

void run_serial(int m)
{
    double *a, *b, *c;
    int n=m;

    a = (double *) malloc(sizeof(*a) * m *n);
    b = (double *) malloc(sizeof(*b) * n);
    c = (double *) malloc(sizeof(*c) * m);
 
    for(int i=0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            a[ i * n + j] = i + j;
        }
    }
    for( int j = 0; j<n;j++)
    {
        b[j] = j;
    }
    double t = omp_get_wtime();
    matrix_vector_serial(a,b,c,m,n);
    t = omp_get_wtime()-t;
    std::cout<<"time serial = "<<t<<std::endl;
    free(a);
    free(b);
    free(c);
}

int main()
{
    run_parallel(COUNT);
    //run_serial(COUNT);
    
    return 0;
}
