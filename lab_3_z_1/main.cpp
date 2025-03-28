#include <iostream>
#include "omp.h"
#include <cstdlib>
#include <time.h>
#include <thread>

#define COUNT 40000
#define THREAD 10


void mul_matrix_vector(double *a, double *b, double *c,int start, int end)
{
    for(int i = start;i<end; i++)
    {
        c[i] = 0.0;
        for(int j = 0; j<COUNT;j++)
        {
            c[i]+=a[i*n + j] * b[j];
        }

    }
}

void initializing(double* a, double* b, double* c, int start, int end)
{
    for(int i=start;i<end;i++)
    {
        for(int j = 0;j<COUNT;j++)
        {
            a[ i * n + j] = i + j;
        }
        c[i] = 0.0;
    }
}

void run_parallel(int m,double* a, double* b, double* c)
{

    std::vector<std::thread> threads;

    for (int i = 0; i< THREAD;i++)
    {
        int thread_id = i;
        int items_per_thread = COUNT / i;
        int lb = thread_id * items_per_thread;
        int ub = (thread_id == i - 1) ? (COUNT-1):(lb + items_per_thread - 1);

        threads.emplace_back(initializing, 
                                std::cref(a), 
                                std::cref(b), 
                                std::ref(c), 
                                lb, ub);
    }

    for (auto& t : threads) {
        t.join();
    }

    threads.clear();
    
    for( int j = 0; j<n;j++)
    {
        b[j] = j;
    }
    double t = omp_get_wtime();


    for (int i = 0; i< THREAD;i++)
    {
        int thread_id = i;
        int items_per_thread = COUNT / i;
        int lb = thread_id * items_per_thread;
        int ub = (thread_id == i - 1) ? (COUNT-1):(lb + items_per_thread - 1);

        threads.emplace_back(mul_matrix_vector, 
                                std::cref(a), 
                                std::cref(b), 
                                std::ref(c), 
                                lb, ub);
    }




    t = omp_get_wtime()-t;
    std::cout<<"time parallel = "<<t<<std::endl;
}

void run_serial(int m,double* a, double* b, double* c)
{
 
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
    mul_matrix_vector(a,b,c,0,COUNT);
    t = omp_get_wtime()-t;
    std::cout<<"time serial = "<<t<<std::endl;
}

int main()
{
    a_serial = (double *) malloc(sizeof(*a) * m *n);
    b_serial = (double *) malloc(sizeof(*b) * n);
    c_serial = (double *) malloc(sizeof(*c) * m);

    a_parallel = (double *) malloc(sizeof(*a) * m *n);
    b_parallel = (double *) malloc(sizeof(*b) * n);
    c_parallel = (double *) malloc(sizeof(*c) * m);


    run_parallel(COUNT,a_parallel,b_parallel, c_parallel);
    run_serial(COUNT,a_serial,b_serial,c_serial);

    free(a_serial);
    free(b_serial);
    free(c_serial);

    free(a_parallel);
    free(b_parallel);
    free(c_parallel);
    
    return 0;
}
