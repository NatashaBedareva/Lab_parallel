#include <iostream>
#include <vector>
#include <cmath>
#include "omp.h"

#define FREAD 40

void simple_iteration(const double* A, const std::vector<double>& b, std::vector<double>& x, double tol, int max_iter,int n) {
    
    

    #pragma omp parallel
    {
        std::vector<double> x_new(n);
        for (int k = 0; k < max_iter; ++k) {

            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += A[i*n+j] * x[j];
                    }
                }
                x_new[i] = (b[i] - sum) / A[i*n+i];
            }
            
            double norm = 0.0;
            for (int i = 0; i < n; ++i) {
                norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
            }
            
            norm = sqrt(norm);
    
            if (norm < tol) {
                break;
            }
            
            x = x_new;
        }
    }
    
}

int main() {

    int n = 1000;
    double* A;
    A = (double *) malloc(sizeof(*A) * n *n);
    for (int i=0;i<n;i++)
	{
		for (int j = 0;j<n;j++)
		{
			if (i == j) {
                A[i * n + j] = n+1;
            } else {
                A[i * n + j] = 1.0;
            }
		}
		
	}
    std::vector<double> b(n, n+1);
    std::vector<double> x(n, 0.0);
    double tol = 1e-6;
    int max_iter = 1000000;
    double t = omp_get_wtime();
    simple_iteration(A, b, x, tol, max_iter,n);
    t = omp_get_wtime()-t;

    std::cout<<"Time parallel V2 = "<<t<<std::endl;
    
    /*
    std::cout << "RESULT: ";
    for (double xi : x) {
        std::cout << xi << " ";
    }
    std::cout << std::endl;

    */

    free(A);
    return 0;
}
