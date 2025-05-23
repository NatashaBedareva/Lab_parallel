#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <iomanip>
#include <tuple>
#include <chrono>
#include <cstdlib>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int NX = 20;      
int NY = 20;     
double TAU = -0.01;  
double EPS = 0.01;   
int MAX_ITER = 1000;
const std::string OUT_FILE = "result.dat";  

using namespace std::chrono;

const std::tuple<int, int, double> sources[] = {
    {NY/2, NX/3, 10},      
    {NY*2/3, NX*2/3, -25}, 
    {5, 10, 15},           
    {15, 25, -10},
    {10, 5, 20}
};
const int NUM_SOURCES = sizeof(sources) / sizeof(sources[0]);

double get_a(int row, int col) {
    if (row == col) return -4.0;
    if ((row + 1 == col && col % NX != 0) || 
        (row - 1 == col && row % NX != 0) ||  
        (row + NX == col && col < (NX * NY)) || 
        (row - NX == col && row >= NX))       
        return 1.0;
    return 0.0;
}

double get_b(int idx) {
    int i = idx / NX;
    int j = idx % NX;
    
    for (int k = 0; k < NUM_SOURCES; k++) {
        if (std::get<0>(sources[k]) == i && std::get<1>(sources[k]) == j) {
            return std::get<2>(sources[k]);
        }
    }
    return 0.0;
}

void init_matrix(double* A, int SIZE) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            A[i * SIZE + j] = get_a(i, j);
        }
    }
}

void init_b(double* b, int SIZE) {
    for (int i = 0; i < SIZE; ++i) {
        b[i] = get_b(i);
    }
}

double norm(const double* vec, int size) 
{
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += vec[i] * vec[i];
    }
    return std::sqrt(result);
}

__global__ void norm_kernel(double* result, const double* vec, int size) {
    extern __shared__ double shared_mem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    shared_mem[tid] = (i < size) ? vec[i] * vec[i] : 0.0;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared_mem[0]);
    }
}

//Ax - b
__global__ void matrix_vector_mult_sub_kernel(double* res, const double* mat, const double* vec, const double* y, int SIZE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE) {
        double sum = -y[i];
        for (int j = 0; j < SIZE; ++j) {
            sum += mat[i * SIZE + j] * vec[j];
        }
        res[i] = sum;
    }
}


__global__ void update_solution_kernel(double* x, const double* delta, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] -= -0.01f * delta[i];
    }
}

void solve_simple_iteration(double* A,
                           double* x,
                           const double* b,
                           int SIZE) {
    double norm_b = norm(b, SIZE);
    double h_res_norm = 0.0;
    int iter = 0;

    double *d_mat, *d_vec, *d_b, *d_res, *d_res_norm;
    
    cudaMalloc(&d_mat, SIZE * SIZE * sizeof(double));
    cudaMalloc(&d_vec, SIZE * sizeof(double));
    cudaMalloc(&d_b, SIZE * sizeof(double));
    cudaMalloc(&d_res, SIZE * sizeof(double));
    cudaMalloc(&d_res_norm, sizeof(double));

    cudaMemcpy(d_mat, A, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, x, SIZE * sizeof(double), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;


    auto start = high_resolution_clock::now();



    do {
        
        h_res_norm = 0.0;
        cudaMemcpy(d_res_norm, &h_res_norm, sizeof(double), cudaMemcpyHostToDevice);
        
        matrix_vector_mult_sub_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_res, d_mat, d_vec, d_b, SIZE);
        cudaDeviceSynchronize();
        
        norm_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(double)>>>(d_res_norm, d_res, SIZE);
        cudaDeviceSynchronize();

        update_solution_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_res, SIZE);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_res_norm, d_res_norm, sizeof(double), cudaMemcpyDeviceToHost);
        h_res_norm = sqrt(h_res_norm);

        std::cout << "Iteration " << std::setw(4) << ++iter 
                  << ": residual = " << std::scientific << std::setprecision(6) 
                  << h_res_norm/norm_b << " (target < " << EPS << ")\r";
        std::cout.flush();
        
        if (iter >= MAX_ITER) {
            std::cout << "\nMaximum iterations (" << MAX_ITER << ") reached\n";
            break;
        }
    } while (h_res_norm/norm_b >= EPS);

    cudaMemcpy(x, d_vec, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    if (iter < MAX_ITER) {
        std::cout << "\nConverged after " << iter << " iterations\n";
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Time run = %.5f s\n",float(duration.count())/1000000);
    
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_b);
    cudaFree(d_res);
    cudaFree(d_res_norm);
}


void save_results(const double* x, int size) {
    std::ofstream out(OUT_FILE, std::ios::binary);
    out.write(reinterpret_cast<const char*>(x), size * sizeof(double));
}

//./main--nx 10 --ny 10 --tau -0.005 --eps 0.001 --max-iter 5000

int main(int argc, char* argv[]) {
   
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("nx", po::value<int>(), "set grid size in X direction")
        ("ny", po::value<int>(), "set grid size in Y direction")
        ("tau", po::value<double>(), "set iteration parameter tau")
        ("eps", po::value<double>(), "set precision epsilon")
        ("max-iter", po::value<int>(), "set maximum number of iterations")
    ;


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("nx")) {
        NX = vm["nx"].as<int>();
    }
    if (vm.count("ny")) {
        NY = vm["ny"].as<int>();
    }
    if (vm.count("tau")) {
        TAU = vm["tau"].as<double>();
    }
    if (vm.count("eps")) {
        EPS = vm["eps"].as<double>();
    }
    if (vm.count("max-iter")) {
        MAX_ITER = vm["max-iter"].as<int>();
    }

    const int SIZE = NX * NY;
     

    double* A;
    double* b;
    double* x;

    A = (double*)malloc(SIZE * SIZE * sizeof(double));
    b = (double*)malloc(SIZE * sizeof(double));
    x = (double*)malloc(SIZE * sizeof(double));

    for (int i = 0; i < SIZE; ++i) {
        x[i] = 0.0;
    }

    
    std::cout << "Solving heat distribution on " << NY << "x" << NX << " grid\n";
    std::cout << "With " << NUM_SOURCES << " heat sources/sinks\n";
    std::cout << "Parameters: tau=" << TAU << ", eps=" << EPS << ", max_iter=" << MAX_ITER << "\n";

    init_matrix(A, SIZE);
    init_b(b, SIZE);


    solve_simple_iteration(A, x, b, SIZE);
    

    save_results(x, SIZE);

    free(A);
    free(b);
    free(x);

    return 0;
}