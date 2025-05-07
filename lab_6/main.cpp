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

void matrix_vector_mult_sub(double* res,
    const double* mat,
    const double* vec,
    const double* y,
    int SIZE) {
    #pragma acc parallel loop copyin(mat[0:SIZE*SIZE], vec[0:SIZE], y[0:SIZE]) \
            copyout(res[0:SIZE]) gang vector
    for (int i = 0; i < SIZE; ++i) {
        double sum = -y[i];
        #pragma acc loop seq
        for (int j = 0; j < SIZE; ++j) {
            sum += mat[i * SIZE + j] * vec[j];
        }
        res[i] = sum;
    }
}

void update_solution(double* x, const double* delta, int size) 
{
    for (int i = 0; i < size; ++i) {
        x[i] -= TAU * delta[i];
    }
}

void solve_simple_iteration(double* A,
                           double* x,
                           const double* b,
                           int SIZE) {
    double* Axmb = (double*)malloc(SIZE * sizeof(double));
    double norm_b = norm(b, SIZE);
    double norm_Axmb;
    int iter = 0;

    do {
        matrix_vector_mult_sub(Axmb, A, x, b, SIZE);
        norm_Axmb = norm(Axmb, SIZE);
        update_solution(x, Axmb, SIZE);
        
        std::cout << "Iteration " << std::setw(4) << ++iter 
                  << ": residual = " << std::scientific << std::setprecision(6) 
                  << norm_Axmb/norm_b << " (target < " << EPS << ")\r";
        std::cout.flush();
        
        if (iter >= MAX_ITER) {
            std::cout << "\nMaximum iterations (" << MAX_ITER << ") reached\n";
            break;
        }
    } while (norm_Axmb/norm_b >= EPS);

    if (iter < MAX_ITER) {
        std::cout << "\nConverged after " << iter << " iterations\n";
    }
    free(Axmb);
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
    double* A = (double*)malloc(SIZE * SIZE * sizeof(double));
    double* b = (double*)malloc(SIZE * sizeof(double));
    double* x = (double*)malloc(SIZE * sizeof(double));

    for (int i = 0; i < SIZE; ++i) {
        x[i] = 0.0;
    }

    std::cout << "Solving heat distribution on " << NY << "x" << NX << " grid\n";
    std::cout << "With " << NUM_SOURCES << " heat sources/sinks\n";
    std::cout << "Parameters: tau=" << TAU << ", eps=" << EPS << ", max_iter=" << MAX_ITER << "\n";

    init_matrix(A, SIZE);
    init_b(b, SIZE);

    auto start = high_resolution_clock::now();

    solve_simple_iteration(A, x, b, SIZE);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    //std::cout << "Time run = " << float(duration.count())/1000000 << " s\n";
    printf("Time run = %.5f s\n",float(duration.count())/1000000);

    save_results(x, SIZE);

    free(A);
    free(b);
    free(x);

    return 0;
}