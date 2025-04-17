#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <fstream>
#include <tuple>
#include <sstream>
#include <stdexcept>
#include <cmath> 
#include <limits> 


template<typename T>
struct taskClient {
    T arg1;
    T arg2;
    std::string name_fun;
    std::function<T(T,T)> function;
};

template<typename T>
std::queue<std::pair<size_t, taskClient<T>>> tasks_client;

template<typename T>
std::unordered_map<int, taskClient<T>> tasks_client_for_test;

bool stop_flag = false;

template<class T>
class Server {
    std::mutex mut;
    std::unordered_map<int, T> results;
    std::queue<std::pair<size_t, std::future<T>>> tasks;

    //std::jthread thread_start_server;
    std::thread thread_add_task;
    
public:
    
    

    Server() {};

    /*
    void server_thread() {
        
        while (!stop_flag )
        {
            std::unique_lock lock_res{mut};
            if (!tasks.empty()) {
                auto& task = tasks.front();

                results[task.first] = task.second.get();
                std::cout << "----work " << task.first << " ---- res = "<< results[task.first]<< "\n";
                tasks.pop();
            }
            else
            {
                std::cout<<"Empty\n";
            }
            lock_res.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    */
    

    void start() {
        std::cout << "Start\n";
        //thread_start_server = std::jthread (&Server::server_thread, this);
        thread_add_task = std::thread (&Server::add_task_thread, this, 1);
    }

    void stop() {
        std::cout<<"Server is stoped!\n";
        stop_flag=true;
        
        thread_add_task.join();
        //thread_start_server.join();
        std::cout << "End\n";
    }

    void add_task_thread(int number_task) {
        while (!stop_flag) {
            std::unique_lock lock_res{mut};
            if (!tasks_client<T>.empty()) {
                auto task = tasks_client<T>.front();
                tasks_client<T>.pop();
                
                int id = task.first;
                T arg1 = task.second.arg1;
                T arg2 = task.second.arg2;

                std::cout << "id = " << id << " arg1 = " << arg1 << " arg2 = " << arg2 << std::endl;

                
                std::future<T> result = std::async(std::launch::async,
                    [function = task.second.function, arg1, arg2]() -> T {
                        return function(arg1, arg2);
                    });

                results[id] = result.get();
                //tasks.push({id, std::move(result)});
                
                
            }
            lock_res.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    T request_result(int id_res) {
        std::lock_guard lock{mut};
        if (results.count(id_res)) 
        {
            return results[id_res];
        }
    }
};



int id_task = 0;
std::mutex mutClient;
//std::unordered_map<int, std::string> id_fun;

const int a = 1;
const int b = 100;
const int c = 5;
int NUM_TASKS = 10;

template<class T>
class ClientPow
{
public:
    std::thread thread;
    void start()
    {
        
        thread = std::thread(&ClientPow::client_works, this);

    }
    
    void client_works()
    {
        std::unique_lock lock_client{mutClient, std::defer_lock};
        for(int i=0;i<NUM_TASKS ;i++)
        {
            taskClient<T> task;

            task.arg1 = a + rand()%(b-a+1);
            task.arg2 = a + rand()%(c-a+1);
            task.function = [this](T x, T y) { return fun_pow(x, y); };
            task.name_fun = "Pow";
            
            lock_client.lock();
            id_task+=1;
            
            tasks_client<T>.push(std::make_pair(id_task, task)); 
            tasks_client_for_test<T>[id_task] = task; 
            lock_client.unlock();
            
        }
        std::cout<<"End Client\n";
    };

    T fun_pow(T x, T y)
    {
        return std::pow(x,y);
    }
};

template<class T>
class ClientSin
{
public:

    std::thread thread;

    void start()
    {
        thread = std::thread(&ClientSin::client_works, this);

    }
    
    void client_works()
    {
        std::unique_lock lock_client{mutClient, std::defer_lock};
        for(int i=0;i<NUM_TASKS ;i++)
        {
            taskClient<T> task;

            task.arg1 = a + rand()%(b-a+1);
            task.function = [this](T x, T y) { return fun_sin(x,y); };
            task.name_fun = "Sin";
            
            lock_client.lock();
            
            id_task+=1;
            tasks_client<T>.push(std::make_pair(id_task, task)); 
            tasks_client_for_test<T>[id_task] = task;
            lock_client.unlock();
           
        }
        std::cout<<"End Client\n";
    };

  
    T fun_sin(T arg1,T arg2)
    {
        return std::sin(arg1);
    }
};

template<class T>
class ClientSqrt
{
public:

    std::thread thread;

    void start()
    {
        thread = std::thread(&ClientSqrt::client_works, this);
    }
    
    void client_works()
    {
        std::unique_lock lock_client{mutClient, std::defer_lock};
        for(int i=0;i<NUM_TASKS ;i++)
        {
            taskClient<T> task;

            task.arg1 = a + rand()%(b-a+1);
            task.function = [this](T x, T y) { return fun_sqrt(x,y); };
            task.name_fun = "Sqrt";
            
            lock_client.lock();
            
            id_task+=1;
            tasks_client<T>.push(std::make_pair(id_task, task)); 
            tasks_client_for_test<T>[id_task] = task;
            lock_client.unlock();

        }
        std::cout<<"End Client\n";
    };

    T fun_sqrt(T arg1,T arg2)
    {
        return std::sqrt(arg1);
    }
};



std::tuple<int, std::string, double> parse_result_line(const std::string& line) {
    std::istringstream iss(line);
    std::string token;
    int id;
    std::string func;
    double result;

    if (!(iss >> token >> id) || token != "ID") {
        throw std::runtime_error("Invalid ID format");
    }

    if (!(iss >> token >> func) || token != "func") {
        throw std::runtime_error("Invalid function format");
    }

    if (!(iss >> token >> result) || token != "result") {
        throw std::runtime_error("Invalid result format");
    }

    return std::make_tuple(id, func, result);
}

const double EPSILON = 1;

void test_parser(std::string test_line) {
    try {
        auto [id, func, result] = parse_result_line(test_line);
        
        std::cout << "Parsed successfully:\n";
        std::cout << "ID: " << id << "\n";
        std::cout << "Function: " << func << "\n";
        std::cout << "Result: " << result << "\n";

        auto task = tasks_client_for_test<double>[id];

        if(func != task.name_fun) {
            std::cerr << "Test failed: name function is different\n";
            return;
        }
        
        double expected = 0.0;
        if(func == "Sin") {
            expected = std::sin(task.arg1);
        }
        else if (func == "Pow") {
            expected = std::pow(task.arg1, task.arg2);
        }
        else if (func == "Sqrt") {
            expected = std::sqrt(task.arg1);
        }
        else {
            std::cerr << "Test failed: unknown function name\n";
            return;
        }

        if (std::fabs(result - expected) > EPSILON) {
            std::cerr << "Test failed: results don't match\n";
            std::cerr << "Expected: " << expected << "\n";
            std::cerr << "Actual: " << result << "\n";
            std::cerr << "Difference: " << std::fabs(result - expected) << "\n";
        } else {
            std::cout << "Test passed!\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Test failed!\n";
    }
}


int main()
{
    Server<double> server; 
    
    ClientPow<double> client_pow;
    ClientSin<double> client_sin;
    ClientSqrt<double> client_sqrt;

    client_sin.start();
    client_pow.start();
    client_sqrt.start();

    server.start();

    client_pow.thread.join();
    client_sin.thread.join();
    client_sqrt.thread.join();

    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    server.stop();

    std::ofstream out;          
    out.open("result.txt");      
    if (out.is_open())
    {
        for (int i = 1; i <= tasks_client_for_test<double>.size(); i++)
    {
        out << "ID " << i << " func " << tasks_client_for_test<double>[i].name_fun 
            << " result " << server.request_result(i) << std::endl;
    }
    }
    out.close(); 
    std::cout << "File has been written" << std::endl;


    std::string line;
 
    std::ifstream in("result.txt");
    if (in.is_open())
    {
        while (std::getline(in, line))
        {
            test_parser(line);
        }
    }
    in.close();


    return 0;
}
