#include <iostream>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

/*

#define COMMAND(NAME)  { #NAME, NAME ## _command }

g++ -std=c++1z prob_3.cpp -fopenmp

struct command{
  char *name;
  void (*function) (void);
};

struct command commands[] = {
  { "quit", quit_command }, { "help", help_command },
  COMMAND (exit), COMMAND (hello)
};

*/

using namespace std;
ofstream fs("prob_2.txt");

void openMPRun(int n, int num_th){
  int suma;
  double serialTime, parallelTime, startTime, endTime;
  clock_t tStart;
  
  omp_set_num_threads(num_th);
  
// O(n)
  
  tStart = clock();
  suma=0;
  for(int i=0; i<=n; i++){
    suma += i;
  }
  serialTime = (double)(clock() - tStart)/CLOCKS_PER_SEC;
  
  suma=0;
  startTime = omp_get_wtime();
# pragma omp parallel for reduction(+:suma)
  for(int i=0; i<=n; i++){
    suma += i;
  }
  endTime = omp_get_wtime();
  parallelTime = endTime - startTime;
  
  fs << n << " " << num_th <<" " << serialTime <<" " << parallelTime << " " << serialTime/(parallelTime*num_th)<<" "<< serialTime/(parallelTime) << endl;
  //cout<<"La suma de los N primeros numeros del natural n es: "<< suma <<endl;

// O(n^2)

  tStart = clock();
  suma=0;
  for(int i=1; i<n+1; i++){
    for(int j=1;j<n+1;j++){
      suma += i*j;
    }
  }
  serialTime = (double)(clock() - tStart)/CLOCKS_PER_SEC; 
  suma=0;
  startTime = omp_get_wtime();
# pragma omp parallel for collapse(2) reduction(+:suma)
  for(int i=1; i<n+1; i++){
    for(int j=1;j<n+1;j++){
      suma += i*j;
    }
  }
  endTime = omp_get_wtime();
  parallelTime = endTime - startTime;
  
  fs << n << " " << num_th <<" " << serialTime <<" " << parallelTime << " " << serialTime/(parallelTime*num_th)<<" "<< serialTime/(parallelTime) << endl;

  //cout<<"La suma de los elementos de la matriz N en terminos i*j es: "<< suma <<endl;
/*
  #pragma omp parallel for schedule(static) num_threads(8)
  for (int i = 0; i < 100; i++) {
    sleep(i);
    printf("Thread %d has completed iteration %d.\n", omp_get_thread_num( ), i);
  }
*/
}

int main(int argc, char *argv[]){
  int ns[][2] = {{100, 4}, {200, 4}, {300, 4}, {100, 8}, {200, 8}, {300, 8}, {100, 16}, {200, 16}, {300, 16}, {100, 32}, {200, 32}, {300, 32}, {100, 64}, {200, 64}, {300, 64}}; 
  fs << "#Intervalos Hilos sec_t par_t eff speedup" << endl;
  for (auto itera: ns) {
    openMPRun(itera[0], itera[1]);
  }
  return 0;
}
