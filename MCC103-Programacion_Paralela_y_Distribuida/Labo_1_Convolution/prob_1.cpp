#include <iostream>
#include <fstream>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

using namespace std;
ofstream fs("prob_1.txt");

double f( double a ){
  return (4.0 / (1.0 + a*a));
}

double trapezoidalRule(double (*f)(double),double a,double b,int n){
	int i;
	double sum=0.0, h=(b-a)/n;
	for(i=0;i<n-1;i++){
    sum = sum + ((*f)(a+h*i) + (*f)(a+h*(i+1)))*h/2;
	}
	return sum;
}

void openMPRun(int argc, int argv[], double a, double b){
  int num_th, th_id;
  int i, n, n1;
  double h, integralValue;
  double a1, b1;
  double startwtime, endwtime, serialTime, parallelTime;
  clock_t tStart;
  
  if( argc == 2 ) {
    n = argv[0];
    num_th = argv[1];
  } else if(argc == 1){
    n = argv[0];
    num_th = omp_get_max_threads();
  } else {
    num_th = omp_get_max_threads();
    cout<<"\nIngrese numero de intervalos: \n";
    cin>>n;
  }
  omp_set_num_threads(num_th);
  if (n == 0)
      cout<<"\nNo puede haber division entre 0.\n\n";
  else{
    h = (b-a)/n; // tamaño de paso
    
    cout<<"\nMetodo Trapecio: \n";
    
    cout<<"\nserialTime: \n";
    
    tStart = clock();
    integralValue = trapezoidalRule(f,a,b,n);
    serialTime = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    
    cout<<"La integral de f es: "<< integralValue <<"\n";
    cout<<"El calculo fue en " << serialTime << " en serialTime.\n";
    
    cout<<"\nParalelo: \n";
    integralValue=0.0;
    n1 = n/num_th; // cantidad de intervalos para cada hilo
    
    #pragma omp parallel shared(a,b,h,n1,num_th) private(th_id, a1, b1)
    {
      th_id=omp_get_thread_num();
      if (th_id == 0){
        cout<<"\nCalculando La integral en OpenMP usando "<<n<<" intervalos con " << num_th <<" hilos\n";
      }
      a1 = a + th_id*n1*h;
      b1 = a1 + n1*h;
      
      startwtime = omp_get_wtime();
      #pragma omp reduction(+:integralValue)
      integralValue += trapezoidalRule(f,a1,b1,n1);
      endwtime = omp_get_wtime();
      #pragma omp barrier
      //cout << integralValue << endl;
      if (th_id == 0){
        parallelTime = endwtime-startwtime;
        cout<<"La integral de f es: "<< integralValue <<"\n\n";
        cout<<"El calculo fue en " << endwtime-startwtime << " en "<< num_th <<" hilos\n\n";
        fs << n <<" "<<num_th <<" " << serialTime <<" " << parallelTime <<" "<< serialTime/(parallelTime*num_th)<<" "<< serialTime/(parallelTime) << " Y" <<endl;
      }
    }
    
    cout << "SpeedUp usando el trapecio: "<< serialTime/(parallelTime) <<endl;
    cout << "Eficiencia usando el trapecio: "<< serialTime/(parallelTime*num_th) <<endl;
    
    //fs << n <<" "<<num_th <<" " << serialTime <<" " << parallelTime <<" "<< serialTime/(parallelTime*num_th)<<" "<< serialTime/(parallelTime) << " Y" <<endl;
    
    cout<<"\nMetodo For: \n";

    cout<<"\nserialTime: \n";
    int i;
	  integralValue=0.0;
    tStart = clock();
	  for(i=0;i<n-1;i++){
      integralValue = integralValue + (f(a+h*i) + f(a+h*(i+1)))*h/2;
	  }
    serialTime = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    
    cout<<"La integral de f es: "<< integralValue <<"\n";
    cout<<"El calculo fue en " << serialTime << " en serialTime.\n";
    
    cout<<"\nParalelo: \n";
	  integralValue=0.0;
    #pragma omp parallel shared(a,b,h,n1,num_th) private(th_id, a1, b1)
    {
      th_id=omp_get_thread_num();
      if (th_id == 0){
        cout<<"\nCalculando La integral en OpenMP usando "<<n<<" intervalos con " << num_th <<" hilos\n";
      }
      a1 = a + th_id*n1*h;
      b1 = a1 + n1*h;
      double my_value=0.0;
      startwtime = omp_get_wtime();
      #pragma omp parallel for
        for(i=0;i<n1-1;i++)
          my_value = my_value + (f(a1+h*i) + f(a1+h*(i+1)))*h/2;
      #pragma omp critical
        integralValue += my_value;
      endwtime = omp_get_wtime();
      #pragma omp barrier
      //cout << integralValue << endl;
      if (th_id == 0){
        parallelTime = endwtime-startwtime;
        cout<<"La integral de f es: "<< integralValue <<"\n\n";
        cout<<"El calculo fue en " << endwtime-startwtime << " en "<< num_th <<" hilos\n\n";
        fs << n <<" "<<num_th <<" " << serialTime <<" " << parallelTime <<" "<<serialTime/(parallelTime*num_th)<<" "<< serialTime/(parallelTime) << " N" <<endl;
      }
    }
    parallelTime = endwtime-startwtime;
    cout << "SpeedUp usando For: "<< serialTime/(parallelTime) <<endl;
    cout << "Eficiencia usando For: "<< serialTime/(parallelTime*num_th) <<endl;
  }
}

int main(int argc, char *argv[]){
  double a,b;
  int ns[][2] = {{100000, 4}, {200000, 4}, {300000, 4}, {100000, 8}, {200000, 8}, {300000, 8}, {100000, 16}, {200000, 16}, {300000, 16}, {100000, 32}, {200000, 32}, {300000, 32}, {100000, 64}, {200000, 64}, {300000, 64}}; 
  cout<<"\nIngrese intervalo: a b\n";
  cin>>a>>b;
  fs << "#Intervalos Hilos sec_t par_t eff speedup isTrapMethod" << endl;
  for (auto itera: ns) {
    openMPRun(2, itera, a, b);
  }
  return 0;
}
