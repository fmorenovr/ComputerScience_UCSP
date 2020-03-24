#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <stdlib.h>

#define SEL_LENGTH 3000
#define MAXNUMBER 1000

using namespace std;
ofstream Jfs("Jacobi.txt");
ofstream GSfs("Gauss-Seidel.txt");
ofstream Gfs("Gauss.txt");
ofstream LUfs("LU.txt");

void print_Matrix(float** matrix, int n, int m){
  for(int i = 0; i < n; i++){
    for(int j = 0; j < m; j++){
      printf(" %0.00lf", matrix[i][j]);
    }
    printf("\n");
  }
}

void print_Vector(float* vector, int n){
  for(int i = 0; i < n; i++){
    printf(" %lf", vector[i]);
  }
  printf("\n");
}

float** create_Matrix(int n, int m){
  float **matrix;
  //allocate space for values at once, so as to be contiguous in memory
  matrix = (float **)malloc(n*sizeof(float *));
  matrix[0] = (float *)malloc(m*sizeof(float));
  for(int i=1; i < n; i++) {
      //matrix[i]=&matrix[0][m*i];
      matrix[i] = (float *)malloc(m*sizeof(float));
  }
  return matrix;
}

float* create_Vector(int n){
  float *vector;
  //allocate space for values at once, so as to be contiguous in memory
  vector = (float *)malloc(n*sizeof(float));
  return vector;
}

void init_Vector(float* x,int n){
  for(int i=0; i<n;i++){
    x[i]=1;
  }
}

float** generate_Matrix(int n, int m){
  float** A = create_Matrix(n,m);
  srand( (unsigned)time( NULL ) );
 
  for(int i=0; i<n;i++){
    for(int j=0;j<m;j++){
      A[i][j] = (rand()%MAXNUMBER)+1;
    }
  }
  return A;
}

float* generate_Vector(int n){
  float* x = create_Vector(n);
  srand( (unsigned)time( NULL ) );
  for(int i=0; i<n;i++){
    x[i] = (rand()%MAXNUMBER)+1;
  }
  return x;
}

float* set_Vector(int n){
  float * v = create_Vector(n);
  init_Vector(v,n);
  return v;
}

float normInf(float* x, float* x_old, int n){
  float norm = abs(x[0] - x_old[0]);
  float temp;
  for(int i =1; i<n;i++){
    temp = abs(x[i] - x_old[i]);
    if(temp > norm){
      norm = temp;
    }
  }
  return norm;
}

float* copyVector(float* x, int n){
  float * v = create_Vector(n);
  for(int i=0; i<n;i++)
    v[i] = x[i];
  return v;
}

void copyVector(float* a, float* b, int n){
  for(int i=0;i<n;i++)
    a[i]=b[i];
}

float dotVector(float* a, float* b, int n){
  float sum =0;
  for(int i=0;i<n;i++){
    sum += a[i]*b[i];
  }
}

float** ExtendMatrixVector(float** A, float* b, int n, int m){
  float** R = create_Matrix(n,m+1);
  for(int i=0; i<n;i++){
    for(int j=0;j<m;j++){
      R[i][j] = A[i][j];
    }
  }
  for(int i=0;i<n;i++){
    R[i][m] = b[i];
  }
  return R;
}

void LUDecomposition(float** A, float* b, float* x, int n) {

    int* P = (int *)malloc(n*sizeof(int));

    int i, j, k, imax; 
    float maxA, *ptr, absA;

    for (i = 0; i <= n; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < n; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < n; k++)
            if ((absA = abs(A[k][i])) > maxA) { 
                maxA = absA;
                imax = k;
            }

        if (imax != i) {
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;
            ptr = A[i];
            A[i] = A[imax];
            A[imax] = ptr;
            P[n]++;
        }

        for (j = i + 1; j < n; j++) {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < n; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
    }
    
    for (int i = 0; i < n; i++) {
        x[i] = b[P[i]];

        for (int k = 0; k < i; k++)
            x[i] -= A[i][k] * x[k];
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int k = i + 1; k < n; k++)
            x[i] -= A[i][k] * x[k];
        x[i] = x[i] / A[i][i];
    }
}

int LUDecompositionOpenMP(float** A, float* b, float* x, int n, int numThreads) {

    int* P = (int *)malloc(n*sizeof(int));

    int i; 

#   pragma omp parallel for default(none) private(i) shared(P,n) schedule(dynamic) num_threads(numThreads)
    for (i = 0; i <= n; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N

#   pragma omp parallel for default(none) private(i) shared(A,P,n) schedule(dynamic) num_threads(numThreads)
      for (i = 0; i < n; i++) {
        int maxA = 0.0;
        int imax = i;

        for (int k = i; k < n; k++) {
          float absA = abs(A[k][i]);
          if ( absA > maxA) { 
            maxA = absA;
            imax = k;
          }
        }
        if (imax != i) {
          int tmp = P[i];
          P[i] = P[imax];
          P[imax] = tmp;
          float* ptr = A[i];
          A[i] = A[imax];
          A[imax] = ptr;
          P[n]++;
        }

        for (int j = i + 1; j < n; j++) {
          A[j][i] /= A[i][i];
          for (int k = i + 1; k < n; k++){
            A[j][k] -= A[j][i] * A[i][k];
          }
        }
      }
    
#   pragma omp parallel for default(none) private(i) shared(A, b, P, x, n) schedule(dynamic) num_threads(numThreads)
    for (i = 0; i < n; i++) {
      x[i] = b[P[i]];
      for (int k = 0; k < i; k++)
        x[i] -= A[i][k] * x[k];
    }

#   pragma omp parallel for default(none) private(i) shared(A, x, n) schedule(dynamic) num_threads(numThreads)
    for (i = n - 1; i >= 0; i--) {
      for (int k = i + 1; k < n; k++)
        x[i] -= A[i][k] * x[k];
      x[i] = x[i] / A[i][i];
    }
}

void GaussElimination(float** A, float* x, int n){
  for (int i=0; i<n; i++) {
    // Search for maximum in this column
    float maxEl = abs(A[i][i]);
    int maxRow = i;
    for (int k=i+1; k<n; k++) {
      if (abs(A[k][i]) > maxEl) {
        maxEl = abs(A[k][i]);
        maxRow = k;
      }
    }
    // Swap maximum row with current row (column by column)
    for (int k=i; k<n+1;k++) {
      float tmp = A[maxRow][k];
      A[maxRow][k] = A[i][k];
      A[i][k] = tmp;
    }
    // Make all rows below this one 0 in current column
    for (int k=i+1; k<n; k++) {
      float c = -A[k][i]/A[i][i];
      for (int j=i; j<n+1; j++) {
        if (i==j) {
          A[k][j] = 0;
        } else {
          A[k][j] += c * A[i][j];
        }
      }
    }
  }
  // Solve equation Ax=b for an upper triangular matrix A
  for (int i=n-1; i>=0; i--) {
    x[i] = A[i][n]/A[i][i];
    for (int k=i-1;k>=0; k--) {
      A[k][n] -= A[k][i] * x[i];
    }
  }
}

void GaussEliminationOpenMP(float** A, float* x, int n, int numThreads){
  int i;
#   pragma omp parallel for default(none) private(i) shared(A,n) schedule(dynamic) num_threads(numThreads)
  for (i=0; i<n; i++) {
    // Search for maximum in this column
    float maxEl = abs(A[i][i]);
    int maxRow = i;
    for (int k=i+1; k<n; k++) {
      if (abs(A[k][i]) > maxEl) {
        maxEl = abs(A[k][i]);
        maxRow = k;
      }
    }
    // Swap maximum row with current row (column by column)
    for (int k=i; k<n+1;k++) {
      float tmp = A[maxRow][k];
      A[maxRow][k] = A[i][k];
      A[i][k] = tmp;
    }
    // Make all rows below this one 0 in current column
    for (int k=i+1; k<n; k++) {
      float c = -A[k][i]/A[i][i];
      for (int j=i; j<n+1; j++) {
        if (i==j) {
          A[k][j] = 0;
        } else {
          A[k][j] += c * A[i][j];
        }
      }
    }
  }
  // Solve equation Ax=b for an upper triangular matrix A
#   pragma omp parallel for default(none) private(i) shared(A,x,n) schedule(dynamic) num_threads(numThreads)
  for (i=n-1; i>=0; i--) {
    x[i] = A[i][n]/A[i][i];
    for (int k=i-1;k>=0; k--) {
      A[k][n] -= A[k][i] * x[i];
    }
  }
}

void GaussSeidel(float** A, float* b, float* x, float Err, int n){
  int i;
  float* x_old = create_Vector(n), dotVect, err=1;
  while (err > Err){
    copyVector(x_old,x,n);
    for(i = 0; i < n; i++ ){
      dotVect = dotVector(A[i],x,n);
      x[i] = (b[i] - dotVect) / A[i][i] + x[i];
    }
    err = normInf(x,x_old,n);
  }
}


void GaussSeidelOpenMP(float** A, float* b, float* x, float Err, int n, int numThreads){
  int i;
  float* x_old = create_Vector(n), dotVect, err=1;
  while (err > Err){
    copyVector(x_old,x,n);
#   pragma omp parallel for default(none) private(i,dotVect) shared(A,b,x,n,x_old) schedule(dynamic) num_threads(numThreads)
    for(i = 0; i < n; i++ ){
      dotVect = dotVector(A[i],x,n);
      x[i] = (b[i] - dotVect) / A[i][i] + x[i];
    }
    err = normInf(x,x_old,n);
  }
}

void Jacobi(float** A, float* b, float* x, float Err, int n){
  int i;
  float* x_old = create_Vector(n), dotVect, err=1;
  while (err > Err){
    copyVector(x_old, x, n);
    for(i = 0; i < n; i++){
      dotVect = dotVector(A[i], x_old, n);
      x[i] = (b[i] - dotVect) / A[i][i] + x_old[i];
    }
    err = normInf(x, x_old, n);
  }
}

void JacobiOpenMP(float** A, float* b, float* x, float Err, int n, int numThreads){
  int i;
  float* x_old = create_Vector(n), dotVect, err=1;
  while(err>Err){
    copyVector(x_old, x, n);
#   pragma omp parallel for default(none) private(i, dotVect) shared(A,b,x,n,x_old) schedule(dynamic) num_threads(numThreads)
    for(i = 0; i < n; i++){
      dotVect = dotVector(A[i], x_old, n);
      x[i] = (b[i] - dotVect) / A[i][i] + x_old[i];
    }
    err = normInf(x, x_old, n);
  }
}

int main(){
  //float x[] = {1.0, 2.0, 2.0};
  //float b[] = {7.0, -21.0,15.0};
  //float A[][3] = {{4.0,-1.0, 1.0}, {4.0, -8.0, 1.0}, {-2.0, 1.0, 5.0}};
  float * b = generate_Vector(SEL_LENGTH);
  float* x = set_Vector(SEL_LENGTH);
  float* Jx = copyVector(x,SEL_LENGTH);
  float* GSx = copyVector(x, SEL_LENGTH);
  float* Gx = copyVector(x, SEL_LENGTH);
  float* LUx = copyVector(x, SEL_LENGTH);
  float** A = generate_Matrix(SEL_LENGTH, SEL_LENGTH);
  
  float Err = 0.0001;
  
  double startTime = omp_get_wtime(), parallelTime;
  GaussSeidel(A,b,GSx, Err, SEL_LENGTH);
  double serialGSTime = omp_get_wtime() - startTime;
  cout << "El tiempo de Gauss-Seidel serial es: " << serialGSTime << endl;
  GSfs << serialGSTime << endl;
  
  startTime = omp_get_wtime();
  Jacobi(A,b,Jx,Err, SEL_LENGTH);
  double serialJTime = omp_get_wtime() - startTime;
  cout << "El tiempo de Jacobi serial es: " << serialJTime << endl;
  Jfs << serialJTime << endl;
  
  startTime = omp_get_wtime();
  float** Ab = ExtendMatrixVector(A, b, SEL_LENGTH, SEL_LENGTH+1);
  GaussElimination(Ab,Gx,SEL_LENGTH);
  double serialGTime = omp_get_wtime() - startTime;
  cout << "El tiempo de Gauss serial es: " << serialGTime << endl;
  Gfs << serialGTime << endl;
  
  startTime = omp_get_wtime();
  LUDecomposition(A, b,LUx,SEL_LENGTH);
  double serialLUTime = omp_get_wtime() - startTime;
  cout << "El tiempo de LU serial es: " << serialLUTime << endl;
  LUfs << serialLUTime << endl;
  
  float* PJx, * PGSx, * PGx, * PLUx;
  
  int ns[] = {2, 4, 8};
  
  for (auto num_threads: ns){
  
    PGSx = copyVector(x, SEL_LENGTH);
    PJx = copyVector(x, SEL_LENGTH);
    PGx = copyVector(x, SEL_LENGTH);
  
    startTime = omp_get_wtime();
    GaussSeidelOpenMP(A,b,PGSx,Err, SEL_LENGTH, num_threads);
    parallelTime = omp_get_wtime() - startTime;
    cout << "El tiempo en realizar Gauss Seidel Paralelo con " << num_threads <<" es: " << parallelTime << endl;    
    cout << "Speed UP: "<< serialGSTime/(parallelTime) << endl;
    cout << "Eficiencia: "<< serialGSTime/(parallelTime*num_threads) << endl;
    GSfs << parallelTime << " " << num_threads << " " << serialGSTime/(parallelTime) << " " << serialGSTime/(parallelTime*num_threads) << endl;
    
    startTime = omp_get_wtime();
    JacobiOpenMP(A,b,PJx,Err, SEL_LENGTH, num_threads);
    parallelTime = omp_get_wtime() - startTime;
    cout << "El tiempo en realizar Jacobi Paralelo con " << num_threads <<" es: " << parallelTime << endl;    
    cout << "Speed UP: "<< serialJTime/(parallelTime) << endl;
    cout << "Eficiencia: "<< serialJTime/(parallelTime*num_threads) << endl;
    Jfs << parallelTime << " " << num_threads << " " << serialJTime/(parallelTime) << " " << serialJTime/(parallelTime*num_threads) << endl;
  
    Ab = ExtendMatrixVector(A, b, SEL_LENGTH, SEL_LENGTH+1);
    startTime = omp_get_wtime();
    GaussEliminationOpenMP(Ab,PGx, SEL_LENGTH, num_threads);
    parallelTime = omp_get_wtime() - startTime;
    cout << "El tiempo en realizar Gauss Paralelo con " << num_threads <<" es: " << parallelTime << endl;    
    cout << "Speed UP: "<< serialGTime/(parallelTime) << endl;
    cout << "Eficiencia: "<< serialGTime/(parallelTime*num_threads) << endl;
    Gfs << parallelTime << " " << num_threads << " " << serialGTime/(parallelTime) << " " << serialGTime/(parallelTime*num_threads) << endl;
  
    startTime = omp_get_wtime();
    LUDecompositionOpenMP(A, b,PGx, SEL_LENGTH, num_threads);
    parallelTime = omp_get_wtime() - startTime;
    cout << "El tiempo en realizar LU Paralelo con " << num_threads <<" es: " << parallelTime << endl;    
    cout << "Speed UP: "<< serialLUTime/(parallelTime) << endl;
    cout << "Eficiencia: "<< serialLUTime/(parallelTime*num_threads) << endl;
    LUfs << parallelTime << " " << num_threads << " " << serialLUTime/(parallelTime) << " " << serialLUTime/(parallelTime*num_threads) << endl;
  
  }
  
  return 0;
}
