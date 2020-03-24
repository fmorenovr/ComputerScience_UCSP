#include<iostream>
#include<fstream>
#include<omp.h>

#define IMAGE_LENGTH 2000
#define KERNEL_LENGTH 5
#define MAX_NUMBER 12

using namespace std;
ofstream fs("convolucion.txt");

void print_Matrix(int** matrix, int n, int m){
  for(int i = 0; i < n; i++){
    for(int j = 0; j < m; j++){
      cout<<" "<<matrix[i][j];            
    }
    cout<<endl;
  }
}

int** create_Matrix(int n, int m){
  int **matrix;
  matrix = (int **)malloc(n*sizeof(int *));
  for(int i=0; i < n; i++) {
    matrix[i] = (int *)malloc(m*sizeof(int));
  }
  return matrix;
}

void generate_Kernel_Matrix(int** a, int n, int m){
  srand( (unsigned)time( NULL ) );
  for(int i=0; i<n;i++){
    for(int j=0;j<m;j++){
      //a[i][j] = (rand()%MAX_NUMBER)+1;;
      a[i][j] = 1;
    }
  }
}

void generate_Image_Matrix(int** a, int n, int m, int index){
  srand( (unsigned)time( NULL ) );
  for(int i=0;i<index;i++){
    for(int j=0;j<m;j++){
      a[i][j] = 0;
    }
  }
  for(int i=0;i<n-index;i++){
    for(int j=0;j<index;j++){
      a[i][j] = 0;
    }
  }
  for(int i=n-index;i<n;i++){
    for(int j=0;j<m;j++){
      a[i][j] = 0;
    }
  }
  for(int i=0;i<n;i++){
    for(int j=m-index;j<m;j++){
      a[i][j] = 0;
    }
  }
  for(int i=index; i<n-index;i++){
    for(int j=index;j<m-index;j++){
      a[i][j] = (rand()%MAX_NUMBER)+1;
      //a[i][j] = j;
    }
  }
}

void init_Matrix(int** a,int n, int m){
  for(int i=0; i<n;i++){
    for(int j=0;j<m;j++){
      a[i][j]=0;
    }
  }
}

bool compare_Matrix(int** A, int** B, int n, int m){
  bool same = true;
  for(int i=0; i<n;i++){
    for(int j=0;j<m;j++){
      if(A[i][j]!=B[i][j]){
        same = false;
        break;
      }
    }
  }
  return same;
}

int sumTermsMatrix(int** a, int n, int m){
  int suma = 0;
  for(int i=0; i<n;i++){
    for(int j=0;j<m;j++){
      suma+=a[i][j];
    }
  }
  return suma;
}

void convolucionOMP(int** kernel, int** image, int** result, int KERNELCOUNT, int numThreads){
  int i, j, n, m;
  int acumulador=0;

# pragma omp parallel for shared(result) private(i,j,n,m) firstprivate(acumulador) schedule(dynamic) num_threads(numThreads)
    for (i=0; i < IMAGE_LENGTH; i++){
      for (j=0; j < IMAGE_LENGTH; j++){
        for (n = 0; n < KERNEL_LENGTH; n++){
          for (m = 0; m < KERNEL_LENGTH; m++){
            acumulador += image[i + n][j + m] * kernel[n][m];
          }
        }
        result[i][j] = acumulador/KERNELCOUNT;
        acumulador = 0;
      }
    }
}

void convolucion(int** kernel, int** image, int** result, int KERNELCOUNT){
  int i, j, n, m;
  int acumulador=0;
  
  for (i=0; i < IMAGE_LENGTH; i++){
    for (j=0; j < IMAGE_LENGTH; j++){
      for (n = 0; n < KERNEL_LENGTH; n++){
        for (m = 0; m < KERNEL_LENGTH; m++){
          acumulador += image[i + n][j + m] * kernel[n][m];
        }
      }
      result[i][j] = acumulador/KERNELCOUNT;
      acumulador = 0;
    }
  }
}

int main(int argc, char** argv) {
  double tStart;
  double serialTime, parallelTime;
  int realKernelLength=0, index = 0;
  if(KERNEL_LENGTH%2!=0){
    realKernelLength = KERNEL_LENGTH - 1;
    index = KERNEL_LENGTH/2;
  } else{
    cout << "Matrix Convolution siempre debe ser 2*N+1"<<endl;
    exit(0);
  }
  int** kernel = create_Matrix(KERNEL_LENGTH, KERNEL_LENGTH);
  int** image = create_Matrix(IMAGE_LENGTH+realKernelLength, IMAGE_LENGTH+realKernelLength);
  int** result = create_Matrix(IMAGE_LENGTH, IMAGE_LENGTH);
  int** resultOMP = create_Matrix(IMAGE_LENGTH, IMAGE_LENGTH);

  generate_Kernel_Matrix(kernel, KERNEL_LENGTH,KERNEL_LENGTH);
  generate_Image_Matrix(image, IMAGE_LENGTH+realKernelLength, IMAGE_LENGTH+realKernelLength, index);
  init_Matrix(result, IMAGE_LENGTH, IMAGE_LENGTH);
  init_Matrix(resultOMP, IMAGE_LENGTH, IMAGE_LENGTH);
  
  int KERNELCOUNT = sumTermsMatrix(kernel,KERNEL_LENGTH,KERNEL_LENGTH);
  
  tStart = omp_get_wtime();
  convolucion(kernel, image, result,KERNELCOUNT);
  serialTime = (double)(omp_get_wtime() - tStart);

  if(IMAGE_LENGTH<10){
    cout << "Image: "<<endl;
    print_Matrix(image, IMAGE_LENGTH+realKernelLength, IMAGE_LENGTH+realKernelLength);
    cout << endl << endl;
    
    cout << "Kernel: "<<endl;
    print_Matrix(kernel, KERNEL_LENGTH, KERNEL_LENGTH);
    cout << endl << endl;

    cout << "Result: "<<endl;
    print_Matrix(result, IMAGE_LENGTH, IMAGE_LENGTH);
  }

  cout << "El tiempo en realizar la convolución en tiempo secuencial es: " << serialTime << endl;
  
  int ns[] = {1, 2, 4, 8};
  
  fs << "ImgLenght KerLenght numthd serialTime parallelTime eff speedUp"<< endl;
  
  for (auto num_threads: ns){
    init_Matrix(resultOMP, IMAGE_LENGTH, IMAGE_LENGTH);
    tStart = omp_get_wtime();
    convolucionOMP(kernel, image, resultOMP,KERNELCOUNT,num_threads);
    parallelTime = (double)(omp_get_wtime() - tStart);
    
    if(IMAGE_LENGTH<20){
      print_Matrix(resultOMP, IMAGE_LENGTH, IMAGE_LENGTH);
    }
    
    cout << "Son iguales: "<< compare_Matrix(result, resultOMP,IMAGE_LENGTH, IMAGE_LENGTH) << endl;

    cout << "El tiempo en realizar la convolución en tiempo Paralelo con " << num_threads <<" es: " << parallelTime << endl;    
    cout << "Speed UP: "<< serialTime/(parallelTime) << endl;
    cout << "Eficiencia: "<< serialTime/(parallelTime*num_threads) << endl;
    
    fs << IMAGE_LENGTH << " " << KERNEL_LENGTH << " " << num_threads <<" " << serialTime <<" " << parallelTime << " " << serialTime/(parallelTime*num_threads)<<" "<< serialTime/(parallelTime) << endl;
  }
  
  free(image);
  free(result);
  free(kernel);
  
  return 0;
}
