#include<stdio.h>
#include<iostream>
#include<fstream>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>

#define IMAGE_LENGTH 2000
#define KERNEL_LENGTH 5
#define MAX_NUMBER 12
#define NumOfBlocks IMAGE_LENGTH/16
#define NumOfThreads 16

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

int** gpu_create_Matrix(int n, int m){
  int **matrix;
  cudaMalloc((void***)&matrix,  n*sizeof(int*));
  for(int i=0; i < n; i++) {
    cudaMalloc((void**) &(matrix[i]), m*sizeof(int));
  }
  return matrix;
}

int** gpu_copy_Matrix(int **host, int n, int m){
  int** device = (int **)malloc(n * sizeof(int *));
  int **aux = (int **)malloc(n * sizeof(int *));
  cudaMalloc((void***)&device,  n*sizeof(int*));
  for(int i=0; i<n; i++) {
    cudaMalloc((void**) &(aux[i]), m*sizeof(int));
    cudaMemcpy (aux[i], host[i], m*sizeof(float), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(device, aux, n*sizeof(int *), cudaMemcpyHostToDevice);
  return device;
}

int** cpu_copy_Matrix(int **device, int n, int m){
  int** host = (int **)malloc(n * sizeof(int *));
  int **aux = (int **)malloc(n * sizeof(int *));
  cudaMalloc((void***)&host,  n*sizeof(int*));
  for(int i=0; i<n; i++) {
    cudaMalloc((void**) &(aux[i]), m*sizeof(int));
    cudaMemcpy (aux[i], device[i], m*sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(host, aux, n*sizeof(int *), cudaMemcpyDeviceToHost);
  return host;
}

int** copy_Matrix(int **orig, int n, int m){
  int** cpy = create_Matrix(n,m);
  for(int i=0; i<n; i++) {
    for(int j=0; i<m; i++){
      cpy[i][j] = orig[i][j];
    }
  }
  return cpy;
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

__global__ void convolucionCUDA(int** kernel, int** image, int** result, int KERNELCOUNT){
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

__global__ void convKernel(int *inData, int *filter, int dataCol, int dataRow, int filRowRad, int filColRad,
               int *outData)
{
  __shared__ int padRect[2*1024];
  int i, col, row, sum = 0;

  int globalCol = threadIdx.x + blockIdx.x * blockDim.x;
  int globalRow = threadIdx.y + blockIdx.y * blockDim.y;
  int globalIdx = globalCol * dataRow + globalRow;

  int localIdx = threadIdx.x * blockDim.y + threadIdx.y;
  int localCells = blockDim.x * blockDim.y;

  int padRectCol = threadIdx.x + filColRad;
  int padRectRow = threadIdx.y + filRowRad;
  int padRectOffset = 2*filRowRad + blockDim.y;
  int padRectCells = padRectOffset * (blockDim.x + 2*filColRad);

  int *padRectOut = (int*)&padRect[((padRectCells-1)/32 + 1) * 32]; //Padding up with 32
  padRectOut[localIdx] = 0;

  int filOffset = filRowRad*2 + 1;
  int filCells = filOffset * (filColRad*2 + 1);
  int *localFilter = (int *)&padRectOut[((localCells-1)/32 + 1) * 32]; //Padding up with 32

  // Copying the filter elements to shared memory
  for(i = 0; i < (filCells/localCells) + 1; i++) {
    int index = i*localCells + localIdx;
    if(index < filCells) {
      localFilter[index] = filter[index];
    }
  }

  // Copying the Data elements to padded shared memory
  for(i = 0; i < (padRectCells/localCells) + 1; i++) {
    int index = i*localCells + localIdx;
    if(index < padRectCells) {
      int prCol = index / padRectOffset;
      int prRow = index % padRectOffset;
      int glCol = prCol + blockIdx.x*blockDim.x - filColRad;
      int glRow = prRow + blockIdx.y*blockDim.y - filRowRad;
      int glIdx = glCol * dataRow + glRow;
      if(glRow >= 0 && glRow < dataRow && glCol >= 0 && glCol < dataCol)
        padRect[index] = inData[glIdx];
      else
        padRect[index] = 0;
    }
  }

  __syncthreads();

  //Taking Convolution
  for(col = -filColRad; col <= filColRad; col++) {
    for(row = -filRowRad; row <= filRowRad; row++) {
      int filCol = filColRad - col;
      int filRow = filRowRad - row;
      int filIdx = filCol*filOffset + filRow;
      int filVal = localFilter[filIdx];

      int prCol = padRectCol + col;
      int prRow = padRectRow + row;
      int prIdx = prCol*padRectOffset + prRow;
      sum += filVal * padRect[prIdx];
    }
  }
  
  padRectOut[localIdx] = sum;
  __syncthreads();

  outData[globalIdx] = padRectOut[localIdx];


}

int main(){
  float parallelTime, serialTime;
  cudaEvent_t tStart, tStop;
  cudaEventCreate(&tStart, 0);
  cudaEventCreate(&tStop, 0);
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
  int** resultCUDA = create_Matrix(IMAGE_LENGTH, IMAGE_LENGTH);

  generate_Kernel_Matrix(kernel, KERNEL_LENGTH,KERNEL_LENGTH);
  generate_Image_Matrix(image, IMAGE_LENGTH+realKernelLength, IMAGE_LENGTH+realKernelLength, index);
  init_Matrix(result, IMAGE_LENGTH, IMAGE_LENGTH);
  
  int KERNELCOUNT = sumTermsMatrix(kernel,KERNEL_LENGTH,KERNEL_LENGTH);
  
  cudaEventRecord(tStart, 0);
  convolucion(kernel, image, result, KERNELCOUNT);
  cudaEventRecord(tStop, 0);
  cudaEventSynchronize(tStop);
  cudaEventElapsedTime(&serialTime, tStart, tStop);
  cudaEventDestroy(tStart);
  cudaEventDestroy(tStop);

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

  cout << "El tiempo en realizar la convolución en tiempo secuencial es: " << serialTime/1000 << endl;
  
//  int ns[] = {1, 2, 4, 8};
  
  fs << "ImgLenght KerLenght numthd serialTime parallelTime eff speedUp"<< endl;
  
  // ----
  int** gpu_kernel = copy_Matrix(kernel, KERNEL_LENGTH, KERNEL_LENGTH);
  int** gpu_image = copy_Matrix(image, IMAGE_LENGTH+realKernelLength, IMAGE_LENGTH+realKernelLength);
  init_Matrix(resultCUDA, IMAGE_LENGTH, IMAGE_LENGTH);
  int** gpu_result = copy_Matrix(resultCUDA, IMAGE_LENGTH, IMAGE_LENGTH);
    
  cudaEventCreate(&tStart, 0);
  cudaEventCreate(&tStop, 0);

  cudaEventRecord(tStart, 0);

  dim3 num_threads(NumOfThreads, 16, 1);
  dim3 numOfBlocks(NumOfBlocks, 1, 1);
  //convolucionCUDA<<<numOfBlocks, num_threads>>>(gpu_image, gpu_kernel, gpu_result, KERNELCOUNT);
  cudaEventRecord(tStop, 0);
  cudaEventSynchronize(tStop);
  cudaEventElapsedTime(&parallelTime, tStart, tStop);
  cudaEventDestroy(tStart);
  cudaEventDestroy(tStop);

  //resultCUDA = cpu_copy_Matrix(gpu_result, IMAGE_LENGTH, IMAGE_LENGTH);
  
  cout << "Son iguales: "<< compare_Matrix(result, resultCUDA, IMAGE_LENGTH, IMAGE_LENGTH) << endl;

  cout << "El tiempo en realizar la convolución en tiempo Paralelo con " << NumOfBlocks << " bloques y " << NumOfThreads <<" hilos es: " << parallelTime << endl;    
  cout << "Speed UP: "<< serialTime/(parallelTime) << endl;
  cout << "Eficiencia: "<< serialTime/(parallelTime*NumOfThreads) << endl;
  
  fs << IMAGE_LENGTH << " " << KERNEL_LENGTH << " " << NumOfThreads <<" " << serialTime <<" " << parallelTime << " " << serialTime/(parallelTime*NumOfThreads)<<" "<< serialTime/(parallelTime) << endl;

  cudaFree(gpu_image);
  cudaFree(gpu_result);
  cudaFree(gpu_kernel);
  
  free(image);
  free(result);
  free(kernel);
  
  return 0;
}
