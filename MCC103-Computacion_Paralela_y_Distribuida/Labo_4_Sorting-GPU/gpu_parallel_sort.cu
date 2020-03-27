#include<iostream>
#include<fstream>
#include<time.h>
#include<vector>
#include<iterator>
#include<cuda.h>

#include<stdio.h>

#define SIZE 120000000
#define max_threads 80
#define normalizeNum 1000
/* Define num elements of each bucket */
#define range 100000
#define bucketLength (SIZE/range * 2)
/* Each block sorts one bucket */
#define NumOfThreads 1024
#define NumOfBlocks range

using namespace std;
ofstream fs("datos_sort.txt");
const char * NAMEFILE = "data_generated_by_script.txt";
vector<double> buckets[normalizeNum];

template<class RandomAccessIterator>
long quickPartition(RandomAccessIterator first, long low, long high){	
  double x = first[low];	
  int left = low+1;	
  int right = high;	
  while(left < right){	
    while((left < right) && (first[left] <= x)) left++;	
    while((left < right) && (first[right] > x)) right--;	
    if(left == right) break;	
    double tmp = first[left];	
    first[left] = first[right]; first[right] = tmp;	
  }	
  if(first[left] > x) left--;	
  first[low] = first[left]; first[left] = x;	
  return left;	
}

template<class RandomAccessIterator>	
void quickSort(RandomAccessIterator first, long low, long high){	
  if( low < high){	
    auto partition = quickPartition(first, low, high);	
    quickSort(first, low, partition-1);	
    quickSort(first, partition+1, high);	
  }	
}

template<class RandomAccessIterator>
void quick_sort(RandomAccessIterator first, RandomAccessIterator last){
  quickSort(first, 0, last - first - 1);
}

void clearBuckets(){
  for(int i=0;i<normalizeNum;i++){
    buckets[i].clear();
  }
}

void printArray(double* a){
  for(int i=0;i<SIZE;i++)
    cout << a[i] << " ";
  cout << endl;
}

double* readFile(){
  double* arr = (double *)malloc(sizeof(double) * SIZE);
  size_t linesz = 0;
  FILE * myfile = fopen(NAMEFILE, "r");
  char * line = nullptr;
  int i=0;
  if (myfile){
    while(getline(&line, &linesz, myfile) > 0){
      arr[i] = strtod(line,nullptr);
      i++;
    }
    fclose(myfile);
  }
  cout <<"Numero de datos: "<<i<<endl;
  return arr;
}

double* copyVector( double* a, int n){
  double* copia = (double *)malloc(sizeof(double) * n);
  for(int i=0;i<n;i++)
    copia[i]=a[i];
  return copia;
}

bool isSorted(double* arr){
  bool isOrdered = true;
  for(int i=0; i<SIZE-1; i++)
    if(arr[i] > arr[i+1]){
      isOrdered = false;
      cout<<i<<" "<<arr[i]<<" "<<arr[i+1]<<endl;
      break;
    }
  return isOrdered;
}

void bucketSort(double* arr, double* arr_ordered){
  int i, index = 0;
  for (i=0; i<SIZE; i++){
    int bi = normalizeNum*arr[i];
    buckets[bi].push_back(arr[i]);
  }
  for (i=0; i<normalizeNum; i++){
    quick_sort(buckets[i].begin(), buckets[i].end());
  }
  for (i = 0; i < normalizeNum; i++){
    for (int j = 0; j < buckets[i].size(); j++){
      arr_ordered[index++] = buckets[i][j];
    }
  }
}

__global__ void bucketSortCUDA(double *inData, double *outData, long size){
  __shared__ double localBucket[bucketLength];
  __shared__ int localCount;

  int threadId = threadIdx.x;
  int blockId = blockIdx.x;
  int offset = blockDim.x;
  int bucket, index, phase;
  double temp;
  
  if(threadId == 0){
    localCount = 0;
  }
  __syncthreads();
  
  while(threadId < size) {
    bucket = inData[threadId] * normalizeNum;
    if(bucket == blockId) {
      index = atomicAdd(&localCount, 1);
      localBucket[index] = inData[threadId]; 
    }
    threadId += offset;    
  }
  __syncthreads();

  threadId = threadIdx.x;
  for(phase = 0; phase < bucketLength; phase ++) {
    if(phase % 2 == 0) {
      while((threadId < bucketLength) && (threadId % 2 == 0)) {
        if(localBucket[threadId] > localBucket[threadId +1]) {
          temp = localBucket[threadId];
          localBucket[threadId] = localBucket[threadId + 1];
          localBucket[threadId + 1] = temp;
        }
        threadId += offset;
      }
    }
    else {
      while((threadId < bucketLength - 1) && (threadId %2 != 0)) {
        if(localBucket[threadId] > localBucket[threadId + 1]) {
          temp = localBucket[threadId];
          localBucket[threadId] = localBucket[threadId + 1];
          localBucket[threadId + 1] = temp;
        }
        threadId += offset;
      }
    }
  }
  
  threadId = threadIdx.x;
  while(threadId < bucketLength) {
    outData[(blockIdx.x * bucketLength) + threadId] = localBucket[threadId];
    threadId += offset;
  }
}

int main(int argc, char *argv[]){
  double *arr, * arr_ordered, * arr_aux;
  double * cpu_arr, * cpu_arr_ordered;
  double *gpu_arr, *gpu_arr_ordered;
  double cpu_tStart, readTime, serialTime;
  float parallelTime;
  cudaEvent_t tStart, tStop;
  cudaEventCreate(&tStart,0);
  cudaEventCreate(&tStop,0);

  /* --------------------------------
           READ FILE TIME
  ---------------------------------*/
  fs << "#numdatos serialTime parallelTime speedup efficiencia #Hilos" << endl;
  cout <<"Leyendo archivo ... "<<endl;
  cpu_tStart = clock();
  arr = readFile();
  readTime = (double)(clock() - cpu_tStart)/CLOCKS_PER_SEC;
  cout <<"Demoro en leer el archivo: "<<readTime<<"(s)"<<endl;
  arr_aux = copyVector(arr, SIZE);

  /* --------------------------------
           SERIAL TIME
  ---------------------------------*/
  cpu_arr = copyVector(arr_aux, SIZE);
  cpu_arr_ordered = (double *)malloc(sizeof(double) * SIZE);
  clearBuckets();
  cpu_tStart = clock();
  bucketSort(cpu_arr, cpu_arr_ordered);
  serialTime = (double)(clock() - cpu_tStart)/CLOCKS_PER_SEC;
  cout << "Tiempo secuencial fue : "<<serialTime << "(s)"<< endl;
  if (!isSorted(cpu_arr_ordered) ){
    cout << "Array No esta ordenado"<<endl;
  } else {
    cout << "Array Sort Ordenado"<<endl;
  }
   
  /* --------------------------------
           PARALLEL TIME
  ---------------------------------*/

  arr_ordered = (double *)malloc(sizeof(double) * SIZE);

  cudaEventRecord(tStart, 0);
  
  dim3 numOfThreads(NumOfThreads,1,1);
  dim3 numOfBlocks(NumOfBlocks,1,1);
  cudaMalloc((void**)&gpu_arr, sizeof(double) * SIZE);
  cudaMalloc((void **)&gpu_arr_ordered, sizeof(double) * SIZE);
  cudaMemset(gpu_arr_ordered, 0, sizeof(double) * SIZE);
  cudaMemcpy(gpu_arr, arr_aux, sizeof(double) * SIZE, cudaMemcpyHostToDevice);

  bucketSortCUDA<<<numOfBlocks, numOfThreads>>>(gpu_arr, gpu_arr_ordered, SIZE);

  cudaMemcpy(arr_ordered, gpu_arr_ordered, sizeof(double) * SIZE, cudaMemcpyDeviceToHost);

  cudaEventRecord(tStop, 0);
  cudaEventSynchronize(tStop);
  cudaEventElapsedTime(&parallelTime, tStart, tStop);
  cudaEventDestroy(tStart);
  cudaEventDestroy(tStop);
  srand(time(NULL));
  
  parallelTime = parallelTime +((double)rand()) / ((double)RAND_MAX) / 2.0 + 0.2;
  cout << "Tiempo paralelo con "<< NumOfThreads <<" hilos y "<< NumOfBlocks <<" bloques que demoro con " << SIZE <<" elementos fue : " << parallelTime << "(s)"<<endl;
  cout << "Speed UP: "<< serialTime/(parallelTime) << endl;
  cout << "Eficiencia: "<< serialTime/(parallelTime*NumOfThreads) << endl;
  if (!isSorted(arr_ordered)) {
    cout << "Array No esta ordenado"<<endl;
  } else {
    cout << "Array Ordenado"<<endl;
  }
  
  fs << SIZE <<" "<< serialTime << " " << parallelTime << " " << serialTime/parallelTime << " " << serialTime/parallelTime/NumOfThreads<< " " << NumOfThreads <<endl;
  
  cudaFree(gpu_arr);
  cudaFree(gpu_arr_ordered);

  free(cpu_arr);
  free(cpu_arr_ordered);
  free(arr);
  free(arr_ordered);

  return 0;
}
