#include<iostream>
#include<fstream>
#include<time.h>
#include<vector>
#include<iterator>
#include<omp.h>

#define SIZE 120000000
#define max_threads 80
#define normalizeNum 1000

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

void bucketSortOpenMP(double* arr, double* arr_ordered, int num_threads){
  omp_set_num_threads(num_threads);
  int i, index=0;
  for (i=0; i<SIZE; i++){
    int bi = normalizeNum*arr[i];
    buckets[bi].push_back(arr[i]);
  }
  double tStart = omp_get_wtime();
  #pragma omp parallel for private(i) shared(buckets) schedule(dynamic,1)
    for (i=0; i<normalizeNum; i++){
      quick_sort(buckets[i].begin(), buckets[i].end());
    }
  cout<<"fase 2 tomo: " << (double)(omp_get_wtime() - tStart) << "(s)"<<endl;
  for (i = 0; i < normalizeNum; i++){
    for (int j = 0; j < buckets[i].size(); j++){
      arr_ordered[index++] = buckets[i][j];
    }
  }
}

int main(int argc, char *argv[]) {
  double* arr, * arr_ordered,* arr_aux;
  double tStart, readTime, serialTime, parallelTime;
  
  /* --------------------------------
           READ FILE TIME
  ---------------------------------*/
  fs << "#numdatos serialTime parallelTime speedup efficiencia #Hilos" << endl;
  cout <<"Leyendo archivo ... "<<endl;
  tStart = clock();
  arr = readFile();
  readTime = (double)(clock() - tStart)/CLOCKS_PER_SEC;
  cout <<"Demoro en leer el archivo: "<<readTime<<"(s)"<<endl;
	arr_aux = copyVector(arr, SIZE);
	
	/* --------------------------------
           SERIAL TIME
  ---------------------------------*/
  arr_ordered = copyVector(arr, SIZE);
  clearBuckets();
  tStart = clock();
  bucketSort(arr_aux, arr_ordered);
  serialTime = (double)(clock() - tStart)/CLOCKS_PER_SEC;
  cout << "Tiempo secuencial fue : "<<serialTime << "(s)"<< endl;
  if (!isSorted(arr_ordered) ){
    cout << "Array No esta ordenado"<<endl;
  } else {
    cout << "Array Sort Ordenado"<<endl;
  }
	
	/* --------------------------------
           PARALLEL TIME
  ---------------------------------*/
	
	int ns[] = {4, 8};
  for (auto num_threads: ns){
	  arr_ordered = copyVector(arr, SIZE);
	  clearBuckets();
    tStart = omp_get_wtime();
    bucketSortOpenMP(arr_aux, arr_ordered, num_threads);
    parallelTime = (double)(omp_get_wtime() - tStart);
    
    cout << "Tiempo paralelo con "<<num_threads <<" hilos que demoro con " << SIZE <<" elementos fue : " << parallelTime << "(s)"<< endl;
    cout << "Speed UP: "<< serialTime/(parallelTime) << endl;
    cout << "Eficiencia: "<< serialTime/(parallelTime*num_threads) << endl;
    fs << SIZE<<" "<< serialTime << " " << parallelTime << " " << serialTime/parallelTime << " " << serialTime/parallelTime/num_threads<< " " << num_threads <<endl;
    if (!isSorted(arr_ordered) ){
      cout << "Array No esta ordenado"<<endl;
    } else {
      cout << "Array Sort Ordenado"<<endl;
    }
  }
  free(arr);
  free(arr_ordered);
  free(arr_aux);
}
