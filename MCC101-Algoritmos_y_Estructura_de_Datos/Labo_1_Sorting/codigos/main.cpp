#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;
// inser_sort
//sorted result is returned as a new array
//assume array is populated with data prior to this call
template<class RandomAccessIterator>
void insert_sort(RandomAccessIterator first, RandomAccessIterator last){
  RandomAccessIterator Index_I, Index_J;
  for (Index_I = first; Index_I != last; Index_I++){
    for (Index_J = Index_I; Index_J != first; Index_J--){
      if(*(Index_J-1) > *(Index_J)){
        swap(*Index_J, *(Index_J-1));
      } else {
        break;
      }
    }
  }
}

template< class RandomAccessIterator>
void merge( RandomAccessIterator first, RandomAccessIterator last, long low, long mid, long high){
  int i, j, k;
  int n1 = mid - low + 1;
  int n2 =  high - mid;
  vector<int> L(n1), H(n2);  // create temp arrays
  // Copy data to temp arrays Low[] and High[]
  for (i = 0; i < n1; i++)
    L.at(i) = *(first+low + i);
  for (j = 0; j < n2; j++)
    H.at(j) = *(first+mid + 1+ j);
  // Merge the temp arrays back into arr[low..high]
  i = 0; // Initial index of first subarray
  j = 0; // Initial index of second subarray
  k = low; // Initial index of merged subarray
  while (i < n1 && j < n2){
    if (L.at(i) <= H.at(j)){
      *(first+k) = L.at(i);
      i++;
    } else{
      *(first+k) = H.at(j);
      j++;
    }
    k++;
  }
  // Copy the remaining elements of L[], if there are any
  while (i < n1){
    *(first+k) = L.at(i);
    i++;
    k++;
  }
  // Copy the remaining elements of R[], if there are any
  while (j < n2){
    *(first+k) = H.at(j);
    j++;
    k++;
  }
}

template<class RandomAccessIterator>
void mergeSort(RandomAccessIterator first, RandomAccessIterator last, long low, long high){
  if( high <= low )
    return;
  long mid = low + ( high - low ) / 2;
  mergeSort( first, last, low, mid);
  mergeSort( first, last, mid + 1, high);
  merge( first, last, low, mid, high);
}

//merge_sort
//sorted result is returned as a new array
//assume array is populated with data prior to this call
template<class RandomAccessIterator>
void merge_sort(RandomAccessIterator first, RandomAccessIterator last){
  mergeSort(first, last, 0, last - first - 1);
}

template<class RandomAccessIterator>
long quickPartition(RandomAccessIterator first, long low, long high){
  auto i = low, j = high + 1;
  while( true ){
    while(first[++i] < first[low])
      if( i == high )
        break;
    while(first[low] < first[--j])
      if(j == low)
        break;
    if(i >= j)
      break;
    swap(first[i], first[j]);
  }
  swap(first[low], first[j] );
  return j;
}

template<class RandomAccessIterator>
void quickSort(RandomAccessIterator first, RandomAccessIterator last, long low, long high){
  if( low < high){
    auto j = quickPartition(first, low, high);
    quickSort(first, last, low, j-1);
    quickSort(first, last, j+1, high);
  }
}

// counting_sort
//sorted result is returned as a new array
//assume array is populated with data prior to this call
template<class RandomAccessIterator>
void quick_sort(RandomAccessIterator first, RandomAccessIterator last){
  quickSort(first, last, 0, last - first - 1);
}

//return 1 if a is sorted (in non-decreasing order);
//return 0 otherwise
//assume array is allocated and populated with data prior to this call
template<class RandomAccessIterator>
bool issorted(RandomAccessIterator first, RandomAccessIterator last){
  bool isOrdered = true;
  RandomAccessIterator Index;
  for(Index = first; Index != last-1; Index++)
    if(*Index > *(Index+1)){
      cout << *Index << " " << *(Index+1) << endl;
      isOrdered = false;
      break;
    }
  return isOrdered;
}

vector<int> generate_data(int size){
  vector<int> a;
  for(int i=0; i<size;i++){
    int random_number = (rand()%size)+1;
    a.push_back(random_number);
  }
  return a;
}

void printArray(vector<int> a){
  for(int i=0;i<a.size();i++)
    cout << a.at(i) << " ";
  cout << endl;
}

vector<int> inverseVector(vector<int> array){
  int size = array.size()-1;
  for(int i=0; i<size/2; i++)
    swap(array.at(i), array.at(size-i) );
  return array;
}

int main(int argc, char** argv) {
  srand((unsigned)time(0));
  ofstream fs("datos_sort.txt"); 
  clock_t tStart;
  double insertionTime, quickTime, mergeTime;
  //int ns[] = {100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000}; 
  int ns[] = {1000000}; 
  //int ns[] = {10};
  fs << "#numdatos insertion merge quick" << endl;
  for (auto size: ns) {
    //declare an array a
    vector<int> arr(size), arrAux(size), arr_insert_sort(size), arr_merge_sort(size), arr_quick_sort(size);
    
    fs << size;
    
    //populate the array with random data
    arr = generate_data(size);
    
    arrAux = arr;
    quick_sort(arrAux.begin(), arrAux.end());
    //printArray(arrAux);
    arrAux = inverseVector(arrAux);
    //printArray(arrAux);
    
    // crear una copia de a llamada aa
    arr_insert_sort = arrAux;
    //printArray(arr_insert_sort);
    
    tStart = clock();
    insert_sort(arr_insert_sort.begin(), arr_insert_sort.end());
    //printArray(arr_insert_sort);
    
    insertionTime = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    
    fs << " " << insertionTime;
    cout << "Tiempo que demoro el Insertion Sort con " << size <<" elementos fue: " << insertionTime << endl;
    
    //printArray(arr_insert_sort);
    if (!issorted(arr_insert_sort.begin(), arr_insert_sort.end())) {
      cout << "Insert Sort No esta ordenado"<<endl;
    } else {
      cout << "Insert Sort Ordenado"<<endl;
    }
    
    // crear una copia de a llamada aa
    arr_merge_sort = arr;//copy(a);
    //printArray(arr_merge_sort);
    
    tStart = clock();
    merge_sort(arr_merge_sort.begin(), arr_merge_sort.end());
    //printArray(arr_merge_sort);
    
    mergeTime = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    
    fs << " " << mergeTime;
    cout << "Tiempo que demoro el Merge Sort con " << size <<" elementos fue: " << mergeTime << endl;
    
    //printArray(arr_merge_sort);
    if (!issorted(arr_merge_sort.begin(), arr_merge_sort.end())) {
      cout << "Merge Sort No esta ordenado"<<endl;
    } else {
      cout << "Merge Sort Ordenado"<<endl;
    }
      
    // crear una copia de a llamada aa
    arr_quick_sort = arr;//copy(a);
    //printArray(arr_quick_sort);
    
    tStart = clock();
    quick_sort(arr_quick_sort.begin(), arr_quick_sort.end());
    //printArray(arr_quick_sort);
    
    quickTime = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    
    fs << " " << quickTime << endl;
    cout << "Tiempo que demoro el Quick Sort con " << size <<" elementos fue: " << quickTime << endl;
    
    //printArray(arr_quick_sort);
    if (!issorted(arr_quick_sort.begin(), arr_quick_sort.end())) {
      cout << "Quick Sort No esta ordenado"<<endl;
    } else {
      cout << "Quick Sort Ordenado"<<endl;
    }
    
    // calcular los tiempo e imprimirlo en consola
    // cout << n << " " << " " << t1 << " " << t2 << " " << t3 << endl;
  }

  return 0;
}
