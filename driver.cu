#include <iostream>
#define N 1024

int main(){
  int* data=nullptr;
  cudaMallocManaged(&data,N*sizeof(int));
  std::cout<<data<<std::endl;
  for (size_t i=0;i<N;++i){
    data[i]=i;
  }
  std::cout<<data[512]<<std::endl;
  cudaFree(data);

  cudaMalloc(&data,N*sizeof(int));
  std::cout<<data<<std::endl;
  cudaFree(data);
  return 0;
}
