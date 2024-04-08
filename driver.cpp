#include <iostream>
#include <hip/hip_runtime.h>
#define N 1024

int main(){
  int* data=nullptr;
  hipMallocManaged(&data,N*sizeof(int));
  std::cout<<data<<std::endl;
  for (size_t i=0;i<N;++i){
    data[i]=i;
  }
  std::cout<<data[512]<<std::endl;
  hipFree(data);

  hipMalloc(&data,N*sizeof(int));
  std::cout<<data<<std::endl;
  hipFree(data);
  return 0;
}
