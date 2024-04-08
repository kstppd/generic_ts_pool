
## Description: 
A thread safe generic memory pool that is memory type agnostic which means that it can manage CUDA,HIP,Stack,Dynamic memory et al.

### CUDA Hooks
The file ```shoehorn_cuda.cpp``` implements hooks for cuda using ```genericTsPool.h``` inspired by https://github.com/sfantao/vlasiator-mempool.git.

### HIP Hooks
The file ```shoehorn_hip.cpp``` implements hooks for HIP using ```genericTsPool.h``` inspired by https://github.com/sfantao/vlasiator-mempool.git.


## A Minimal Example 
```c++
//main.cpp
size_t sz=1<<12;
char block[sz];  // <-- this can be whatever memory block and the pool will never try 
                                                     //      to dereference it. 
GENERIC_TS_POOL::MemPool pool(&block,sz);

//Allocate an array of 4 doubles
double* number = pool.allocate<double>(4);

// Do smth with it
number[0]=1.0;
number[1]=2.0;
.  .  .
.  .  .
//Deallocate the array
pool.deallocate(number);

//Defrag the pool.
pool.defrag();
```
