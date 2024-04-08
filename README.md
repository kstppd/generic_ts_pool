
## Description: 
A thread safe generic memory pool that is memory type agnostic which means that it can manage CUDA,HIP,Stack,Dynamic memory et al.

### CUDA Hooks
The file ```shoehorn.cpp``` implements hooks for cuda using the ```genericTsPool.h``` inspired by https://github.com/sfantao/vlasiator-mempool.git.

```c++
        //main.cpp
        //Example Minimal Usage:
        size_t sz=1<<12;
        char block[sz];  // <-- this can be whatever memory block
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
