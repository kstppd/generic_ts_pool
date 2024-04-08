#if 0 
g++ -I/usr/include/ -fPIC --shared  -O3  -o libshoehorn.so shoehorn.cpp -ldl -lcudart 
exit 0
#endif
/*
   Inspired by Samule Antao's work -->https://github.com/sfantao/vlasiator-mempool 
   Running Vlasiator with it --> mpirun -n tasks -x LD_PRELOAD=./libshoehorn.so vlasiator....
*/
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "genericTsPool.h"

static constexpr size_t KB = 1024ull;
static constexpr size_t MB = KB * KB;
static constexpr size_t GB = KB * KB * KB;
static constexpr size_t TB = KB * KB * KB * KB;

// Enable pools and pick pool sizes
#define DEVICE_POOL_ENABLE    //hooks cudaMalloc, cudaMallocAsync, cudaFree and cudaFreeAsync
#define MANAGED_POOL_ENABLE   //hooks cudaMallocManaged and cudaFree
#define SNITCH_MODE           // will print allocated and free blocks at the end of program.

//Pick size for each pool
static constexpr size_t DEVICE_POOL_SIZE  = 4 * GB;
static constexpr size_t MANAGED_POOL_SIZE = 4 * GB;

//----------Do not touch-----------------
#define UNUSED_ON_PURPOSE(x) (void)x
#ifdef DEVICE_POOL_ENABLE
static GENERIC_TS_POOL::MemPool *devicePool;
bool initDevice = false;
#endif

#ifdef MANAGED_POOL_ENABLE
static GENERIC_TS_POOL::MemPool *managedPool;
bool initManaged = false;
#endif
  
__attribute__((constructor))
static void initme(){
   {
#ifdef MANAGED_POOL_ENABLE
      printf("Installing Managed Pool!\n");
      cudaError_t (*cudaMalloc_ptr)(void**, size_t, unsigned int) =
          (cudaError_t(*)(void**, size_t, unsigned int))dlsym(RTLD_NEXT, "cudaMallocManaged");
      assert(cudaMalloc_ptr);
      void* buffer = nullptr;
      cudaMalloc_ptr(&buffer, MANAGED_POOL_SIZE, cudaMemAttachGlobal);
      managedPool = new GENERIC_TS_POOL::MemPool(buffer, MANAGED_POOL_SIZE);
      initManaged = true;
#endif
   }

   {
#ifdef DEVICE_POOL_ENABLE
      printf("Installing Device Pool!\n");
      cudaError_t (*cudaMalloc_ptr)(void**, size_t) = (cudaError_t(*)(void**, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
      assert(cudaMalloc_ptr);
      void* buffer = nullptr;
      cudaMalloc_ptr(&buffer, DEVICE_POOL_SIZE);
      devicePool = new GENERIC_TS_POOL::MemPool(buffer, DEVICE_POOL_SIZE);
      initDevice = true;
#endif
   }
}

__attribute__((destructor))
static void finitme(){
   {
#ifdef MANAGED_POOL_ENABLE
      #ifdef SNITCH_MODE
      managedPool->defrag();
      managedPool->stats();
      #endif
      cudaError_t (*cudaFree_ptr)(void*) = (cudaError_t(*)(void*))dlsym(RTLD_NEXT, "cudaFree");
      managedPool->destroy_with(cudaFree_ptr);
      delete managedPool;
#endif
   }

   {
#ifdef DEVICE_POOL_ENABLE
      #ifdef SNITCH_MODE
      devicePool->defrag();
      devicePool->stats();
      #endif
      cudaError_t (*cudaFree_ptr)(void*) = (cudaError_t(*)(void*))dlsym(RTLD_NEXT, "cudaFree");
      devicePool->destroy_with(cudaFree_ptr);
      delete devicePool;
#endif
   }
}

//Exposed API
extern "C" {
   #ifdef MANAGED_POOL_ENABLE
   cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned int flags) {
      UNUSED_ON_PURPOSE(flags);
      *ptr = (void*)managedPool->allocate<char>(size);
      if (ptr==nullptr){return cudaErrorMemoryAllocation;}
      return cudaSuccess;
   }
   #endif

   #ifdef DEVICE_POOL_ENABLE
   cudaError_t cudaMalloc(void** ptr, size_t size) {
      *ptr = (void*)devicePool->allocate<char>(size);
      if (ptr==nullptr){return cudaErrorMemoryAllocation;}
      return cudaSuccess;
   }

   cudaError_t cudaMallocAsync(void** ptr, size_t size,cudaStream_t s) {
      UNUSED_ON_PURPOSE(s);
      *ptr = (void*)devicePool->allocate<char>(size);
      if (ptr==nullptr){return cudaErrorMemoryAllocation;}
      return cudaSuccess;
   }

   cudaError_t cudaFreeAsync(void* ptr,cudaStream_t s) {
      UNUSED_ON_PURPOSE(s);
      devicePool->deallocate(ptr);
      return cudaSuccess;
   }
   #endif

   cudaError_t cudaFree(void* ptr) {
      bool ok = false;
      #ifdef MANAGED_POOL_ENABLE
      ok = managedPool->deallocate(ptr);
      #endif
      if (!ok) {
         #ifdef DEVICE_POOL_ENABLE
         devicePool->deallocate(ptr);
         #endif
      }
      return cudaSuccess;
   }
}
