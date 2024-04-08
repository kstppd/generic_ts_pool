#if 0 
/opt/rocm/llvm/bin/clang++   -I${ROCM_PATH}/include -O3 -std=c++20  -fPIC -shared  -D__HIP_PLATFORM_AMD__ -o libshoehorn.so shoehorn_hip.cpp -ldl
exit 0
#endif
/*
   Inspired by Samuel Antao's work -->https://github.com/sfantao/vlasiator-mempool 
   Running Vlasiator with it --> mpirun -n tasks -x LD_PRELOAD=./libshoehorn.so vlasiator....
*/
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <cstddef>
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
      hipError_t (*hipMalloc_ptr)(void**, size_t, unsigned int) = (hipError_t(*)(void**, size_t, unsigned int))dlsym(RTLD_NEXT, "hipMallocManaged");
      assert(hipMalloc_ptr);
      void* buffer = nullptr;
      hipMalloc_ptr(&buffer, MANAGED_POOL_SIZE, hipMemAttachGlobal);
      managedPool = new GENERIC_TS_POOL::MemPool(buffer, MANAGED_POOL_SIZE);
      initManaged = true;

#endif
   }

   {
#ifdef DEVICE_POOL_ENABLE
      printf("Installing Device Pool!\n");
      hipError_t (*hipMalloc_ptr)(void**, size_t) = (hipError_t(*)(void**, size_t))dlsym(RTLD_NEXT, "hipMalloc");
      assert(hipMalloc_ptr);
      void* buffer = nullptr;
      hipMalloc_ptr(&buffer, DEVICE_POOL_SIZE);
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
      hipError_t (*hipFree_ptr)(void*) = (hipError_t(*)(void*))dlsym(RTLD_NEXT, "hipFree");
      managedPool->destroy_with(hipFree_ptr);
      delete managedPool;
#endif
   }

   {
#ifdef DEVICE_POOL_ENABLE
      #ifdef SNITCH_MODE
      devicePool->defrag();
      devicePool->stats();
      #endif
      hipError_t (*hipFree_ptr)(void*) = (hipError_t(*)(void*))dlsym(RTLD_NEXT, "hipFree");
      devicePool->destroy_with(hipFree_ptr);
      delete devicePool;
#endif
   }
}

//Exposed API
extern "C" {
   #ifdef MANAGED_POOL_ENABLE
   hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
      UNUSED_ON_PURPOSE(flags);
      *ptr = (void*)managedPool->allocate<char>(size);
      if (ptr==nullptr){return hipErrorMemoryAllocation;}
      return hipSuccess;
   }
   #endif

   #ifdef DEVICE_POOL_ENABLE
   hipError_t hipMalloc(void** ptr, size_t size) {
      *ptr = (void*)devicePool->allocate<char>(size);
      if (ptr==nullptr){return hipErrorMemoryAllocation;}
      return hipSuccess;
   }

   hipError_t hipMallocAsync(void** ptr, size_t size,hipStream_t s) {
      UNUSED_ON_PURPOSE(s);
      *ptr = (void*)devicePool->allocate<char>(size);
      if (ptr==nullptr){return hipErrorMemoryAllocation;}
      return hipSuccess;
   }

   hipError_t hipFreeAsync(void* ptr,hipStream_t s) {
      UNUSED_ON_PURPOSE(s);
      devicePool->deallocate(ptr);
      return hipSuccess;
   }
   #endif

   hipError_t hipFree(void* ptr) {
      bool ok = false;
      #ifdef MANAGED_POOL_ENABLE
      ok = managedPool->deallocate(ptr);
      #endif
      if (!ok) {
         #ifdef DEVICE_POOL_ENABLE
         devicePool->deallocate(ptr);
         #endif
      }
      return hipSuccess;
   }
}
