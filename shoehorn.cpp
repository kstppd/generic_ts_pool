#if 0 
g++ -std=c++20 -fPIC -shared -O3 -o libshoehorn.so  shoehorn.cpp
exit 0
#endif
//Kostis Papadakis 2023
//Inspired by Samuel Antao's work: https://github.com/sfantao/vlasiator-mempool.git
#include "genericTsPool.h"
#include <dlfcn.h>
#include <mutex>
#include <stdarg.h>

static constexpr size_t KB = 1024ull;
static constexpr size_t MB = KB * KB;
static constexpr size_t GB = KB * KB * KB;
static constexpr size_t TB = KB * KB * KB * KB;

//Enable pools and pick pool sizes
#define DEVICE_POOL_ENABLE
#define MANAGED_POOL_ENABLE
//#define HOST_POOL_ENABLE
static constexpr size_t DEVICE_POOL_SIZE = 32 * GB;
static constexpr size_t MANAGED_POOL_SIZE = 32 * GB;
static constexpr size_t HOST_POOL_SIZE = 512 * MB;

#ifdef DEVICE_POOL_ENABLE
GENERIC_TS_POOL::MemPool devicePool;
bool initDevice = false;
#endif

#ifdef MANAGED_POOL_ENABLE
GENERIC_TS_POOL::MemPool managedPool;
bool initManaged = false;
#endif

#ifdef HOST_POOL_ENABLE
GENERIC_TS_POOL::MemPool hostPool;
bool initHost = false;
#endif

// Mutextes used only for initializing the pools
std::mutex _device_lock;
std::mutex _device_async_lock;
std::mutex _managed_lock;
std::mutex _host_lock;

// Original signatures
#define cudaError_t int
#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
typedef cudaError_t (*orig_Malloc_type)(void** ptr, size_t n);
typedef cudaError_t (*orig_Free_type)(void* ptr);
const char* c_Malloc{"cudaMalloc"};
const char* c_MallocManaged{"cudaMallocManaged"};
const char* c_MallocHost{"cudaMallocHost"};

extern "C" {

#ifdef MANAGED_POOL_ENABLE
cudaError_t cudaMallocManaged(void** ptr, size_t n) {
   if (!initManaged) {
      _device_lock.lock();
      if (initManaged) {
         return cudaMallocManaged(ptr, n);
      }
      initManaged = true;
      void* buffer = nullptr;
      orig_Malloc_type origmalloc = (orig_Malloc_type)dlsym(RTLD_NEXT, c_MallocManaged);
      origmalloc((void**)&buffer, MANAGED_POOL_SIZE);
      managedPool.resize(buffer, MANAGED_POOL_SIZE);
      _device_lock.unlock();
   }
   *ptr = managedPool.allocate<char>(n);
   return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
#endif

#ifdef DEVICE_POOL_ENABLE
cudaError_t cudaMalloc(void** ptr, size_t n) {
   if (!initDevice) {
      _device_lock.lock();
      if (initDevice) {
         return cudaMalloc(ptr, n);
      }
      initDevice = true;
      void* buffer = nullptr;
      orig_Malloc_type origmalloc = (orig_Malloc_type)dlsym(RTLD_NEXT, c_Malloc);
      origmalloc((void**)&buffer, DEVICE_POOL_SIZE);
      devicePool.resize(buffer, DEVICE_POOL_SIZE);
      _device_lock.unlock();
   }
   *ptr = devicePool.allocate<char>(n);
   return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
#endif

#ifdef DEVICE_POOL_ENABLE
cudaError_t cudaMallocAsync(void** ptr, size_t n) {
   if (!initDevice) {
      _device_async_lock.lock();
      if (initDevice) {
         return cudaMallocAsync(ptr, n);
      }
      initDevice = true;
      void* buffer = nullptr;
      orig_Malloc_type origmalloc = (orig_Malloc_type)dlsym(RTLD_NEXT, c_Malloc);
      origmalloc((void**)&buffer, DEVICE_POOL_SIZE);
      devicePool.resize(buffer, DEVICE_POOL_SIZE);
      _device_async_lock.unlock();
   }
   *ptr = devicePool.allocate<char>(n);
   return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
#endif

#ifdef HOST_POOL_ENABLE
cudaError_t cudaMallocHost(void** ptr, size_t n) {
   if (!initHost) {
      _host_lock.lock();
      if (initHost) {
         return cudaMallocHost(ptr, n);
      }
      initHost = true;
      void* buffer = nullptr;
      orig_Malloc_type origmalloc = (orig_Malloc_type)dlsym(RTLD_NEXT, c_MallocHost);
      origmalloc((void**)&buffer, HOST_POOL_SIZE);
      hostPool.resize(buffer, HOST_POOL_SIZE);
      _host_lock.unlock();
   }
   *ptr = hostPool.allocate<char>(n);
   return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaFreeHost(void* ptr) {
   hostPool.deallocate(ptr);
   return cudaSuccess;
}
#endif

#if defined(MANAGED_POOL_ENABLE) && defined(DEVICE_POOL_ENABLE)
cudaError_t cudaFree(void* ptr) {
   bool done = devicePool.deallocate(ptr);
   if (!done){
      managedPool.deallocate(ptr);
   }
   return cudaSuccess;
}
#endif

#if defined(MANAGED_POOL_ENABLE) && !defined(DEVICE_POOL_ENABLE)
cudaError_t cudaFree(void* ptr) {
   managedPool.deallocate(ptr);
   return cudaSuccess;
}
#endif

#if !defined(MANAGED_POOL_ENABLE) && defined(DEVICE_POOL_ENABLE)
cudaError_t cudaFree(void* ptr) { devicePool.deallocate(ptr); }
#endif

#ifdef DEVICE_POOL_ENABLE
cudaError_t cudaFreeAsync(void* ptr) {
   devicePool.deallocate(ptr);
   return cudaSuccess;
}
#endif
}
