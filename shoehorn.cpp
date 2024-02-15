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
//#define DEVICE_POOL_ENABLE
#define MANAGED_POOL_ENABLE
//#define HOST_POOL_ENABLE
static constexpr size_t DEVICE_POOL_SIZE = 3 * GB;
static constexpr size_t MANAGED_POOL_SIZE = 3 * GB;
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
const char* c_Free{"cudaFree"};
const char* c_FreeHost{"cudaFreeHost"};

template<typename type>
void init_pool(GENERIC_TS_POOL::MemPool& p,size_t size, const char* symbol){
   void* buffer = nullptr;
   type original = (type)dlsym(RTLD_NEXT, symbol);
   original((void**)&buffer,size);
   if (buffer==nullptr){
      throw std::runtime_error("Failed to initialize Pools");
   }
   p.resize(buffer, size);
}

static void __attribute__((destructor))destroy_pools(){
#ifdef MANAGED_POOL_ENABLE
   {
      printf("Destroying managed Pool\n");
      orig_Free_type deallocMethod = (orig_Free_type)dlsym(RTLD_NEXT, c_Free);
      managedPool.destroy_with(deallocMethod);
   }
#endif
#ifdef DEVICE_POOL_ENABLE
   {
      printf("Destroying devide Pool\n");
      orig_Free_type deallocMethod = (orig_Free_type)dlsym(RTLD_NEXT, c_Free);
      devicePool.destroy_with(deallocMethod);
   }
#endif
#ifdef HOST_POOL_ENABLE
   {
      printf("Destroying host Pool\n");
      orig_Free_type deallocMethod = (orig_Free_type)dlsym(RTLD_NEXT, c_FreeHost);
      hostPool.destroy_with(deallocMethod);
   }
#endif
}

extern "C" {

#ifdef MANAGED_POOL_ENABLE
cudaError_t cudaMallocManaged(void** ptr, size_t n) {
   if (!initManaged) {
      _managed_lock.lock();
      if (initManaged) {
         return cudaMallocManaged(ptr, n);
      }
      initManaged = true;
      printf("Intializing managed pool with size= %zu B\n",MANAGED_POOL_SIZE);
      init_pool<orig_Malloc_type>(managedPool,MANAGED_POOL_SIZE,c_MallocManaged);
      _managed_lock.unlock();
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
      printf("Intializing device pool with size= %zu B\n",DEVICE_POOL_SIZE);
      init_pool<orig_Malloc_type>(devicePool,DEVICE_POOL_SIZE,c_Malloc);
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
      printf("Intializing device pool with size= %zu B\n",DEVICE_POOL_SIZE);
      init_pool<orig_Malloc_type>(devicePool,DEVICE_POOL_SIZE,c_Malloc);
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
      printf("Intializing host pool with size= %zu B\n",HOST_POOL_SIZE);
      init_pool<orig_Malloc_type>(hostPool,HOST_POOL_SIZE,c_MallocHost);
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
cudaError_t cudaFree(void* ptr) {
   devicePool.deallocate(ptr); 
   return cudaSuccess;
}
#endif

#ifdef DEVICE_POOL_ENABLE
cudaError_t cudaFreeAsync(void* ptr) {
   devicePool.deallocate(ptr);
   return cudaSuccess;
}
#endif
}
