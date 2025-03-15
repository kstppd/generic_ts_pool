#if 0 
g++ -I/usr/include/ -fPIC --shared -g3 -ggdb  -O3 -march=native -o shoehorn_cpu.so shoehorn_cpu.cpp -ldl
exit 0
#endif
#define _GNU_SOURCE
#include "genericTsPool.h"
#include <assert.h>
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static constexpr size_t KB = 1024ull;
static constexpr size_t MB = KB * KB;
static constexpr size_t GB = KB * KB * KB;
static constexpr size_t TB = KB * KB * KB * KB;


static size_t POOL_SIZE = 8ull * GB;
GENERIC_TS_POOL::MemPool* p = nullptr;
static int installed = 0;
static void* (*original_malloc)(size_t size) = nullptr;
static void (*original_free)(void*) = nullptr;

static void initme() {
   (void)write(STDOUT_FILENO, "Init POOL\n", strlen("Init POOL\n"));
   original_malloc = reinterpret_cast<void* (*)(size_t)>(dlsym(RTLD_NEXT, "malloc"));
   assert(original_malloc != NULL);
   void* buffer = original_malloc(POOL_SIZE);
   assert(buffer);
   void* mem = original_malloc(sizeof(GENERIC_TS_POOL::MemPool));
   p = new (mem) GENERIC_TS_POOL::MemPool(buffer, POOL_SIZE);
   installed = 1;
}

void __attribute__((destructor)) fini() {
   original_free = reinterpret_cast<void (*)(void*)>(dlsym(RTLD_NEXT, "free"));
   original_free(p);
}

extern "C" {
void* malloc(size_t size) {
   if (installed == 0) {
      initme();
   }
   void* ptr = (void*)p->allocate<char>(size);
   return ptr;
}

void* realloc(void* ptr, size_t size) {
   if (installed == 0) {
      initme();
   }

   void* newptr = (void*)p->allocate<char>(size);
   if (ptr == NULL) {
      return newptr;
   }
   memmove(newptr, ptr, size);
   return newptr;
}

void* calloc(size_t nmemb, size_t size) {
   if (installed == 0) {
      initme();
   }
   void* ptr = (void*)p->allocate<char>(nmemb * size);
   memset(ptr, 0, nmemb * size);
   return ptr;
}

void* memalign(size_t blocksize, size_t bytes) {
   abort();
   return nullptr;
}

void free(void* ptr) {
   if (installed == 0) {
      initme();
   }
}
}
