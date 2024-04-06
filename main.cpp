#include "genericTsPool.h"
#include <iostream>

void go (){
   constexpr size_t bytes=128*1024*1024;
   void* block = malloc (bytes);
   GENERIC_TS_POOL::MemPool p (block,bytes);
   
   for (size_t i =0 ; i < 10000000 ;i++){
      float* n= p.allocate<float>(10000);
      for (size_t j = 0; j < 100; j++) {
         n[j] = j;
      }
      p.deallocate(n);
   }
   free(block);
}


int main(){
   go();
   return 0 ;
}
