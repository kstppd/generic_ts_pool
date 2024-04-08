#include "genericTsPool.h"
#include <iostream>
#include <random>
#include <cassert>

template <typename T>
bool allocate_to_the_brim_and_test(size_t len){
   
   T* data = new T[len];
   if (!data){
      std::cerr<<"Failed to allocate initial buffer"<<std::endl;
      return false;
   }

   std::random_device dev;
   std::mt19937 rng(dev());
   std::uniform_int_distribution<size_t> dist(1<<2,len/2);
   GENERIC_TS_POOL::MemPool p(data, len * sizeof(T));

   bool ok =true;
   int step=0;
   std::vector<T*> ptrs;
   for (;;){
      size_t sz=dist(rng);
      T* array=p.allocate<T>(sz);
      if (array==nullptr){
         // printf("Reached max allocation with load = %f.\n",p.load());
         assert(p.load()<1.0 && "erroneous load factor in pool!");
         if (step==0){
            step++;
            p.release();
            ptrs.clear();
            continue;
         }else{
            break;
         }
      }
      ptrs.push_back(array);

      //Set
      for (size_t i =0;i<sz;++i){
         array[i]=i;
      }

      //Check
      for (size_t i =0;i<sz;++i){
         if (array[i]!=(T)i){
            std::cout<<"ERROR: Expected "<<i <<" got "<<array[i]<<std::endl;
            ok=false;
            break;
         }
      }
   }

   for (auto ptr:ptrs){
      p.deallocate(ptr);
   }
   p.defrag();
   //Now we should have 0 alloc blocks and only one large degraged full free block
   ok = ok&& (p.size()==0);

   delete[] data;
   data=nullptr;
   return ok;
}

int main(){
   //Simple sanity check
   bool ok = true;
   size_t test_size = 1<<24;
   for (size_t i=0; i< 10; ++i){
      printf("Running test rep %zu/10...\n",i);
      ok= ok&& allocate_to_the_brim_and_test<float>(test_size);
      ok= ok&& allocate_to_the_brim_and_test<double>(test_size);
      ok= ok&& allocate_to_the_brim_and_test<int>(test_size);
   }
   if (ok){
      printf("Test passed with success!\n");
   }else{
      printf("Test failed!\n");
   }
   return ok?1:0 ;
}
