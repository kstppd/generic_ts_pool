#include "genericTsPool.h"
#include <gtest/gtest.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

TEST(MemPoolTest, Unit) {
  char block[1 << 12];
  GENERIC_TS_POOL::MemPool p(&block, 1 << 12);

  std::vector<double *> ptrs;
  for (size_t i = 0; i < 4 * 8; i++) {
    auto number = p.allocate<double>(4);
    ptrs.push_back(number);
  }
  for (auto i : ptrs) {
    p.deallocate(i);
  }
  p.defrag();
  EXPECT_TRUE(p.size() == 0);
}

int main(int argc, char *argv[]) {
  srand(time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
