#include <gtest/gtest.h>
#include <llvm/Support/TargetSelect.h>

int main(int argc, char** argv)
{
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
