import os

if __name__ == "__main__":
    file_list = os.listdir("./cuda/")
    for f in file_list:
        fh = open("./cuda/" + f, "r")
        lines = fh.readlines()
        # Step 1
        neo_lines = []
        # Step 2
        neo_lines = ["#include<ATen/ceil_div.h>\n"]
        # Step 3
        neo_lines = ["#include<ATen/ceil_div.h>\n", "#include <ATen/cuda/ThrustAllocator.h>\n"]
        fh.close()
        fh = open(f, "w")
        for line in lines:
            # Step 1
            s = line.replace("THCudaCheck", "AT_CUDA_CHECK")
            s = s.replace("#include <THC/THC.h>", "")
            # Step 2
            s = s.replace("dim3 grid(std::min(at::ceil_div((int)**, (int)512), 4096));", "dim3 grid(std::min(((int)** + 512 -1) / 512, 4096));")
            # Step3
            s = s.replace("THCState *state = at::globalContext().lazyInitCUDA();", "")
            s = s.replace("mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));", "mask_dev = (unsigned long long*) c10::cuda::CUDACachingAllocator::raw_alloc(boxes_num * col_blocks * sizeof(unsigned long long));")
            s = s.replace("THCudaFree(state, mask_dev);", "c10::cuda::CUDACachingAllocator::raw_delete(mask_dev);")
            neo_lines.append(s)
        fh.writelines(neo_lines)
