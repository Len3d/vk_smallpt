// vadd.cl -- a C99-style OpenCL C kernel compiled to SPIR-V by clspv and run on Vulkan compute.
//
// The kernel does one thing: out[i] = a[i] + b[i]
//
// The clspv mapping rules this example relies on:
//   1. Every __kernel function becomes one SPIR-V entry point whose name is kept by default (vadd).
//   2. __global pointer arguments map to storage buffers in descriptor set 0, with binding numbers
//      assigned in declaration order: a -> 0, b -> 1, out -> 2.
//      The host therefore builds a descriptor set layout with three STORAGE_BUFFER bindings 0/1/2.
//   3. get_global_id(0) maps to Vulkan's gl_GlobalInvocationID.x, so the total work item count of
//      vkCmdDispatch equals the number of array elements.
//      The workgroup size (local size) can be specified in two ways:
//        (a) without reqd_work_group_size, clspv emits it as the WorkgroupSize spec constants
//            (SpecId 0/1/2, defaulting to 1,1,1). The host must then set them through
//            VkSpecializationInfo, otherwise a workgroup runs a single thread - the pitfall this
//            example hit at the beginning.
//        (b) with __attribute__((reqd_work_group_size(64,1,1))), clspv emits
//            OpExecutionMode LocalSize 64 1 1 directly, so the host needs no spec constants, but
//            the kernel is fixed at 64 threads per workgroup and the dispatch must use 64 as well.
//      This example uses (b), matching local_size_x=64 of the GLSL reference implementation.
//   4. The kernel calls no libc/libclc math functions, so it does not depend on the clspv builtins
//      library and the compiled module is very small.
//
// Build command (note the clspv argument order: input file first, options after; -O needs a level):
//   clspv src/vadd.cl -o vadd.spv -O3
//
// In production it is better to export a descriptor map and build the layout from it instead of
// relying on memory:
//   clspv src/vadd.cl -o vadd.spv -O3 --descriptormap=vadd.map
//
// To verify the bindings, query the module directly:
//   clspv-reflection vadd.spv

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vadd(__global const float *a,
                   __global const float *b,
                   __global float *out)
{
    const unsigned int gid = (unsigned int)get_global_id(0);
    out[gid] = a[gid] + b[gid];
}
