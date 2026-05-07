#!/usr/bin/env python3
"""Fix nccl.h by adding missing ncclWindow_t typedef for DeepEP compilation."""
import os, sys

nccl_h = os.path.expanduser(
    "~/.conda/envs/megatron310/lib/python3.10/site-packages/nvidia/nccl/include/nccl.h"
)
with open(nccl_h) as f:
    content = f.read()

if "ncclWindow_t" in content:
    print("Already fixed")
    sys.exit(0)

insert = "struct ncclWindow_vidmem;\ntypedef struct ncclWindow_vidmem* ncclWindow_t;\n"
content = content.replace("#define NCCL_MAJOR", insert + "#define NCCL_MAJOR")
with open(nccl_h, "w") as f:
    f.write(content)
print("nccl.h fixed: ncclWindow_t typedef added")
