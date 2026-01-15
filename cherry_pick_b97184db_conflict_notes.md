# b97184db cherry-pick 冲突处理记录

本文记录将上游提交 `b97184db566352ed2095d389bf09ff76649cc231` 回移到当前 `Megatron-LM-Enhanced` 分支时，已手工处理的核心冲突。

## 1. `megatron/core/models/gpt/gpt_model.py`

### 1.1 冲突原因

上游版本把 `gpt_model.py` 作为完整 `GPTModel` 实现文件直接修改；当前分支则把这里改造成了一个路由包装层，根据 `enable_module_queue`、`post_process` 和 PP 拓扑，在 `GPTModelNormal` 与 `GPTModelModuleQueue` 之间切换。

### 1.2 处理方式

保留当前分支的包装层设计，不直接合入上游的大段 `GPTModel` 实现。

### 1.3 处理结果

- 保留 `GPTModel.__new__()` 的选择逻辑。
- 保留 `GPTModelNormal` / `GPTModelModuleQueue` 的导出方式。
- 去掉误并入的上游主体实现，避免破坏当前分支已有的 module queue 定制。

## 2. `megatron/core/transformer/attention.py`

### 2.1 冲突原因

上游在 QKV 和 core attention 路径中加入了细粒度激活卸载包裹与 `group_commit()`；但同时还带入了当前老分支未完全具备的能力：

- `get_query_key_value_tensors(..., output_gate=...)`
- 动态 batching 下的量化 scale 辅助清零逻辑

### 2.2 处理方式

采用“保留细粒度卸载主路径，去掉当前分支不兼容扩展”的合并方案。

### 2.3 处理结果

- 保留 `qkv_linear` 的 `off_interface(...)` 包裹与 `group_commit()`。
- 保留 `core_attn` 的 `group_commit()`，这样 QKV/core-attention 两段细粒度卸载链路还在。
- **没有**保留 `output_gate=` 参数，因为当前分支的 `get_query_key_value_tensors()` 接口不支持它。
- **没有**保留 `is_using_quantization_scales()` 相关逻辑，因为当前分支还没有对应工具函数与完整配套实现，直接并入会引入新错误。

## 3. `megatron/core/transformer/transformer_layer.py`

### 3.1 冲突原因

上游把 MLP 后处理拆成 `_forward_post_mlp()`，并加入了一个针对 MoE + TE CUDA Graph router capture 的特化分支。当前分支没有完整同步这套 CUDA Graph 相关上下文与依赖。

### 3.2 处理方式

保留当前分支原有的 MLP 后处理结构，不强行引入上游的 MoE CUDA Graph 特化路径。

### 3.3 处理结果

- 保留当前分支已有的 `mlp_norm` 细粒度卸载/提交路径。
- 去掉仅在较新上游上下文中才完整成立的 `_forward_post_mlp()` 拆分和 `CudaGraphScope.moe_router` 特化逻辑。
- 这样可以避免把不完整的 CUDA Graph / MoE 路径半并入当前分支。

## 4. `megatron/training/arguments.py`

### 4.1 冲突原因

这里是纯参数面冲突：当前分支没有这些 CLI 参数，上游新增了 fine-grained activation offloading 所需开关。

### 4.2 处理方式

直接保留上游新增参数。

### 4.3 处理结果

新增以下参数：

- `--fine-grained-activation-offloading`
- `--offload-modules`
- `--min-offloaded-tensor-size`
- `--batch-invariant-mode`

## 5. `tests/test_utils/recipes/moe.yaml`

### 5.1 冲突原因

本地分支在 MR 测试列表里保留了 FSDP 暂不可用的注释和旧命名测试项；上游则新增了两条 fine-grained offloading 的 MoE 功能测试，并对部分 case 名称做了整理。

### 5.2 处理方式

合并两边内容，但优先保留当前分支里已经存在、且目录路径真实存在的测试 case 命名。

### 5.3 处理结果

- 保留本地关于 EP + FSDP 暂不可用的注释说明。
- 新增两条 fine-grained offloading MR 测试：
  - `gpt3_moe_mcore_te_tp2_pp2_ep4_etp1_fine_grained_offloading`
  - `gpt3_moe_mcore_te_tp2_pp2_ep4_etp1_no_mtp_no_a2a_ovlp_fine_grained_offloading`
- 保留本地已有的 `gpt3_mr_mcore_te_tp2_pp1_te_8experts2parallel_ddp_average_in_collective_dgx_a100_1N8G` 命名，避免 recipe 与现有测试目录不一致。

## 6. `tests/unit_tests/models/test_mamba_moe_model.py`

### 6.1 冲突原因

这是一个 delete/update 型冲突：当前分支索引侧没有保留该文件，而上游新增内容仍然存在。

### 6.2 处理方式

保留上游版本。

### 6.3 处理结果

- 继续保留该单元测试文件。
- 文件中已经包含 fine-grained activation offloading 相关默认配置字段，可用于覆盖新增配置项。

## 7. `docs/source/api-guide/fine_grained_activation_offloading.md`

### 7.1 冲突原因

这是上游新增文档，当前分支本地不存在对应文件。

### 7.2 处理方式

直接保留上游文档和配图资源。

### 7.3 处理结果

- 保留 API guide 文档。
- 保留配图 `docs/images/fine_grained_activation_offloading/offloading_and_recomputing.png`。

## 当前结论

这次处理的原则是：

1. **保留当前分支已有定制结构**，尤其是 module queue 相关改动；
2. **优先落下 fine-grained activation offloading 的主干能力**；
3. **不把当前分支缺少配套依赖的新上游细节硬塞进来**，避免引入新的运行时/接口错误。

目前已经没有未合并状态的冲突文件；剩余状态主要是本次 cherry-pick 本身带来的正常已修改/新增文件，后续可以继续做功能验证或直接继续完成 cherry-pick。

