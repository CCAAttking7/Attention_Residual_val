import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import numpy as np
from src.modeling_attnres_llama import KimiLlamaDecoderLayer

# ================= 配置区 =================
BASELINE_PATH = "/root/autodl-tmp/attnres-checkpoints/baseline/model.safetensors"
V3_PATH = "/root/autodl-tmp/attnres-checkpoints/attnres_v1/model.safetensors"
BASE_MODEL_ID = "unsloth/Llama-3.2-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SIZE = 3500  # 每组测试的上下文长度
MAX_TEST_CHUNKS = 5  # 最大测试组数
# ==========================================


# ================= 模型手术与前向劫持 =================
# 这里self，经过刚才的绑定，现在指的是整个Llama的模型实例，而不是单层的DecoderLayer了，这样我们就能在整个模型的前向过程中注入我们自定义的逻辑，允许我们在评测时动态地构建历史状态并传递给每一层，模拟真实的长上下文处理场景。
def eval_kimi_model_forward(
    self, input_ids=None, position_ids=None, inputs_embeds=None, **kwargs
):
    # 词表映射
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # input_embeds 的 shape 是 (batch_size, seq_len, hidden_size)
    batch_size, seq_len = inputs_embeds.shape[:2]
    device = inputs_embeds.device

    # 如果 position_ids 没有提供，我们就自动生成一个连续的 position_ids，确保每个 token 都有对应的位置编码，这对于后续的旋转位置编码计算是必要的。
    if position_ids is None:
        position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

    # RoPE
    cos, sin = self.rotary_emb(inputs_embeds, position_ids)
    # ⚠️ 【维度陷阱填坑】
    # Llama 每个头的维度是 64。由于复数的对称性，HuggingFace 底层只返回了 32 维的 cos 和 sin 来省内存。
    # 但我们劫持了 forward，需要自己把这 32 维“复制并拼接”成 64 维，否则后面算 Attention 旋转时维度对不上。
    # torch.cat([cos, cos], dim=-1) 的意思就是：在最后一维上，把自己和自己拼起来 (32 + 32 = 64)
    if cos.shape[-1] < self.layers[0].self_attn.head_dim:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    position_embeddings = (cos, sin)

    hidden_states = inputs_embeds
    history_cache = []
    block_size = 4

    for i, layer_module in enumerate(self.layers):
        # 准备历史信息：如果 cache 里有东西，就浅拷贝 (list()) 一份传给当前层；如果是空，就传 None
        # 为什么要 list()？防止当前层在内部修改历史记录，污染全局环境，确保每层拿到的历史都是独立的副本。
        current_history = list(history_cache) if len(history_cache) > 0 else None

        # 调用具体的某一层的计算逻辑（这里调用的其实就是咱们的 KimiLlamaDecoderLayer 的 forward）
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=None,  # 不传 mask，让底层自动算因果掩码
            position_ids=position_ids,
            history=current_history,
            position_embeddings=position_embeddings,
            use_cache=False,  # 评测全量文本，不开 KV Cache 加速
        )
        hidden_states = layer_outputs[0]

        if (i + 1) % block_size == 0:
            # detach() 极其重要！它代表“切断梯度图”。
            # 在推理或计算过程中，告诉 PyTorch：“我只存这个张量的纯数值，你别去追踪它的求导历史了。”
            # 这能省下巨量的内存，防止爆显存。
            history_cache.append(hidden_states.detach())

    hidden_states = self.norm(hidden_states)
    # BaseModelOutputWithPast 是 HuggingFace 定义的一个标准输出格式，包含了模型的最后隐藏状态和可选的 past_key_values（虽然我们这里不使用缓存，但保持这个格式可以兼容 HuggingFace 的其他工具和接口）。通过返回这个对象，我们确保了我们的评测模型在接口上与原版 Llama 完全兼容，方便后续的评测和分析工作。
    return BaseModelOutputWithPast(last_hidden_state=hidden_states)


def patch_model_for_eval(model, config):
    for i in range(len(model.model.layers)):
        old_layer = model.model.layers[i]
        # 刚事例化出来默认是在CPU，精度是float32的，我们需要把它搬到正确的设备和精度上
        new_layer = KimiLlamaDecoderLayer(config, layer_idx=i)
        # 搬到和旧层相同的设备和数据类型上，确保权重加载时不会因为设备或dtype不匹配而报错
        new_layer.to(
            device=old_layer.input_layernorm.weight.device,
            dtype=old_layer.input_layernorm.weight.dtype,
        )
        # 加载旧层权重，strict=False 允许新层有额外的参数（如 gate 和 w_l），不会因为这些参数在旧层中不存在而报错
        new_layer.load_state_dict(old_layer.state_dict(), strict=False)
        model.model.layers[i] = new_layer

    # 劫持模型的 forward 方法，替换为我们自定义的 eval_kimi_model_forward，这样在评测时就会使用新的前向逻辑，允许我们在不修改原始模型代码的情况下，注入新的计算流程和数据流。
    # __get__把我们定义的函数绑定到模型实例上，使其成为模型的一个方法，这样在调用 model() 时就会执行 eval_kimi_model_forward 中定义的逻辑。
    model.model.forward = eval_kimi_model_forward.__get__(model.model)
    return model


# ================= 数据准备 =================
print("[INFO] 初始化测试环境...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

try:
    with open("src/haystack.txt", "r", encoding="utf-8") as f:
        real_text = f.read()
    print("[INFO] 成功加载本地语料库")
except FileNotFoundError:
    print("[WARN] 未找到本地语料，使用备用生成文本")
    real_text = "The Industrial Revolution was... " * 500

# input_ids是一个列表，包含了文本中每个 token 对应的 ID，这些 ID 是根据模型的词表映射生成的。
# 我们将这个列表切分成多个长度为 CHUNK_SIZE 的片段，每个片段都被包装成一个新的输入 prompt，包含了原始文本的一部分和一个固定的查询（needle），
# 以模拟模型在处理长上下文时的输入格式。这些切分后的输入将被用来评测模型在不同层数上的隐藏状态数值动态变化，以及最终的目标 token 预测性能。
full_tokens = tokenizer(real_text, add_special_tokens=False).input_ids
test_prompts_ids = []
needle = "The secret access code for the experiment is [Omeg_a77Cat_Beta-9]."
target_str = "[Omeg_a77Cat_Beta-9]"
target_ids = tokenizer(target_str, return_tensors="pt").input_ids[0].to(device)
target_len = target_ids.shape[0]

for i in range(0, min(len(full_tokens), CHUNK_SIZE * MAX_TEST_CHUNKS), CHUNK_SIZE):
    chunk_tokens = full_tokens[i : i + CHUNK_SIZE]
    if len(chunk_tokens) == CHUNK_SIZE:
        filler_decoded = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        prompt = f"{needle}\n\n{filler_decoded}\n\nNow, please tell me, the secret access code for the experiment is"
        inp_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        test_prompts_ids.append(inp_ids)

print(
    f"[INFO] 构建完成，共切分出 {len(test_prompts_ids)} 组上下文 (长度: {CHUNK_SIZE} tokens/组)"
)


# ================= 评估核心 =================
def run_evaluation_loop(model, model_name):
    print(f"\n[EVAL] 开始评估: {model_name}")
    model.eval()
    avg_target_loss = 0.0
    avg_layer_norms = np.zeros(len(model.model.layers))

    for run_idx, input_ids in enumerate(test_prompts_ids):
        current_run_norms = []
        layers = model.model.layers
        # hook是用于捕获层输出的钩子函数，get_norm_hook 是我们定义的一个函数，用于在每层前向传播结束后计算隐藏状态的 L2 范数，并将其存储在 current_run_norms 列表中。通过注册这个钩子，我们可以在不修改模型内部代码的情况下，动态地监控每一层的输出特征的数值动态变化，这对于分析模型在处理长上下文时的稳定性和表现非常有帮助。
        hooks = []

        def get_norm_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            current_norm = (
                torch.norm(hidden_states.detach().float(), dim=-1).mean().item()
            )
            current_run_norms.append(current_norm)

        for layer in layers:
            hooks.append(layer.register_forward_hook(get_norm_hook))

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        for hook in hooks:
            hook.remove()

        shift_logits = logits[..., -(target_len + 1) : -1, :].contiguous()
        shift_labels = input_ids[..., -target_len:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).item()

        avg_target_loss += loss
        avg_layer_norms += np.array(current_run_norms)
        print(
            f"       -> Chunk {run_idx + 1}/{len(test_prompts_ids)} Target Loss: {loss:.4f}"
        )
        torch.cuda.empty_cache()

    avg_target_loss /= len(test_prompts_ids)
    avg_layer_norms /= len(test_prompts_ids)
    print(f"[RESULT] {model_name} 平均 Target Loss: {avg_target_loss:.4f}")
    return avg_layer_norms.tolist(), avg_target_loss


results = {}

# 1. Baseline 测试
print("\n[PHASE 1] 加载 Baseline 模型...")
model_baseline = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, dtype=torch.bfloat16
).to(device)
state_dict_base = load_file(BASELINE_PATH)
model_baseline.load_state_dict(state_dict_base, strict=False)
results["Baseline"] = run_evaluation_loop(model_baseline, "Baseline")
del model_baseline
torch.cuda.empty_cache()

# 2. V3 测试
print("\n[PHASE 2] 加载 V3_AttnRes 模型并执行架构替换...")
config = AutoConfig.from_pretrained(BASE_MODEL_ID)
model_v3 = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.bfloat16).to(
    device
)
model_v3 = patch_model_for_eval(model_v3, config)
state_dict_v3 = load_file(V3_PATH)
model_v3.load_state_dict(state_dict_v3, strict=False)

for layer in model_v3.model.layers:
    # 这里是把我们新加的 gate 和 w_l 参数也搬到正确的设备和精度上，确保它们在评测时能够正常工作，不会因为数据类型不匹配而报错或性能下降。
    layer.attn_fusion.gate.data = layer.attn_fusion.gate.data.to(
        device, dtype=torch.bfloat16
    )
    layer.mlp_fusion.gate.data = layer.mlp_fusion.gate.data.to(
        device, dtype=torch.bfloat16
    )
    layer.attn_fusion.w_l.data = layer.attn_fusion.w_l.data.to(
        device, dtype=torch.bfloat16
    )
    layer.mlp_fusion.w_l.data = layer.mlp_fusion.w_l.data.to(
        device, dtype=torch.bfloat16
    )

results["V3_AttnRes"] = run_evaluation_loop(model_v3, "V3_AttnRes")
del model_v3
torch.cuda.empty_cache()

# ================= 绘图与出图 (Log-Transformed Linear Plot) =================
print("\n[INFO] 正在生成实验对比图表 (Linear vs Log-Transformed)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

baseline_norms, baseline_loss = results["Baseline"]
v3_norms, v3_loss = results["V3_AttnRes"]
layers_idx = np.arange(len(baseline_norms))

# ---- 图 1: 原始 L2 Norm (线性坐标) ----
# 直观展示数值随层数加深的累加膨胀效应，红蓝线在最后的差距代表了稳定性的提升
ax1.plot(
    layers_idx,
    baseline_norms,
    label=f"Baseline (Avg Loss: {baseline_loss:.3f})",
    marker="o",
    color="#e74c3c",
    linewidth=2,
    markersize=5,
)
ax1.plot(
    layers_idx,
    v3_norms,
    label=f"V3_AttnRes (Avg Loss: {v3_loss:.3f})",
    marker="s",
    color="#3498db",
    linestyle="--",
    linewidth=2,
    markersize=5,
)
ax1.set_title("Hidden State Norm (Linear Raw)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Layer Index", fontsize=10)
ax1.set_ylabel("Mean L2 Norm", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend()

# ---- 图 2: Log10 转换后的 Norm (线性显示) ----
# 🌟 这里按照你的意思：直接对数据取 Log10，不在坐标轴上做压缩。
# 这样能极大地拉伸底层数据，看清每一层门控注入后对数值动力学的微观影响。
log_baseline = np.log10(baseline_norms)
log_v3 = np.log10(v3_norms)

ax2.plot(
    layers_idx, log_baseline, label="Baseline", marker="o", color="#e74c3c", linewidth=2
)
ax2.plot(
    layers_idx,
    log_v3,
    label="V3_AttnRes",
    marker="s",
    color="#3498db",
    linestyle="--",
    linewidth=2,
)
ax2.set_title("Hidden State Norm (Log10 Transformed)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Layer Index", fontsize=10)
ax2.set_ylabel("Log10(Mean L2 Norm)", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend()

# 整体优化
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle(
    f"Architecture Dynamics Analysis: Llama-3.2-1B vs AttnRes V3\n(Evaluation on {len(test_prompts_ids)} Chunks of 3.5K real-world context)",
    fontsize=14,
)

# 保存文件
save_path = "kimi_norm_comparison_final.png"
plt.savefig(save_path, dpi=300)

print("-" * 60)
print(f"[DATA] Baseline Average Target Loss: {baseline_loss:.4f}")
print(f"[DATA] V3_AttnRes Average Target Loss: {v3_loss:.4f}")
print(f"[INFO] 最终评测报告图已生成并保存至: {save_path}")
print("-" * 60)
