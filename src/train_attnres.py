import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from dotenv import load_dotenv
from modelscope import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import logging as hf_logging
from transformers import get_cosine_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

# 导入魔改层
from modeling_attnres_llama import KimiLlamaDecoderLayer

# 实验记录
# (1)V1发现门控没有作用/但仍然比baseline好，说明AttnRes的历史融合确实在起作用，但门控参数可能需要调整
# step:2000 gate:-5.0(0.0067) lr:2e-4(linear-Cosine) 门控/打分权重bf16 全局学习率
# loss:3.98106
# ppl:53.57398

# (2)V2
# step:2000 gate:-2.0(0.1192) lr:2e-4(linear-Cosine) 门控/打分权重float32 全局学习率
# gates/layer_0_attn_prob 0.1192
# gates/layer_0_mlp_prob 0.1192
# gates/layer_8_attn_prob 0.11877
# gates/layer_8_mlp_prob 0.118
# gates/layer_15_attn_prob 0.12057
# gates/layer_15_mlp_prob 0.11887
# train/loss 3.99995
# train/ppl 54.59554

#(3)V3
# step:2000 gate:-2.0(0.1192) lr:2e-4(linear-Cosine) 门控/打分权重float32 gate50倍全局学习率

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_MODE"] = os.getenv("WANDB_MODE")

hf_logging.set_verbosity_error()
logging.getLogger("accelerate").setLevel(logging.ERROR)

# ===== 配置 =====
MAX_STEPS = 2000
CFG = {
    "model_id": "LLM-Research/Llama-3.2-1B-Instruct",
    "data_dir": "/root/autodl-tmp/edu_fineweb10B",
    "output_dir": "/root/autodl-tmp/attnres-checkpoints/attnres_v1",
    "seq_len": 1024,
    "batch_size": 6,
    "grad_accum": 8,
    "lr": 2e-4,
    "max_steps": MAX_STEPS,
    "warmup_steps": MAX_STEPS // 10,
    "log_every": 10,
    "save_every": 1000,
    "seed": 42,
    "max_files": 20,
    "wandb_project": "AttnRes-Llama3.2-1B-Experiment",
}


class NPYDataset(Dataset):
    def __init__(self, data_dir, seq_len, max_files):
        self.seq_len = seq_len
        files = sorted(Path(data_dir).glob("edufineweb_train_*.npy"))[:max_files]
        all_tokens = np.concatenate([np.load(f) for f in files])
        n = len(all_tokens) // seq_len
        self.data = torch.tensor(
            all_tokens[: n * seq_len].reshape(n, seq_len), dtype=torch.long
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x, "labels": x.clone()}


def patch_model_with_kimi(model, config, accelerator):
    accelerator.print("🔪 正在执行模型层级手术...")
    for i in range(len(model.model.layers)):
        old_layer = model.model.layers[i]
        new_layer = KimiLlamaDecoderLayer(config, layer_idx=i)
        # 1. 这一步把整个新层变成了 bfloat16
        new_layer.to(old_layer.input_layernorm.weight.dtype)
        new_layer.load_state_dict(old_layer.state_dict(), strict=False)
        
        # 🌟 2. 【问题四】：把门控和打分权重强制提权回 float32，因为bf16会为了了节省显存而把小权重压成0，导致门控失效和权重失效。我们需要让它们保持在float32的精度上，才能正常学习到微妙的权重调整。
        new_layer.attn_fusion.gate.data = new_layer.attn_fusion.gate.data.float()
        new_layer.mlp_fusion.gate.data = new_layer.mlp_fusion.gate.data.float()
        new_layer.attn_fusion.w_l.data = new_layer.attn_fusion.w_l.data.float()
        new_layer.mlp_fusion.w_l.data = new_layer.mlp_fusion.w_l.data.float()
        model.model.layers[i] = new_layer
    accelerator.print("✅ 手术圆满完成！")
    return model


# ==========================================
# 🧠 神经中枢劫持 (完美门控 + 原生 SDPA 遮罩)
# ==========================================
def kimi_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_len = inputs_embeds.shape[:2]
    device = inputs_embeds.device

    # 1. 确保 position_ids 存在
    if position_ids is None:
        position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

    # 2. 计算 RoPE 并对齐 64 维
    # 【问题一】这里解决了好久，原因是Llama每个注意力头确实是64维，但是RoPE数学特性前半部分和后半部分有某种对称/重复
    # huggingface直接返回32维的cos/sin，如果我们不劫持模型，内部会自己展开成64维；但我们劫持了模型，外部又没有展开，就会导致维度不匹配的错误。
    cos, sin = self.rotary_emb(inputs_embeds, position_ids)
    if cos.shape[-1] < self.layers[0].self_attn.head_dim:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    position_embeddings = (cos, sin)

    hidden_states = inputs_embeds
    history_cache = []
    block_size = 4

    for i, layer_module in enumerate(self.layers):
        # ⚠️ 必须用 list() 浅拷贝，防止梯度检查点报错
        # 【问题二】python里的list()函数是一个浅拷贝操作，它会创建一个新的列表对象，但列表中的元素仍然是原来列表中元素的引用。
        # 如果我们直接传 history_cache 给模型，在前向传播时不断append新的hidden_states到history_cache中，
        # history_cache对象本身没有改变（仍然是同一个列表对象），但它的内容在不断变化，这会导致梯度检查点机制无法正确追踪历史状态的变化，从而报错。
        current_history = list(history_cache) if len(history_cache) > 0 else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(h_states, hist, pos_emb, attn_msk, p_ids):
                    return module(
                        h_states,
                        history=hist,
                        position_embeddings=pos_emb,
                        attention_mask=attn_msk,
                        position_ids=p_ids,
                        use_cache=False,
                    )

                return custom_forward

            layer_outputs = self._gradient_checkpointing_func(
                create_custom_forward(layer_module),
                hidden_states,
                current_history,
                position_embeddings,
                attention_mask,  # ✨ 直接传 None，让 PyTorch SDPA 自动处理 Causal Mask
                position_ids,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,  # ✨ 直接传 None
                position_ids=position_ids,
                history=current_history,
                position_embeddings=position_embeddings,
                use_cache=False,
            )

        hidden_states = layer_outputs[0]

        if (i + 1) % block_size == 0:
            # 这里detach()是为了切断历史状态的梯度流，防止内存爆炸，同时也符合我们设计的“每个block只关注当前状态和历史状态的融合”，不让历史状态的梯度回传到更早的层。
            # 如果不detach()，历史状态的梯度会一直回传到最初的输入层，导致内存占用随着训练步数增加而不断增长，最终可能导致OOM错误。
            history_cache.append(hidden_states.detach())

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states)


def main():
    set_seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)
    # 【问题三】：DDP在开局会清点所有参数，我们每4次才存一次历史状态，这意味着第 1、2、3、4 层的 BlockAttnRes 模块，收到的是空历史，直接触发了 if not history_states: return current_state，导致这前 4 层的 w_l 和 norm 参数根本没参与计算，也就没有梯度
    # 如果不告诉DDP这些参数是“偶尔用一下”的，就会因为某些历史状态在某些步数没有被使用到而报错说“参数没有被使用”。加上DistributedDataParallelKwargs(find_unused_parameters=True)这个配置后，DDP就会允许有一些参数在某些步数没有被使用，从而避免了这个问题。
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=CFG["grad_accum"],
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs],  # 👈 加上这行通行证
    )

    accelerator.print("=== 加载模型 ===")
    model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"],
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = patch_model_with_kimi(model, model.config, accelerator)
    model.model.forward = kimi_model_forward.__get__(model.model)
    model.gradient_checkpointing_enable()

    accelerator.print("=== 加载数据 ===")
    dataset = NPYDataset(CFG["data_dir"], CFG["seq_len"], CFG["max_files"])
    dataloader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    #optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)

    #这里我们认为，模型中除了门控和打分权重以外的其他参数（比如原有的线性层权重、RMSNorm权重等）都已经在预训练中学到了比较好的表示，我们不希望它们在微调过程中发生太大的变化，所以保持较低的学习率；而新加的门控参数和打分权重一开始是随机初始化的，需要快速学习到合理的值，所以给它们一个较高的学习率，这样可以让模型更快地适应新的融合机制，提高训练效率。
    # 🌟 === 新写法：为新增参数分组提速 ===
    custom_params = []
    base_params = []
    for name, param in model.named_parameters():
        if "fusion" in name:  # 自动匹配我们新加的 attn_fusion 和 mlp_fusion 模块
            custom_params.append(param)
        else:
            base_params.append(param)

    # 基础模型用 2e-4，新加的器官用 1e-2 (提速 50 倍)
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": CFG["lr"]},
            {"params": custom_params, "lr": CFG["lr"] * 50} 
        ],
        weight_decay=0.01
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CFG["warmup_steps"],
        num_training_steps=CFG["max_steps"],
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    accelerator.init_trackers(
        project_name=CFG["wandb_project"],
        config=CFG,
        init_kwargs={"wandb": {"name": "V3-AttnRes_Gate(-2)_50xLR" }},  # 🌟 给每次实验的 WandB run 起个名字，方便区分
    )

    model.train()
    data_iter = iter(dataloader)
    running_loss, t0 = 0.0, time.time()

    for step in range(CFG["max_steps"]):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"]
        device = input_ids.device
        seq_len = input_ids.shape[1]

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        running_loss += accelerator.reduce(loss.detach(), reduction="mean").item()

        if (step + 1) % CFG["log_every"] == 0:
            avg_loss = running_loss / CFG["log_every"]
            ppl = math.exp(min(avg_loss, 20))

            # 计算时间与速度
            t1 = time.time()
            dt = t1 - t0
            total_tokens = (
                CFG["log_every"]
                * CFG["grad_accum"]
                * accelerator.num_processes
                * CFG["batch_size"]
                * CFG["seq_len"]
            )
            tok_sec = total_tokens / dt
            step_time_ms = (dt / CFG["log_every"]) * 1000

            # 获取当前学习率
            current_lr = lr_scheduler.get_last_lr()[0]

            # 获取门控值并记录
            gate_stats = {}
            raw_model = accelerator.unwrap_model(model)

            # 抓取第0层(底层), 第8层(中层), 第15层(高层)的gate值
            for i in [0, 8, 15]:
                attn_gate = raw_model.model.layers[i].attn_fusion.gate.item()
                mlp_gate = raw_model.model.layers[i].mlp_fusion.gate.item()

                # 算Sigmoid，转成真实的概率值记录下来
                gate_stats[f"gates/layer_{i}_attn_prob"] = torch.sigmoid(
                    torch.tensor(attn_gate)
                ).item()
                gate_stats[f"gates/layer_{i}_mlp_prob"] = torch.sigmoid(
                    torch.tensor(mlp_gate)
                ).item()

            # 取中层(第8层)的 Attention 门控值在终端显示
            mid_layer_gate = gate_stats["gates/layer_8_attn_prob"]

            accelerator.print(
                f"Step {step + 1:4d}/{CFG['max_steps']} | "
                f"lr={current_lr:.2e} | "
                f"loss={avg_loss:.4f} | ppl={ppl:.2f} | "
                f"Gate8={mid_layer_gate:.4f} | "
                f"dt={step_time_ms:.0f}ms | tok/sec={tok_sec:,.0f}"
            )

            # 同步更新给 WandB
            log_data = {
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "train/step": step + 1,
                "train/lr": current_lr,
            }
            log_data.update(gate_stats)  # 把门控统计数据也加入日志

            accelerator.log(log_data, step=step + 1)

            # 重置累加器和计时器
            running_loss = 0.0
            t0 = time.time()

        if (step + 1) % CFG["save_every"] == 0:
            accelerator.wait_for_everyone()
            accelerator.save_state(CFG["output_dir"])

    accelerator.end_training()


if __name__ == "__main__":
    main()
