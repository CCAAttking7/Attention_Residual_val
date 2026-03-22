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
    "wandb_project": "AttnRes-Llama3.2-1B-Experiment"
}

class NPYDataset(Dataset):
    def __init__(self, data_dir, seq_len, max_files):
        self.seq_len = seq_len
        files = sorted(Path(data_dir).glob("edufineweb_train_*.npy"))[:max_files]
        all_tokens = np.concatenate([np.load(f) for f in files])
        n = len(all_tokens) // seq_len
        self.data = torch.tensor(all_tokens[:n * seq_len].reshape(n, seq_len), dtype=torch.long)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x, "labels": x.clone()}

def patch_model_with_kimi(model, config, accelerator):
    accelerator.print("🔪 正在执行模型层级手术...")
    for i in range(len(model.model.layers)):
        old_layer = model.model.layers[i]
        new_layer = KimiLlamaDecoderLayer(config, layer_idx=i)
        new_layer.to(old_layer.input_layernorm.weight.dtype)
        new_layer.load_state_dict(old_layer.state_dict(), strict=False)
        model.model.layers[i] = new_layer
    accelerator.print("✅ 手术圆满完成！")
    return model

# ==========================================
# 🧠 神经中枢劫持 (RoPE 维度对齐完全体)
# ==========================================
# def kimi_model_forward(
#     self, input_ids=None, attention_mask=None, position_ids=None,
#     past_key_values=None, inputs_embeds=None, use_cache=None,
#     output_attentions=None, output_hidden_states=None,
#     return_dict=None, cache_position=None, **kwargs
# ):
#     if inputs_embeds is None:
#         inputs_embeds = self.embed_tokens(input_ids)

#     batch_size, seq_len = inputs_embeds.shape[:2]
#     device = inputs_embeds.device

#     # ===== 关键修复：正确生成 position_ids =====
#     if position_ids is None:
#         position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

#     # ===== 关键修复：正确生成 cache_position =====
#     if cache_position is None:
#         cache_position = torch.arange(seq_len, device=device)

#     # ===== 关键修复：正确生成 4D causal attention mask =====
#     # 这是你原版代码完全缺失的部分
#     causal_mask = self._update_causal_mask(
#         attention_mask, inputs_embeds, cache_position,
#         past_key_values, output_attentions
#     )

#     # ===== RoPE =====
#     cos, sin = self.rotary_emb(inputs_embeds, position_ids)
#     if cos.shape[-1] < self.layers[0].self_attn.head_dim:
#         cos = torch.cat([cos, cos], dim=-1)
#         sin = torch.cat([sin, sin], dim=-1)

#     position_embeddings = (cos, sin)
#     hidden_states = inputs_embeds
#     history_cache = []
#     block_size = 4

#     for i, layer_module in enumerate(self.layers):
#         current_history = list(history_cache) if history_cache else None

#         if self.gradient_checkpointing and self.training:
#             # gradient checkpointing 兼容写法
#             def create_custom_forward(module):
#                 def custom_forward(h_states, hist, pos_emb, attn_msk, p_ids):
#                     return module(
#                         h_states, history=hist,
#                         position_embeddings=pos_emb,
#                         attention_mask=attn_msk,
#                         position_ids=p_ids,
#                         use_cache=False
#                     )
#                 return custom_forward

#             layer_outputs = self._gradient_checkpointing_func(
#                 create_custom_forward(layer_module),
#                 hidden_states, current_history,
#                 position_embeddings, causal_mask, position_ids
#             )
#         else:
#             layer_outputs = layer_module(
#                 hidden_states,
#                 attention_mask=causal_mask,  # 用处理过的 causal_mask
#                 position_ids=position_ids,
#                 history=current_history,
#                 position_embeddings=position_embeddings,
#                 use_cache=False
#             )

#         hidden_states = layer_outputs[0]
#         # detach 是必要的（显存），但配合门控机制，
#         # AttnRes 模块自身的参数仍然能通过当前 block 内的梯度学习
#         if (i + 1) % block_size == 0:
#             history_cache.append(hidden_states.detach())
#     hidden_states = self.norm(hidden_states)
#     return BaseModelOutputWithPast(last_hidden_state=hidden_states)
# ==========================================
# 🧠 神经中枢劫持 (完美门控 + 原生 SDPA 遮罩)
# ==========================================
def kimi_model_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None, **kwargs):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_len = inputs_embeds.shape[:2]
    device = inputs_embeds.device

    # 1. 确保 position_ids 存在
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # 2. 计算 RoPE 并对齐 64 维
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
        current_history = list(history_cache) if len(history_cache) > 0 else None
        
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(h_states, hist, pos_emb, attn_msk, p_ids):
                    return module(h_states, history=hist, position_embeddings=pos_emb, 
                                  attention_mask=attn_msk, position_ids=p_ids, use_cache=False)
                return custom_forward
            
            layer_outputs = self._gradient_checkpointing_func(
                create_custom_forward(layer_module),
                hidden_states,
                current_history,
                position_embeddings,
                attention_mask, # ✨ 直接传 None，让 PyTorch SDPA 自动处理 Causal Mask
                position_ids
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask, # ✨ 直接传 None
                position_ids=position_ids,
                history=current_history,
                position_embeddings=position_embeddings,
                use_cache=False
            )
            
        hidden_states = layer_outputs[0]

        if (i + 1) % block_size == 0:
            history_cache.append(hidden_states.detach())

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states)

def main():
    set_seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        mixed_precision="bf16", 
        gradient_accumulation_steps=CFG["grad_accum"], 
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs] # 👈 加上这行通行证
    )

    accelerator.print("=== 加载模型 (离线优先) ===")
    model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"], 
        dtype=torch.bfloat16, 
        trust_remote_code=True,
        local_files_only=True # 如果报错没有文件，请删掉这一行
    )
    model = patch_model_with_kimi(model, model.config, accelerator)
    model.model.forward = kimi_model_forward.__get__(model.model)
    model.gradient_checkpointing_enable()

    accelerator.print("=== 加载数据 ===")
    dataset = NPYDataset(CFG["data_dir"], CFG["seq_len"], CFG["max_files"])
    dataloader = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=CFG["warmup_steps"], num_training_steps=CFG["max_steps"])

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    accelerator.init_trackers(project_name=CFG["wandb_project"], config=CFG, init_kwargs={"wandb": {"name": "AttnRes_Run_v1"}})

    model.train()
    data_iter = iter(dataloader)
    running_loss, t0 = 0.0, time.time()

    for step in range(CFG["max_steps"]):
        try: batch = next(data_iter)
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
            if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), 1.0)
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
            total_tokens = CFG["log_every"] * CFG["grad_accum"] * accelerator.num_processes * CFG["batch_size"] * CFG["seq_len"]
            tok_sec = total_tokens / dt
            step_time_ms = (dt / CFG["log_every"]) * 1000 
            
            # 获取当前学习率
            current_lr = lr_scheduler.get_last_lr()[0] 

            # 原汁原味的完美打印
            accelerator.print(
                f"Step {step+1:4d}/{CFG['max_steps']} | "
                f"lr={current_lr:.2e} | "
                f"loss={avg_loss:.4f} | ppl={ppl:.2f} | "
                f"dt={step_time_ms:.0f}ms | tok/sec={tok_sec:,.0f}"
            )            
            
            # 同步更新给 WandB
            accelerator.log({
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "train/step": step + 1,
                "train/lr": current_lr
            }, step=step + 1)
            
            # 重置累加器和计时器
            running_loss = 0.0
            t0 = time.time()

        if (step + 1) % CFG["save_every"] == 0:
            accelerator.wait_for_everyone()
            accelerator.save_state(CFG["output_dir"])
    
    accelerator.end_training()

if __name__ == "__main__": main()