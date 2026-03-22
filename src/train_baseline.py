"""
Llama-3.2-1B baseline 训练
数据: FineWeb-Edu npy
模型: LLM-Research/Llama-3.2-1B-Instruct

📊 模型规格:
├── 参数量: 1.23B (1,234,567,872)
├── 层数: 16层 Transformer Decoder
├── 隐藏维度: d_model=2048
├── 注意力头数: 16头 (head_dim=128)
├── 中间层维度: FFN=5632 (2.75x d_model)
├── 位置编码: RoPE (base=10000)
├── 激活函数: SiLU (Swish)
├── 归一化: RMSNorm (pre-norm)
├── 词汇表: 128k (Llama3 tokenizer)
├── 参数共享: embed_tokens ↔ lm_head.weight
└── 峰值显存: ~6GB (A100, batch=4, seq=1024)

⚙️ 训练配置:
├── 有效batch: 96 tokens (6×8×2 双卡)
├── 总tokens: 196.6M (2000步×1024×96)
├── 学习率: 2e-4 (cosine衰减)
├── 优化器: AdamW (β1=0.9, β2=0.95, ε=1e-8)
├── 权重衰减: 0.01
└── 梯度裁剪: 1.0
"""

"""
训练记录：
Step : 2000
loss 4.03517
lr 0.00019
ppl 56.5523
图像将导出在文件夹assets下，命名为train_loss_baseline.png和train_ppl_baseline.png
"""

import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv
from modelscope import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import logging as hf_logging
from transformers import get_cosine_schedule_with_warmup

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
    "output_dir": "/root/autodl-tmp/attnres-checkpoints/baseline",
    "seq_len": 1024,
    "batch_size": 6,        # per GPU
    "grad_accum": 8,        # 等效micro_size
    "lr": 2e-4,
    "max_steps": MAX_STEPS,
    #最开始发现loss起始会上升，有几点关注
    #（1）AdamW冷启动的盲目性
    #（2）2e-4，模型已经预训练好了，这个可能有点大，如果训练不稳定可以改小一点，比如 1e-4 或 5e-5，或者把 warmup_steps 增加到 MAX_STEPS//5
    #（3）数据分布切换导致的训练震荡，预热阶段更温和的学习率可以帮助模型更快适应新数据。
    "warmup_steps": MAX_STEPS//10, # 预热步数，10% 的总步数
    "log_every": 10,
    "save_every": 1000,
    "seed": 42,
    "max_files": 20,        
    "wandb_project": "AttnRes-Llama3.2-1B-Baseline"
}
# ================

# 这个 Dataset 假设每个 npy 文件都是一个一维的 token id 数组，我们把它们拼接起来，然后切成固定长度的 chunk 作为训练样本。
class NPYDataset(Dataset):
    def __init__(self, data_dir, seq_len, max_files):
        self.seq_len = seq_len
        files = sorted(Path(data_dir).glob("edufineweb_train_*.npy"))[:max_files]
        # 把文件全部读进内存并拼接，注意这里假设每个文件都是一维的 token id 数组
        # 如果文件太大，改成懒加载__getitem__版本，避免一次性占用过多内存
        all_tokens = np.concatenate([np.load(f) for f in files])
        # 切成固定长度的 chunk
        n = len(all_tokens) // seq_len
        self.data = torch.tensor(
            all_tokens[:n * seq_len].reshape(n, seq_len),
            dtype=torch.long
        )
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        #这里id和label一样，llama内部会自动shift，所以直接返回x作为input_ids和labels
        return {"input_ids": x, "labels": x.clone()}

def main():
    set_seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)

    #NOTE：accelerator 会自动处理分布式训练、混合精度等细节，我们只需要正常写训练代码即可。
    #NOTE：device_map="auto" 会自动把模型加载到 GPU 上流水传递，适合推理；
    #NOTE: 而accelertor.prepare() 会把模型、优化器、数据加载器等准备好，各自forward+backward，最后Allreduce平均梯度（DDP）
    #二者不能同时使用，推理时用device_map="auto"，训练时用accelerator.prepare()，不需要device_map参数。

    # 这里我们使用 bfloat16 混合精度，权重存储、前向、反向、梯度投用，tf32底层矩阵乘法会自动开启
    # 里面的accelerator.launch()自动识别卡的数量记录到accelerator.num_processes
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=CFG["grad_accum"],log_with="wandb")#wandb日志记录

    # 只在主进程打印
    accelerator.print("=== 加载模型 ===")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"],
        dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # ================= 显存优化与架构动机 =================
    # 1. 强制开启梯度检查点 (Gradient Checkpointing)。HuggingFace 默认是关闭的。
    # 2. 开启后，前向传播不再保存所有中间激活值，用大约 20% 的计算时间开销换取极大的显存节省，从而拉高 batch_size。
    # 3. 这也是 Kimi 论文中采用 Block-wise AttnRes 而非 Layer-wise 的核心动机。
    #    若采用逐层密集残差，配合 GC 会导致极度严重的重计算灾难（Recomputation Overhead）。
    #    通过划分 Block，我们只在块边界进行 Attention 融合，完美兼容了 GC 机制。
    model.gradient_checkpointing_enable()

    accelerator.print("=== 加载数据 ===")
    dataset = NPYDataset(CFG["data_dir"], CFG["seq_len"], CFG["max_files"])
    accelerator.print(f"加载前 {CFG['max_files']} 个 npy 文件...")
    accelerator.print(f"共 {len(dataset):,} 个样本，每个 {CFG['seq_len']} tokens")
    dataloader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,#训练时打乱数据，推理时不需要
        num_workers=2,# 磁盘->CPU->GPU，多进程预加载数据，减少GPU等待。
                    # 本脚本数据已预加载进内存，影响不大；
                    # 若改为懒加载（每次读磁盘），建议设4-8。
        pin_memory=True#加速数据传输到GPU
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=0.01
    )

    #lr调度器，先预热再余弦衰减，适合大多数训练场景，能让模型更快收敛，最终性能更好。
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=CFG["warmup_steps"],
        num_training_steps=CFG["max_steps"]
    )

    #DDP训练需要把模型、优化器、数据加载器等都交给accelerator.prepare()来处理，它会自动把模型加载到GPU上，处理梯度同步等细节。
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

    accelerator.print(f"\n=== 开始训练 ===")
    accelerator.print(f"GPU 数量: {accelerator.num_processes}")
    accelerator.print(f"每步有效 batch: {CFG['batch_size'] * CFG['grad_accum'] * accelerator.num_processes}")
    accelerator.print(f"总步数: {CFG['max_steps']}\n")

    # 初始化 wandb 追踪器，记录训练过程中的超参数和指标。每个进程都会调用，但只有主进程会真正连接到 wandb，其他进程会被 accelerator 屏蔽掉，避免重复记录。
    accelerator.init_trackers(
        project_name=CFG["wandb_project"], 
        config=CFG,
        #改的时候换个名字，就会在wandb上创建一个新的实验，不会覆盖之前的结果。
        init_kwargs={"wandb": {"name": "Baseline_Run"}}
        )

    model.train()
    data_iter = iter(dataloader)
    running_loss = 0.0

    t0 = time.time()

    for step in range(CFG["max_steps"]):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)#重新创建dataloader时shuffle数据
            batch = next(data_iter)

        #NOTE:每步先自己算梯度累加 → ✅ 每步都 backward，累积梯度
        #NOTE:每步 loss 自动 /8 → ✅ 每步 loss 乘以 1/GRAD_ACCUM（梯度缩放）
        #NOTE:到最后一刻这个梯度先和其他7个平均 → ✅ 第8步 AllReduce（平均）
        #NOTE:再 step，再置0 → ✅ optimizer.step() + zero_grad()

        # accelerator.accumulate() 会自动处理梯度累积，等效于把 batch_size 扩大 GRAD_ACCUM 倍，同时在每 GRAD_ACCUM 步进行一次梯度更新。
        with accelerator.accumulate(model):
            outputs = model(**batch)#batch是字典，**是解包，相当于model(input_ids=batch["input_ids"], labels=batch["labels"])
            #会自动除以 GRAD_ACCUM 来平均 loss，适合混合精度训练，避免数值不稳定。
            loss = outputs.loss
            #自动处理bf16混合精度的反向传播，内部会自动缩放梯度以避免数值下溢。
            accelerator.backward(loss)
            #梯度裁剪
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            #如果当前不是累积的最后一步，optimizer.step()会被accelerator屏蔽掉，等到最后一步才真正更新参数。
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        #分布式训练中，每个进程计算的 loss 是局部的，需要用 accelerator.reduce() 来平均所有进程的 loss，得到全局的平均 loss。
        dist_loss = accelerator.reduce(loss.detach(), reduction="mean")
        running_loss += dist_loss.item()

        # ================= 日志打印块 =================
        if (step + 1) % CFG["log_every"] == 0:
            avg_loss = running_loss / CFG["log_every"]
            ppl = math.exp(min(avg_loss, 20)) # 困惑度，相当于平均每个 token 的预测分布中，正确答案的概率的倒数。越低越好，20以上就没什么意义了。

            t1 = time.time()
            dt = t1 - t0

            #total_tokens = log间隔 * grad_accum * 显卡数 * 每卡batch * 序列长度
            total_tokens = CFG["log_every"] * CFG["grad_accum"] * accelerator.num_processes * CFG["batch_size"] * CFG["seq_len"]
            tok_sec = total_tokens / dt
            step_time_ms = (dt / CFG["log_every"]) * 1000 # 换算成毫秒
            current_lr = lr_scheduler.get_last_lr()[0] # 获取当前学习率

            #只在master进程打印，其他进程会被accelerator屏蔽掉，避免重复输出。
            accelerator.print(
                f"Step {step+1:4d}/{CFG['max_steps']} | "
                f"lr={current_lr:.2e} | "
                f"loss={avg_loss:.4f} | ppl={ppl:.2f} | "
                f"dt={step_time_ms:.0f}ms | tok/sec={tok_sec:,.0f}"
                        )            
            running_loss = 0.0

            t0 = time.time() #重置计时器

            accelerator.log({
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "train/step": step + 1,
                "train/lr": current_lr
            }, step=step + 1)

        # ================= 模型保存块 =================
        # 独立出来，只要这一步是 save_every (比如400，是8的倍数) 的整数倍就会触发
        if (step + 1) % CFG["save_every"] == 0:
            # 加上这行，让跑得快的进程在这里等一下跑得慢的，保证安全握手
            accelerator.wait_for_everyone() 
            accelerator.save_state(CFG["output_dir"])
            # 只让主进程报告好消息，保持控制台干净
            if accelerator.is_main_process:
                print(f"✅ Step {step + 1}: Checkpoint 安全保存完成！")
    
    accelerator.print("✅ 训练完成！checkpoint保存在:", CFG["output_dir"])
    accelerator.end_training()

if __name__ == "__main__":
    main()