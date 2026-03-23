# 这里写AttnRes的部件，主要是为了在LlamaDecoderLayer中替换原有的残差连接为AttnRes。
import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from typing import Optional, Tuple, List, Any

"""核心思路：我们在验证AttnRes的时候，做了几点和论文不一样的地方
由于是在已经预训练的Llama-3.2基础上微调，我们必须非常小心地保护原有的预训练知识，不能让模型在训练初期就因为输入全0而失忆。因此：
(1)我们保留了原版Llama的残差主干道，而不是像Kimi那样在每个Block边界清零，这样可以避免模型失忆，稳定训练。
(2)我们将AttnRes融合出的历史特征作为一种“全局视野眼镜”喂给Attention/MLP子层，但子层算出的知识增量必须加回到原始主干道上，这样既安全地继承了Llama的预训练知识，又平稳地植入了Kimi的长上下文跨层记忆科技。
(3)发现只实现(1)(2)模型会在中期loss突然升高，于是我们考虑加gate来缓慢释放对历史的视野，但是削弱了AttnRes的作用
"""


# 对历史block的输出进行残差融合
class BlockAttnRes(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # 伪Query，全零初始化保证初期softmax后权重均匀分布，不会过早偏向某个历史block
        self.w_l = nn.Parameter(torch.zeros(hidden_size))
        self.norm = LlamaRMSNorm(
            hidden_size
        )  # 每个block都要做RMSNorm归一化，保证打分时不同block的状态在同一尺度上，避免某个block因为数值较大而占主导地位。注意这里的RMSNorm是LlamaRMSNorm，和LlamaDecoderLayer中使用的norm保持一致。
        # 门控标量，初始化为 0，sigmoid(0)=0.5，
        # 但配合下面的逻辑，初始时 fused 权重极小
        # 后面可以改大一点试一试，或者干脆直接固定一个小值（比如0.01），让模型专注于利用历史信息，看看效果如何。
        self.gate = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007

    def forward(self, history_states, current_state):
        # 如果是第一层，返回当前状态
        if not history_states:
            return current_state

        # 1.(num_blocks+1, batch_size, seq_len, hidden_size)
        # torch.stack在dim=0上创造一个新维度n+1，torch.cat在dim=0上直接拼接成n维，区别在于前者会增加一个维度，后者不会。这里我们需要一个新的维度来表示历史block和当前block，所以用stack。
        all_states = torch.stack(history_states + [current_state], dim=0)

        # 2.归一化，只用于进行K打分，不改变用来融合的all_states
        normed_states = self.norm(all_states)

        # 3.打分 , 维度是(num_blocks+1, batch_size, seq_len)，每个位置对历史block和当前block的权重
        # NOTE:爱因斯坦求和约定torch.einsum：箭头左边是输入张量的维度，箭头右边是输出张量的维度，
        # NOTE:匹配相乘：箭头左侧不同张量出现相同字母，说明这个维度上逐元素相乘；
        # NOTE:消失相加：如果在箭头左侧出现了某个字母，但在箭头右侧消失了，说明 PyTorch 会在相乘之后，沿着这个维度把所有值加起来（求和降维）。
        logits = torch.einsum("nbsd,d->nbs", normed_states, self.w_l)

        # 4.softmax归一化，维度不变，权重和为1
        alphas = torch.softmax(logits, dim=0)

        # 5.加权求和，得到融合后的状态，维度(batch_size, seq_len, hidden_size)
        fused_state = torch.einsum("nbsd,nbs->bsd", all_states, alphas)
        g = torch.sigmoid(self.gate)

        return (1 - g) * current_state + g * fused_state


# 继承原版LlamaDecoderLayer，并“强行”缝入两个 BlockAttnRes (Attention前 和 MLP前)
class KimiLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.attn_fusion = BlockAttnRes(config.hidden_size)
        self.mlp_fusion = BlockAttnRes(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        history: Optional[List[torch.Tensor]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # 吸收额外参数，避免签名不匹配
    ) -> Tuple[torch.Tensor]:
        """NOTE
        =========================================================================================
        🚀 Kimi 架构与 Llama 微调的融合设计说明
        =========================================================================================
        【1. 我们与 Kimi 原版的差异：残差主干道的去留】
        - Kimi 原版 (From Scratch)：为了极致压缩显存，Kimi 在每个 Block 边界会极其残忍地将残差主干道清零
          (对应伪代码的 b_n^0 := 0 或 partial_block = None)，强迫模型在下一个 Block 必须依赖 BlockAttnRes
          融合出的历史特征来重新构建残差流。
        - 我们的魔改版 (Fine-tuning)：我们绝不清空残差主干道，而是保留了 Llama 贯穿 16 层的完整残差连接。

        【2. 为什么要这么写？(决定生死的 Trade-off)】
        - Llama-3.2 已经是一个经过海量数据预训练的成熟模型，它脑海里的权重极度依赖一条连续不断的残差主干道。
        - 如果我们盲目照搬 Kimi 的“清零”机制，Llama 会因为主干道断裂、遭遇大量全 0 输入而瞬间“失忆”，
          导致第 1 步训练的 Loss 直接爆炸，甚至报 NaN。
        - 因此，我们采用了一种更优雅的“微创手术 (Adapter 范式)”：
          我们将 AttnRes 融合出的历史特征 (h_fused) 作为一种“全局视野眼镜”喂给 Attention/MLP 子层；
          但子层算出的知识增量 (attn_out / mlp_out)，必须老老实实地加回到那条未被篡改的原始主干道 (hidden_states) 上。

        结论：这样既安全地继承了 Llama 强大的预训练知识，又极其平稳地植入了 Kimi 的长上下文跨层记忆科技！
        =========================================================================================
        """
        residual = hidden_states

        # Kimi 融合：Attention 前融合历史
        h_fused_attn = self.attn_fusion(history or [], hidden_states)

        h = self.input_layernorm(h_fused_attn)

        # 调用 Attention，显式传递补齐后的 position_embeddings
        attn_outputs = self.self_attn(
            hidden_states=h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        attn_output = attn_outputs[0]

        # 加回原始残差（Llama 风格）
        mid_state = residual + attn_output

        # MLP 前再融合一次
        h_fused_mlp = self.mlp_fusion(history or [], mid_state)

        h = self.post_attention_layernorm(h_fused_mlp)
        mlp_output = self.mlp(h)

        # 加回残差
        output = mid_state + mlp_output

        return (output,)
