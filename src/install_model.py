#从modelscope加载llama3.2-1B-Instruct模型，并进行简单测试，确认模型是否正确加载和运行。
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "LLM-Research/Llama-3.2-1B-Instruct"

print("=== 测试确认的模型 ===")
try:
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ 成功!")
    print(f"参数量: {model.num_parameters():,}")
    print(f"层数: {len(model.model.layers) if hasattr(model.model, 'layers') else 'N/A'}")
    print(f"hidden_size: {getattr(model.config, 'hidden_size', 'N/A')}")
    
    # 简单测试
    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    print("前向 OK")
    
except Exception as e:
    print(f"❌ 错误: {e}")
