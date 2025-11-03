"""
修复 DeepSeek-OCR 模型文件以支持 Intel XPU (Intel Arc GPU)
将所有硬编码的 .cuda() 和 torch.autocast("cuda") 替换为设备无关的版本
"""

import re
import os

# 目标文件路径（根据您的错误信息）
model_file = r'D:\huggingface\modules\transformers_modules\deepseek_ai\DeepSeek_OCR\_c968b433af61a059311cbf8997765023806a24d\modeling_deepseekocr.py'

print("=" * 60)
print("修复 DeepSeek-OCR 以支持 Intel XPU (Arc GPU)")
print("=" * 60)

# 检查文件是否存在
if not os.path.exists(model_file):
    print(f"[ERROR] 找不到文件 {model_file}")
    print("请检查路径是否正确")
    exit(1)

print(f"\n正在读取模型文件...")
print(f"路径: {model_file}")

with open(model_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 备份原文件
backup_file = model_file + '.backup'
if not os.path.exists(backup_file):
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[OK] 已备份原文件到: {backup_file}")
else:
    print(f"[OK] 备份文件已存在")

# 统计修改前的数量
cuda_count_before = content.count('.cuda()')
autocast_count_before = content.count('torch.autocast("cuda"')

print(f"\n修改前:")
print(f"  .cuda() 调用: {cuda_count_before} 处")
print(f"  torch.autocast('cuda') 调用: {autocast_count_before} 处")

# ===== 修改 1: 在 infer 方法开始处添加 device 和 device_type 定义 =====
device_detect_code = """    def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
        self.disable_torch_init()

        os.makedirs(output_path, exist_ok=True)"""

device_detect_code_fixed = """    def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
        self.disable_torch_init()
        
        # Detect device the model is on
        device = next(self.parameters()).device
        device_type = device.type if device.type != 'cpu' else 'cpu'
        # Detect model dtype
        model_dtype = next(self.parameters()).dtype

        os.makedirs(output_path, exist_ok=True)"""

if device_detect_code in content:
    content = content.replace(device_detect_code, device_detect_code_fixed)
    print("[OK] 已在 infer() 方法中添加 device 定义")
else:
    print("[WARN] 未找到 infer() 方法的目标代码，尝试其他方法...")

# ===== 修改 2: 在 forward 方法开始处添加 device 定义 =====
# 尝试匹配不同数量的空行
forward_patterns = [
    ("""    ) -> Union[Tuple, BaseModelOutputWithPast]:




        if inputs_embeds is None:""", 
     """    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Detect device for this forward pass
        if input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device
        else:
            device = next(self.parameters()).device

        if inputs_embeds is None:"""),
    ("""    ) -> Union[Tuple, BaseModelOutputWithPast]:



        if inputs_embeds is None:""",
     """    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Detect device for this forward pass
        if input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device
        else:
            device = next(self.parameters()).device

        if inputs_embeds is None:"""),
    ("""    ) -> Union[Tuple, BaseModelOutputWithPast]:


        if inputs_embeds is None:""",
     """    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Detect device for this forward pass
        if input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device
        else:
            device = next(self.parameters()).device

        if inputs_embeds is None:"""),
]

forward_modified = False
for forward_start, forward_start_fixed in forward_patterns:
    if forward_start in content:
        content = content.replace(forward_start, forward_start_fixed)
        print("[OK] 已在 forward() 方法中添加 device 定义")
        forward_modified = True
        break

if not forward_modified:
    print("[WARN] 未找到 forward() 方法的目标代码，跳过")

# ===== 修改 3: 将 .cuda() 替换为 .to(device) =====
original_content = content
content = content.replace('.cuda()', '.to(device)')
if content != original_content:
    print(f"[OK] 已将所有 .cuda() 替换为 .to(device)")
else:
    print(f"[WARN] 未找到 .cuda() 调用")

# ===== 修改 4: 将 torch.autocast("cuda") 替换为 torch.autocast(device_type) =====
original_content = content
content = content.replace('torch.autocast("cuda"', 'torch.autocast(device_type')
if content != original_content:
    print(f"[OK] 已将所有 torch.autocast('cuda') 替换为 torch.autocast(device_type)")
else:
    print(f"[WARN] 未找到 torch.autocast('cuda') 调用")

# 统计修改后
cuda_count_after = content.count('.cuda()')
to_device_count = content.count('.to(device)')
autocast_count_after = content.count('torch.autocast("cuda"')
autocast_device_type = content.count('torch.autocast(device_type')

print(f"\n修改后:")
print(f"  .cuda() 调用: {cuda_count_after} 处 (应该是 0)")
print(f"  .to(device) 调用: {to_device_count} 处 (新增)")
print(f"  torch.autocast('cuda') 调用: {autocast_count_after} 处 (应该是 0)")
print(f"  torch.autocast(device_type) 调用: {autocast_device_type} 处 (新增)")

# 写入修改后的内容
with open(model_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n" + "=" * 60)
if cuda_count_after == 0 and autocast_count_after == 0:
    print("[OK] 修复完成！模型现在支持 Intel XPU (Arc GPU)")
else:
    print("[WARN] 修复完成，但可能还有部分未处理")
print("=" * 60)
print(f"\n如需恢复原文件，运行:")
print(f'  copy "{backup_file}" "{model_file}"')
print("\n现在可以运行 pdf_to_markdown.py 了!")

