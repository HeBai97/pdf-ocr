from transformers import AutoModel, AutoTokenizer
import torch
import os
from PIL import Image
import fitz  # PyMuPDF
import sys
from io import StringIO
import re

# 尝试导入IPEX
try:
    import intel_extension_for_pytorch as ipex
    ipex_available = True
except (ImportError, OSError) as e:
    ipex_available = False
    print(f"IPEX不可用: {type(e).__name__}")

# 尝试使用XPU（Intel Arc），如果不可用则使用CPU
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = 'xpu'
    print(f"使用Intel XPU: {torch.xpu.get_device_name(0)}")
    if not ipex_available:
        print("警告: IPEX未安装，性能可能受影响")
else:
    device = 'cpu'
    print("XPU不可用，使用CPU模式")

# 清理OCR输出文本的函数
def clean_ocr_output(text):
    """清理OCR输出，移除特殊标记"""
    if not text:
        return ""
    
    # 移除 <|ref|>...<|/ref|> 标记
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
    # 移除 <|det|>...<|/det|> 标记（包含坐标）
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    # 移除多余的空行
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    # 移除开头和结尾的空白
    text = text.strip()
    
    return text

# PDF转图片函数
def pdf_to_images(pdf_path, output_folder="temp_images", dpi=200):
    """将PDF的每一页转换为图片"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    
    print(f"PDF共有 {len(pdf_document)} 页")
    
    for page_num in range(len(pdf_document)):
        # 获取页面
        page = pdf_document[page_num]
        
        # 设置缩放因子以提高图片质量
        zoom = dpi / 72  # 默认PDF是72 DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # 渲染页面为图片
        pix = page.get_pixmap(matrix=mat)
        
        # 保存图片
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        print(f"已转换第 {page_num + 1} 页")
    
    pdf_document.close()
    return image_paths

# 加载DeepSeek-OCR模型
model_name = 'deepseek-ai/DeepSeek-OCR'

print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 对于Arc显卡，不使用flash_attention_2，使用默认的attention
if device == 'xpu':
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        use_safetensors=True,
        torch_dtype=torch.bfloat16
    )
    model = model.eval()
    model = model.to('xpu')
    # 使用IPEX优化（如果可用）
    if ipex_available:
        model = ipex.optimize(model, dtype=torch.bfloat16)
        print("已启用IPEX优化")
else:
    # CPU模式：使用float32以避免dtype不匹配问题
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        use_safetensors=True,
        torch_dtype=torch.float32
    )
    model = model.eval()
    model = model.to('cpu')
    # 确保所有参数都是float32
    model = model.float()

print("模型加载完成！")

# 处理PDF
pdf_file = 'japanese_test.pdf'
output_md = 'japanese_test_output.md'

print(f"\n开始处理PDF文件: {pdf_file}")

# 1. 将PDF转换为图片
image_paths = pdf_to_images(pdf_file)

# 2. 对每张图片进行OCR
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
all_results = []

import time

for i, image_path in enumerate(image_paths, 1):
    print(f"\n{'='*60}")
    print(f"正在处理第 {i}/{len(image_paths)} 页")
    print('='*60)
    
    start_time = time.time()
    
    # 捕获 stdout 来获取模型输出
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # 执行OCR
        res = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=image_path, 
            output_path='./', 
            base_size=1024,  # 可根据显存调整
            image_size=1024, 
            crop_mode=False,
            save_results=False,  # 不保存中间结果
            test_compress=True
        )
    finally:
        # 恢复 stdout
        sys.stdout = old_stdout
    
    # 获取捕获的输出
    captured_text = captured_output.getvalue()
    
    # 清理输出文本（移除特殊标记）
    cleaned_text = clean_ocr_output(captured_text)
    
    elapsed_time = time.time() - start_time
    print(f"第 {i} 页处理耗时: {elapsed_time:.2f} 秒")
    print(f"识别到文本长度: {len(cleaned_text)} 字符")
    # 收集结果
    all_results.append(f"{cleaned_text}")

# 3. 合并所有结果并保存为Markdown
with open(output_md, 'w', encoding='utf-8') as f:
    for result in all_results:
        f.write(result)

print(f"\n{'='*60}")
print("处理完成！")
print(f"Markdown文档已保存至: {output_md}")
print('='*60)

# 4. 清理临时图片文件（可选）
import shutil
if os.path.exists("temp_images"):
    shutil.rmtree("temp_images")
    print("已清理临时图片文件")

