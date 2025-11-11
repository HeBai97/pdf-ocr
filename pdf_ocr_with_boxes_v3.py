"""
PDF OCR文本框可视化工具 V11 - 基于 DeepSeek-OCR (支持 Intel XPU)
✅ 最终修复版：直接捕获模型输出
"""

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
import os
import shutil
import argparse
import time
import re
import sys
from io import StringIO
from reportlab.pdfgen import canvas

# 尝试导入IPEX
try:
    import intel_extension_for_pytorch as ipex
    ipex_available = True
except (ImportError, OSError) as e:
    ipex_available = False
    print(f"IPEX不可用: {type(e).__name__}")

# 设备检测
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = 'xpu'
    print(f"✅ 使用 Intel XPU: {torch.xpu.get_device_name(0)}")
    if not ipex_available:
        print("⚠️ 警告: IPEX未安装，性能可能受影响")
else:
    device = 'cpu'
    print("⚠️ XPU不可用，使用CPU模式")

# ==================== 配置 ====================
class Config:
    MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'
    DPI = 300
    OUTPUT_FOLDER = "ocr_boxes_output"
    BOX_WIDTH = 3
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0),
        (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 128),
    ]
    # OCR 配置
    BASE_SIZE = 1024
    IMAGE_SIZE = 1024
    CROP_MODE = False

config = Config()

# ==================== 模型加载 ====================
def load_model():
    print(f"\n🚀 正在加载 DeepSeek-OCR 模型: {config.MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)

    if device == 'xpu':
        model = AutoModel.from_pretrained(
            config.MODEL_NAME, 
            trust_remote_code=True, 
            use_safetensors=True,
            torch_dtype=torch.bfloat16
        )
        model = model.eval()
        model = model.to('xpu')
        if ipex_available:
            model = ipex.optimize(model, dtype=torch.bfloat16)
            print("✅ 已启用IPEX优化")
    else:
        model = AutoModel.from_pretrained(
            config.MODEL_NAME, 
            trust_remote_code=True, 
            use_safetensors=True,
            torch_dtype=torch.float32
        )
        model = model.eval()
        model = model.to('cpu')
        model = model.float()

    print("✅ 模型加载完成！")
    return model, tokenizer

# ==================== PDF 转图像 ====================
def pdf_to_images(pdf_path, dpi=300):
    os.makedirs("temp_pdf_images", exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    print(f"📄 PDF 共有 {len(doc)} 页")

    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_path = f"temp_pdf_images/page_{i+1}.png"
        pix.save(img_path)
        image_paths.append(img_path)
        print(f"   ➜ 已转换第 {i+1} 页 ({pix.width}x{pix.height})")
    doc.close()
    return image_paths

# ==================== OCR 识别（带坐标）====================
def ocr_with_boxes(image_path, model, tokenizer):
    try:
        print(f"🔍 处理图片: {image_path}")
        
        # 使用更简单的提示词
        prompt = "<image>\n<|grounding|>OCR this document with bounding boxes."
        
        start_time = time.time()
        
        # 重定向标准输出以捕获模型输出
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # 调用 infer 方法 - 它会将结果打印到控制台
            result = model.infer(
                tokenizer, 
                prompt=prompt, 
                image_file=image_path, 
                output_path="./temp_ocr_results", 
                base_size=config.BASE_SIZE,
                image_size=config.IMAGE_SIZE, 
                crop_mode=config.CROP_MODE,
                save_results=False,  # 不保存到文件，我们直接捕获输出
                test_compress=True
            )
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout
        
        # 获取捕获的输出
        output_text = captured_output.getvalue()
        elapsed_time = time.time() - start_time
        
        print(f"   ➜ OCR 处理耗时: {elapsed_time:.2f} 秒")
        
        # 从输出中提取OCR结果
        ocr_result = extract_ocr_from_output(output_text)
        
        # 解析返回结果
        boxes = parse_deepseek_ocr_result(ocr_result, image_path)
        return boxes
        
    except Exception as e:
        print(f"[X] OCR 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_ocr_from_output(output_text):
    """
    从控制台输出中提取OCR结果
    """
    try:
        # 查找包含OCR结果的段落
        lines = output_text.split('\n')
        ocr_lines = []
        capture = False
        
        for line in lines:
            # 查找开始标记
            if '<|ref|>' in line:
                capture = True
            if capture:
                ocr_lines.append(line)
            # 查找结束标记（压缩比信息）
            if 'compression ratio:' in line:
                break
        
        return '\n'.join(ocr_lines)
        
    except Exception as e:
        print(f"   [X] 提取OCR输出失败: {e}")
        return output_text

def parse_deepseek_ocr_result(ocr_result, image_path):
    """
    解析 DeepSeek-OCR 的特殊输出格式
    格式: <|ref|>text<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
    """
    boxes = []
    
    try:
        # 打开图像获取尺寸
        pil_image = Image.open(image_path)
        w, h = pil_image.size
        print(f"   ➜ 图像尺寸: {w}x{h}")
        
        if not ocr_result:
            print("   ⚠️ OCR 结果为空")
            return boxes
            
        print(f"   ➜ OCR 原始输出长度: {len(ocr_result)} 字符")
        print(f"   ➜ OCR 输出预览: {ocr_result[:500]}...")
        
        # DeepSeek-OCR 的特殊格式解析
        # 格式: <|ref|>文本内容<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
        pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),(\d+),(\d+),(\d+)\]\]<\|/det\|>'
        matches = re.findall(pattern, ocr_result)
        
        print(f"   ➜ 找到 {len(matches)} 个标准文本框")
        
        for match in matches:
            if len(match) == 5:
                text, x1, y1, x2, y2 = match
                try:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # 坐标处理 - DeepSeek-OCR 返回的坐标可能是相对坐标
                    # 需要根据实际图像尺寸进行缩放
                    scale_x = w / config.BASE_SIZE
                    scale_y = h / config.BASE_SIZE
                    
                    x1_scaled = int(x1 * scale_x)
                    y1_scaled = int(y1 * scale_y)
                    x2_scaled = int(x2 * scale_x)
                    y2_scaled = int(y2 * scale_y)
                    
                    # 确保坐标在图像范围内
                    x1_final = max(0, min(x1_scaled, w))
                    y1_final = max(0, min(y1_scaled, h))
                    x2_final = max(0, min(x2_scaled, w))
                    y2_final = max(0, min(y2_scaled, h))
                    
                    # 创建边界框
                    box = [
                        [x1_final, y1_final],
                        [x2_final, y1_final],
                        [x2_final, y2_final],
                        [x1_final, y2_final]
                    ]
                    
                    boxes.append((text.strip(), box))
                    print(f"      - 文本: '{text[:20]}...' 坐标: [{x1_final},{y1_final},{x2_final},{y2_final}]")
                    
                except ValueError as ve:
                    print(f"      [X] 坐标转换错误: {ve}")
                    continue
        
        # 如果没有找到标准格式，尝试其他可能的格式
        if not boxes:
            print("   ⚠️ 尝试备用格式解析...")
            # 尝试其他可能的坐标格式
            alt_patterns = [
                r'\[(\d+),(\d+),(\d+),(\d+)\]\s*(.*?)(?=\[|$)',
                r'\((\d+),(\d+),(\d+),(\d+)\)\s*(.*?)(?=\(|$)',
            ]
            
            for alt_pattern in alt_patterns:
                alt_matches = re.findall(alt_pattern, ocr_result)
                if alt_matches:
                    print(f"   找到 {len(alt_matches)} 个备用格式文本框")
                    for alt_match in alt_matches:
                        if len(alt_match) == 5:
                            x1, y1, x2, y2, text = alt_match
                            try:
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                                boxes.append((text.strip(), box))
                            except ValueError:
                                continue
                    break
        
        # 如果仍然没有找到坐标，显示原始结果用于调试
        if not boxes:
            print("   ⚠️ 未找到标准文本框格式")
            # 保存完整原始结果到文件用于分析
            debug_file = image_path.replace('.png', '_debug.txt')
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(ocr_result)
            print(f"   💾 完整原始结果已保存到: {debug_file}")
            
            # 创建基于文本位置的模拟框
            lines = ocr_result.split('\n')
            valid_lines = [line.strip() for line in lines if len(line.strip()) > 5]
            if valid_lines:
                print(f"   创建 {len(valid_lines[:20])} 个模拟文本框")
                for i, line in enumerate(valid_lines[:20]):  # 限制数量避免过多
                    box_height = 40
                    y_start = i * box_height + 100
                    box_width = min(len(line) * 12 + 100, w - 200)
                    box = [
                        [100, y_start],
                        [100 + box_width, y_start],
                        [100 + box_width, y_start + box_height],
                        [100, y_start + box_height]
                    ]
                    boxes.append((line, box))
        
    except Exception as e:
        print(f"   [X] 解析OCR结果失败: {e}")
        import traceback
        traceback.print_exc()
    
    return boxes

# ==================== 绘制文本框 ====================
def draw_boxes_on_image(image_path, text_boxes, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    print(f"   ➜ 绘制 {len(text_boxes)} 个文本框...")

    for idx, (text, box) in enumerate(text_boxes):
        color = config.COLORS[idx % len(config.COLORS)]
        coords = [tuple(pt) for pt in box]
        
        # 绘制边界框
        draw.line(coords + [coords[0]], fill=color + (255,), width=config.BOX_WIDTH)
        
        # 在框内添加文本标签（背景）
        if text:
            try:
                # 简化文本显示
                display_text = text[:20] + "..." if len(text) > 20 else text
                # 计算文本位置（框的左上角）
                text_x = coords[0][0]
                text_y = coords[0][1] - 25
                
                # 绘制文本背景
                bbox = draw.textbbox((text_x, text_y), display_text)
                draw.rectangle(bbox, fill=(0, 0, 0, 200))
                # 绘制文本
                draw.text((text_x, text_y), display_text, fill=(255, 255, 255, 255))
            except Exception as e:
                print(f"      [X] 绘制文本失败: {e}")

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    combined.save(output_path, "PNG")
    print(f"   ➜ 已保存: {output_path}")
    return output_path

# ==================== 图像转 PDF ====================
def images_to_pdf(image_paths, output_pdf):
    if not image_paths:
        return False
    c = canvas.Canvas(output_pdf)
    for path in image_paths:
        with Image.open(path) as img:
            w, h = img.size
            c.setPageSize((w, h))
            c.drawImage(path, 0, 0, width=w, height=h)
            c.showPage()
    c.save()
    print(f"✅ 标注 PDF 已生成: {output_pdf}")
    return True

# ==================== 保存文本 ====================
def save_text(all_texts, output_name):
    txt_path = os.path.join(config.OUTPUT_FOLDER, f"{output_name}_ocr_text.md")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_texts))
    print(f"✅ 识别文本已保存: {txt_path}")

# ==================== 主处理函数 ====================
def process_pdf(pdf_path, model, tokenizer, output_name=None):
    if not os.path.exists(pdf_path):
        print(f"[X] 文件不存在: {pdf_path}")
        return

    if output_name is None:
        output_name = os.path.splitext(os.path.basename(pdf_path))[0]

    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📄 开始处理 PDF: {pdf_path}")
    print(f"🧠 使用模型: {config.MODEL_NAME}")
    print(f"💻 设备: {device.upper()}")
    print(f"{'='*60}")

    # 1. PDF -> Images
    print("\n[1/4] 🖼️  将 PDF 转换为图像...")
    image_paths = pdf_to_images(pdf_path, dpi=config.DPI)
    print(f"✅ 完成：共 {len(image_paths)} 页")

    # 2. OCR + 绘图
    print("\n[2/4] 🔍 执行 OCR 并绘制文本框...")
    annotated_images = []
    all_texts = []

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n--- 第 {i}/{len(image_paths)} 页 ---")
        
        boxes = ocr_with_boxes(img_path, model, tokenizer)
        print(f"   ➜ 识别到 {len(boxes)} 个文本块")

        # 提取文本
        texts = [text for text, _ in boxes]
        page_text = "\n".join(texts)
        print(f"   ➜ 文本长度: {len(page_text)} 字符")
        all_texts.append(f"# 第 {i} 页\n\n{page_text}\n\n")

        # 绘图
        out_img = os.path.join(config.OUTPUT_FOLDER, f"{output_name}_page_{i}_annotated.png")
        draw_boxes_on_image(img_path, boxes, out_img)
        annotated_images.append(out_img)

    # 3. 生成 PDF
    print("\n[3/4] 📄 生成标注 PDF...")
    pdf_out = os.path.join(config.OUTPUT_FOLDER, f"{output_name}_annotated.pdf")
    images_to_pdf(annotated_images, pdf_out)

    # 4. 保存文本
    print("\n[4/4] 💾 保存识别文本...")
    save_text(all_texts, output_name)

    # 清理临时文件
    print("\n🧹 清理临时文件...")
    if os.path.exists("temp_pdf_images"):
        shutil.rmtree("temp_pdf_images")
    if os.path.exists("temp_ocr_results"):
        shutil.rmtree("temp_ocr_results")
    print("✅ 临时文件已清理")

    print(f"\n🎉 处理完成！输出目录: {config.OUTPUT_FOLDER}/")
    print(f"   • 标注 PDF: {output_name}_annotated.pdf")
    print(f"   • 识别文本: {output_name}_ocr_text.md")
    print(f"   • 标注图片: {output_name}_page_*.png")
    print(f"{'='*60}\n")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="PDF OCR 文本框可视化工具 (DeepSeek-OCR)")
    parser.add_argument("pdf_file", help="输入 PDF 文件路径")
    parser.add_argument("-o", "--output", help="输出文件名前缀")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 转图像 DPI")
    parser.add_argument("--base_size", type=int, default=1024, help="OCR 基础尺寸")
    args = parser.parse_args()

    config.DPI = args.dpi
    config.BASE_SIZE = args.base_size
    config.IMAGE_SIZE = args.base_size
    
    model, tokenizer = load_model()
    process_pdf(args.pdf_file, model, tokenizer, args.output)

if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        default_pdf = "japanese_test.pdf"
        if os.path.exists(default_pdf):
            print(f"🔍 使用默认文件: {default_pdf}")
            model, tokenizer = load_model()
            process_pdf(default_pdf, model, tokenizer)
        else:
            print("📌 用法: python pdf_ocr_with_boxes.py <pdf文件>")
            print("示例:")
            print("   python pdf_ocr_with_boxes.py input.pdf --dpi 300 --base_size 1024")
    else:
        main()