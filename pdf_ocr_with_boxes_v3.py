"""
PDF MinerU 解析标注工具 V2 - 基于 Intel Arc A40 显卡优化
✅ 使用正确的 MinerU 模型路径和配置
"""

import torch
import os
import argparse
import time
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import numpy as np
import json
import shutil
import gc
import sys
from pathlib import Path
import copy
from typing import List

# ==================== 环境配置 ====================
class Config:
    def __init__(self, cache_dir=None):
        self.device = self.setup_device()
        self.modelscope_cache = self.setup_modelscope_cache(cache_dir)
        self.mineru_model_path = self.setup_mineru_model_path()
    
    def setup_device(self):
        """检测并配置 Intel Arc A40 显卡"""
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = 'xpu'
            print(f"✅ 检测到 Intel GPU: {torch.xpu.get_device_name(0)}")
            print(f"   ➜ 可用设备数量: {torch.xpu.device_count()}")
            torch.xpu.set_device(0)
        else:
            device = 'cpu'
            print("⚠️ 未检测到 Intel XPU，使用 CPU 模式")
        return device
    
    def setup_modelscope_cache(self, cache_dir=None):
        """设置 ModelScope 缓存路径"""
        if cache_dir:
            cache_path = Path(cache_dir)
        else:
            cache_path = Path("D:/modelscope")
        
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ['MODELSCOPE_CACHE'] = str(cache_path)
        print(f"📁 ModelScope 缓存路径: {cache_path}")  
        return cache_path
    
    def setup_mineru_model_path(self):
        """设置 MinerU 模型路径 - 使用您提供的正确路径"""
        # 使用您提供的实际模型路径
        mineru_path = Path("D:/modelscope/hub/models/OpenDataLab/MinerU2___5-2509-1___2B")
        pdf_kit_path = Path("D:/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1___0")
        
        print(f"🔍 MinerU 模型路径: {mineru_path}")
        print(f"🔍 PDF-Extract-Kit 路径: {pdf_kit_path}")
        
        # 检查路径是否存在
        if not mineru_path.exists():
            print(f"⚠️  MinerU 模型路径不存在: {mineru_path}")
        if not pdf_kit_path.exists():
            print(f"⚠️  PDF-Extract-Kit 路径不存在: {pdf_kit_path}")
        
        return {
            "mineru": mineru_path,
            "pdf_kit": pdf_kit_path
        }

# ==================== 直接使用 magic-pdf 的 MinerU 解析器 ====================
class MinerUParser:
    def __init__(self, config):
        self.config = config
        self.setup_environment()
    
    def setup_environment(self):
        """设置 MinerU 环境"""
        print("\n🚀 初始化 MinerU 解析器...")
        
        # 设置环境变量Q
        os.environ['MODELSCOPE_CACHE'] = str(self.config.modelscope_cache)
        
        try:
            # 直接使用 magic-pdf 的解析功能
            import magic_pdf
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.config.enums import SupportedPdfParseMethod
            from magic_pdf.operators.models import InferenceResult
            
            self.PymuDocDataset = PymuDocDataset
            self.doc_analyze = doc_analyze
            self.SupportedPdfParseMethod = SupportedPdfParseMethod
            self.InferenceResult = InferenceResult
            
            print("✅ magic-pdf 模块导入成功")
            
        except ImportError as e:
            print(f"❌ magic-pdf 模块导入失败: {e}")
            print("💡 请确保已安装: pip install magic-pdf")
            raise
    
    def parse_pdf(self, pdf_path, lang="ch"):
        """解析 PDF 文档"""
        print(f"📄 开始解析 PDF: {pdf_path}")
        
        try:
            # 读取 PDF 文件
            with open(pdf_path, 'rb') as f:
                binary = f.read()
            
            # 创建数据集
            ds = self.PymuDocDataset(binary)
            
            # 分类并应用解析
            parse_method = ds.classify()
            print(f"🔍 检测到解析方法: {parse_method}")
            
            if parse_method == self.SupportedPdfParseMethod.OCR or lang not in ['ch', 'en']:
                print("🔍 使用 OCR 模式解析...")
                infer_result = ds.apply(self.doc_analyze, ocr=True, lang=lang)
                pipe_result = infer_result.pipe_ocr_mode(None)
            else:
                print("🔍 使用文本模式解析...")
                infer_result = ds.apply(self.doc_analyze, ocr=False, lang=lang)
                pipe_result = infer_result.pipe_txt_mode(None)
            
            # 获取中间结果
            middle_json = pipe_result.get_middle_json()
            middle_res = json.loads(middle_json)['pdf_info']
            
            print(f"✅ PDF 解析完成，共 {len(middle_res)} 页")
            return middle_res
            
        except Exception as e:
            print(f"❌ PDF 解析失败: {e}")
            import traceback
            traceback.print_exc()
            raise

# ==================== PDF 转图像 ====================
def pdf_to_images(pdf_path, dpi=200):
    """将 PDF 转换为图像"""
    temp_dir = "temp_pdf_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    
    try:
        doc = fitz.open(pdf_path)
        print(f"📄 PDF 共有 {len(doc)} 页")

        for i in range(len(doc)):
            page = doc[i]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_path = os.path.join(temp_dir, f"page_{i+1}.png")
            pix.save(img_path)
            image_paths.append(img_path)
            print(f"   ➜ 已转换第 {i+1} 页 ({pix.width}x{pix.height})")
        
        doc.close()
        
    except Exception as e:
        print(f"❌ PDF 转换失败: {e}")
    
    return image_paths

# ==================== 文本块处理函数 ====================
def _is_latin_start(text: str) -> bool:
    """检查文本是否以拉丁字母开头"""
    try:
        if len(text) == 0:
            return False
        return text[0].isalpha()
    except Exception as e:
        print(f"文本检查错误: {text}")
        return False

def _merge_all_lines_on_block(block: dict, tag: str = 'content') -> str:
    """合并块中的所有行"""
    try:
        lines = block.get('lines', [])
        res = ''
        for line in lines:
            spans = line.get('spans', [])
            for span in spans:
                cur_text = span.get(tag, '')
                if _is_latin_start(cur_text):
                    if len(res) > 0 and res[-1] == '-':
                        res = res[:-1] + cur_text
                    else:
                        res += ' ' + cur_text
                else:
                    res += cur_text
        return res.lstrip()
    except Exception as e:
        print(f"合并行错误: {e}")
        return ""

def extract_text_blocks(middle_res):
    """从 MinerU 解析结果中提取文本块"""
    text_blocks = []
    
    for page_idx, page in enumerate(middle_res):
        page_num = page_idx + 1
        chunks = page.get('para_blocks', [])
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_type = chunk.get('type', 'unknown')
            bbox = chunk.get('bbox', [0, 0, 0, 0])
            
            # 提取所有类型的文本块
            if chunk_type in ['title', 'list', 'index', 'text', 'interline_equation']:
                text = _merge_all_lines_on_block(chunk)
                
                if text.strip():  # 只保留非空文本
                    text_blocks.append({
                        'page_num': page_num,
                        'block_index': chunk_idx,
                        'type': chunk_type,
                        'bbox': bbox,
                        'text': text,
                        'confidence': 0.95
                    })
    
    print(f"📝 提取到 {len(text_blocks)} 个文本块")
    return text_blocks

# ==================== 绘制文本框 ====================
def draw_boxes_on_image(image_path, text_blocks, output_path):
    """在图像上绘制文本框"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            img = img.convert("RGBA")
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # 颜色配置 - 根据块类型使用不同颜色
            TYPE_COLORS = {
                'title': (255, 0, 0),      # 红色 - 标题
                'text': (0, 255, 0),       # 绿色 - 正文
                'list': (0, 0, 255),       # 蓝色 - 列表
                'index': (255, 165, 0),    # 橙色 - 索引
                'interline_equation': (128, 0, 128),  # 紫色 - 公式
                'unknown': (128, 128, 128) # 灰色 - 未知
            }
            
            BOX_WIDTH = 3

            print(f"   ➜ 绘制 {len(text_blocks)} 个文本框...")

            for idx, block in enumerate(text_blocks):
                block_type = block['type']
                color = TYPE_COLORS.get(block_type, (128, 128, 128))
                bbox = block['bbox']
                text = block['text']
                
                # 确保 bbox 坐标有效
                if len(bbox) != 4:
                    continue
                
                # 转换 bbox 坐标 [x0, y0, x1, y1] -> 多边形坐标
                x0, y0, x1, y1 = bbox
                # 确保坐标在图像范围内
                x0 = max(0, min(x0, img_width))
                y0 = max(0, min(y0, img_height))
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                
                coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                
                # 绘制边界框
                draw.line(coords + [coords[0]], fill=color + (255,), width=BOX_WIDTH)
                
                # 在框内添加文本标签
                if text:
                    try:
                        display_text = f"{block_type}: {text[:20]}..." if len(text) > 20 else f"{block_type}: {text}"
                        text_x = max(10, x0)
                        text_y = max(10, y0 - 35)
                        
                        # 绘制文本背景
                        try:
                            font = ImageFont.load_default()
                            text_bbox = draw.textbbox((text_x, text_y), display_text, font=font)
                        except:
                            text_bbox = (text_x, text_y, text_x + len(display_text) * 8, text_y + 20)
                        
                        # 扩展背景框
                        text_bbox = (text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+2)
                        draw.rectangle(text_bbox, fill=(0, 0, 0, 200))
                        
                        # 绘制文本
                        draw.text((text_x, text_y), display_text, fill=(255, 255, 255, 255))
                        
                    except Exception as e:
                        print(f"      [X] 绘制文本失败: {e}")

            combined = Image.alpha_composite(img, overlay).convert("RGB")
            combined.save(output_path, "PNG")
            print(f"   ➜ 已保存标注图片: {output_path}")
            
    except Exception as e:
        print(f"   [X] 绘制图像失败: {e}")
    
    return output_path

# ==================== 安全清理临时文件 ====================
def safe_cleanup_temp_files():
    """安全清理临时文件"""
    temp_dir = "temp_pdf_images"
    
    if os.path.exists(temp_dir):
        print("🧹 清理临时文件...")
        try:
            gc.collect()
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(temp_dir)
                    print("✅ 临时文件已清理")
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"   ⚠️ 文件占用，等待重试... ({attempt + 1}/{max_retries})")
                        time.sleep(1)
                    else:
                        print("   ⚠️ 无法清理部分临时文件")
        except Exception as e:
            print(f"   ⚠️ 清理临时文件时出错: {e}")

# ==================== 主处理函数 ====================
def process_pdf_with_mineru(pdf_path, output_name=None, cache_dir=None, lang="ch"):
    """使用 MinerU 处理 PDF 并生成标注图片"""
    if not os.path.exists(pdf_path):
        print(f"[X] 文件不存在: {pdf_path}")
        return

    if output_name is None:
        output_name = os.path.splitext(os.path.basename(pdf_path))[0]

    output_folder = "ocr_boxes_output"
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📄 开始处理 PDF: {pdf_path}")
    print(f"🧠 使用框架: MinerU (magic-pdf)")
    print(f"💻 设备: {device.upper()}")
    print(f"🌐 语言: {lang}")
    if cache_dir:
        print(f"💾 缓存目录: {cache_dir}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # 1. 初始化配置和解析器
        print("\n[1/4] 🚀 初始化 MinerU 解析器...")
        config = Config(cache_dir)
        parser = MinerUParser(config)
        
        # 2. 使用 MinerU 解析 PDF
        print("\n[2/4] 🔍 使用 MinerU 解析 PDF...")
        middle_res = parser.parse_pdf(pdf_path, lang=lang)
        
        # 保存解析结果
        middle_res_file = os.path.join(output_folder, f"{output_name}_mineru_result.json")
        with open(middle_res_file, 'w', encoding='utf-8') as f:
            json.dump(middle_res, f, ensure_ascii=False, indent=2)
        print(f"✅ MinerU 解析结果已保存: {middle_res_file}")
        
        # 3. 提取文本块
        print("\n[3/4] 📝 提取文本块信息...")
        text_blocks = extract_text_blocks(middle_res)
        
        # 按页面分组文本块
        page_blocks = {}
        for block in text_blocks:
            page_num = block['page_num']
            if page_num not in page_blocks:
                page_blocks[page_num] = []
            page_blocks[page_num].append(block)
        
        # 4. PDF -> Images + 绘制标注
        print("\n[4/4] 🖼️  生成标注图像...")
        image_paths = pdf_to_images(pdf_path, dpi=200)
        annotated_images = []
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n--- 第 {i}/{len(image_paths)} 页 ---")
            
            if i in page_blocks:
                page_blocks_i = page_blocks[i]
                print(f"   ➜ 本页有 {len(page_blocks_i)} 个文本块")
                
                # 绘图
                out_img = os.path.join(output_folder, f"{output_name}_page_{i}_annotated.png")
                draw_boxes_on_image(img_path, page_blocks_i, out_img)
                annotated_images.append(out_img)
            else:
                print("   ➜ 本页没有检测到文本块")
        
        # 5. 保存处理摘要
        elapsed_time = time.time() - start_time
        
        summary = {
            "pdf_file": pdf_path,
            "total_pages": len(image_paths),
            "parsed_blocks": len(text_blocks),
            "output_folder": output_folder,
            "annotated_images": [os.path.basename(p) for p in annotated_images],
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": round(elapsed_time, 2),
            "device_used": device,
            "model_used": "MinerU (magic-pdf)",
            "language": lang,
            "block_types": {
                block_type: len([b for b in text_blocks if b['type'] == block_type])
                for block_type in set(b['type'] for b in text_blocks)
            }
        }
        
        summary_file = os.path.join(output_folder, f"{output_name}_processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 处理摘要已保存: {summary_file}")
        
        # 保存文本块详细信息
        blocks_file = os.path.join(output_folder, f"{output_name}_text_blocks.json")
        with open(blocks_file, 'w', encoding='utf-8') as f:
            json.dump(text_blocks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 文本块详情已保存: {blocks_file}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 安全清理临时文件
    safe_cleanup_temp_files()

    print(f"\n🎉 MinerU PDF 解析标注完成！")
    print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
    print(f"📁 输出目录: {output_folder}/")
    print(f"🖼️  标注图片: {len(annotated_images)} 张")
    print(f"📝 解析块数: {len(text_blocks)} 个")
    print(f"{'='*60}\n")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="PDF MinerU 解析标注工具 V2")
    parser.add_argument("pdf_file", help="输入 PDF 文件路径")
    parser.add_argument("-o", "--output", help="输出文件名前缀")
    parser.add_argument("--cache-dir", help="ModelScope 缓存目录")
    parser.add_argument("--lang", default="ch", choices=['ch', 'en'], help="文档语言")
    parser.add_argument("--dpi", type=int, default=200, help="PDF 转图像 DPI")
    
    args = parser.parse_args()

    process_pdf_with_mineru(
        args.pdf_file, 
        args.output, 
        args.cache_dir,
        args.lang
    )

if __name__ == "__main__":
    # 全局设备变量
    device = "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu"
    
    if len(sys.argv) == 1:
        print("📌 用法: python pdf_mineru_annotation_v2.py <pdf文件> [选项]")
        print("\n示例:")
        print("  # 基本使用")
        print("  python pdf_mineru_annotation_v2.py document.pdf")
        print("  # 指定输出名称和语言")
        print("  python pdf_mineru_annotation_v2.py document.pdf -o result --lang en")
        print("  # 指定缓存目录")
        print("  python pdf_mineru_annotation_v2.py document.pdf --cache-dir D:/modelscope")
    else:
        main()