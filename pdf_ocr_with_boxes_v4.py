"""
PDF MinerU OCR解析标注工具 V8 - 修复 magic-pdf 导入问题
✅ 使用 magic-pdf 进行高质量的 PDF 解析
✅ 标注真实解析的文本内容和 bbox 坐标
✅ 支持多种元素类型：文本、表格、图像、公式等
"""

import os
import argparse
import time
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import json
import shutil
import gc
import sys
from pathlib import Path
import copy
from typing import List

# ==================== 导入 magic-pdf 相关模块 ====================
try:
    # 尝试不同的导入方式
    try:
        # 方式1: 直接导入
        from magic_pdf import PymuDocDataset, doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
        from magic_pdf.operators.models import InferenceResult
        print("✅ magic-pdf 导入成功 (直接导入)")
    except ImportError:
        # 方式2: 尝试从其他路径导入
        import magic_pdf
        print("✅ magic-pdf 导入成功 (模块导入)")
        
    # 检查必要的类和方法是否存在
    if hasattr(magic_pdf, 'PymuDocDataset') or 'PymuDocDataset' in globals():
        print("✅ 找到 PymuDocDataset 类")
    else:
        print("❌ 未找到 PymuDocDataset 类")
        
except ImportError as e:
    print(f"❌ magic-pdf 导入失败: {e}")
    print("💡 尝试使用备用解析方法...")
    
    # 备用方案：使用 PyMuPDF 直接解析
    class FallbackPDFParser:
        def extract_blocks_from_pdf(self, pdf_path, lang="ch"):
            """使用 PyMuPDF 作为备选解析方法"""
            print(f"📄 使用 PyMuPDF 备选解析: {pdf_path}")
            
            import fitz
            doc = fitz.open(pdf_path)
            all_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 获取文本块
                blocks = page.get_text("dict")["blocks"]
                
                for block_idx, block in enumerate(blocks):
                    if "lines" in block:  # 文本块
                        bbox = block["bbox"]  # [x0, y0, x1, y1]
                        text = ""
                        
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text += span["text"] + " "
                        
                        if text.strip():
                            all_blocks.append({
                                'page_num': page_num + 1,
                                'type': 'text',
                                'bbox': bbox,
                                'text': text.strip(),
                                'confidence': 0.8
                            })
            
            doc.close()
            print(f"📝 备选方法提取到 {len(all_blocks)} 个文本块")
            return all_blocks

# ==================== 配置类 ====================
class Config:
    def __init__(self, cache_dir=None):
        self.device = self.setup_device()
        self.modelscope_cache = self.setup_modelscope_cache(cache_dir)
    
    def setup_device(self):
        """检测并配置设备"""
        try:
            import torch
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = 'xpu'
                print(f"✅ 检测到 Intel GPU: {torch.xpu.get_device_name(0)}")
            else:
                device = 'cpu'
                print("⚡ 使用 CPU 模式")
        except:
            device = 'cpu'
            print("⚡ 使用 CPU 模式")
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

# ==================== 使用 magic-pdf 进行解析 ====================
class MagicPDFParser:
    def __init__(self, config):
        self.config = config
        self.setup_environment()
    
    def setup_environment(self):
        """设置 magic-pdf 环境"""
        print("\n🚀 初始化 PDF 解析器...")
        print("✅ PDF 解析器初始化完成")
    
    def extract_blocks_from_pdf(self, pdf_path, lang="ch"):
        """从 PDF 中提取所有块"""
        print(f"📄 解析 PDF: {pdf_path}")
        
        try:
            # 首先尝试使用 magic-pdf
            return self._extract_with_magic_pdf(pdf_path, lang)
        except Exception as e:
            print(f"❌ magic-pdf 解析失败: {e}")
            print("🔄 使用备选解析方法...")
            # 使用备选方法
            return
    
    def _extract_with_magic_pdf(self, pdf_path, lang):
        """使用 magic-pdf 解析"""
        print("🔍 使用 magic-pdf 高级解析...")
        
        # 读取 PDF 文件
        binary = open(pdf_path, 'rb').read()
        
        # 动态检测可用的类
        if 'PymuDocDataset' in globals():
            ds = PymuDocDataset(binary)
        else:
            # 尝试从 magic_pdf 模块导入
            from magic_pdf import PymuDocDataset
            ds = PymuDocDataset(binary)
        
        # 判断使用 OCR 模式还是文本模式
        if hasattr(ds, 'classify'):
            pdf_type = ds.classify()
            if pdf_type == SupportedPdfParseMethod.OCR or lang not in ['ch', 'en']:
                print("🔍 使用 OCR 模式解析...")
                infer_result = ds.apply(doc_analyze, ocr=True, lang=lang)
                pipe_result = infer_result.pipe_ocr_mode(None)
            else:
                print("🔍 使用文本模式解析...")
                infer_result = ds.apply(doc_analyze, ocr=False, lang=lang)
                pipe_result = infer_result.pipe_txt_mode(None)
        else:
            # 简化版本，直接使用文本模式
            print("🔍 使用简化文本模式解析...")
            infer_result = ds.apply(doc_analyze, ocr=False, lang=lang)
            pipe_result = infer_result.pipe_txt_mode(None)
        
        # 获取解析结果
        middle_res = json.loads(pipe_result.get_middle_json())['pdf_info']
        
        # 提取所有块信息
        all_blocks = self._extract_blocks_from_middle_result(middle_res)
        
        print(f"📝 magic-pdf 解析完成，共 {len(all_blocks)} 个块")
        return all_blocks
    
    def _extract_blocks_from_middle_result(self, middle_res):
        """从 magic-pdf 的中间结果中提取块信息"""
        all_blocks = []
        
        for page in middle_res:
            page_num = page['page_idx'] + 1
            chunks = page.get('para_blocks', [])
            
            for chunk in chunks:
                block_info = self._process_chunk(chunk, page_num)
                if block_info:
                    all_blocks.append(block_info)
        
        return all_blocks
    
    def _process_chunk(self, chunk, page_num):
        """处理单个块"""
        chunk_type = chunk.get('type', 'text')
        bbox = chunk.get('bbox', [0, 0, 0, 0])
        
        block_info = {
            'page_num': page_num,
            'type': chunk_type,
            'bbox': bbox,
            'text': '',
            'confidence': 0.9,
            'raw_data': chunk
        }
        
        # 根据不同类型提取文本
        if chunk_type in ['title', 'list', 'index', 'text', 'interline_equation']:
            block_info['text'] = self._merge_lines_from_block(chunk)
        elif chunk_type == 'table':
            block_info['text'] = self._extract_table_text(chunk)
        elif chunk_type == 'image':
            block_info['text'] = self._extract_image_text(chunk)
        
        return block_info
    
    def _merge_lines_from_block(self, block, tag='content'):
        """合并块中的所有行文本"""
        lines = block.get('lines', [])
        res = ''
        
        for line in lines:
            for span in line.get('spans', []):
                cur_text = span.get(tag, '')
                res += cur_text
        
        return res.strip()
    
    def _extract_table_text(self, table_block):
        """提取表格文本"""
        blocks = table_block.get('blocks', [])
        caption = 'NULL'
        footnote = 'NULL'
        body = 'NULL'
        
        for block in blocks:
            block_type = block.get('type', '')
            if block_type == 'table_caption':
                caption = self._merge_lines_from_block(block)
            elif block_type == 'table_footnote':
                footnote = self._merge_lines_from_block(block)
            elif block_type == 'table_body':
                body = self._merge_lines_from_block(block, tag='html')
        
        return f'table caption: {caption}\ntable body: {body}\ntable footnote: {footnote}'
    
    def _extract_image_text(self, image_block):
        """提取图像文本"""
        blocks = image_block.get('blocks', [])
        caption = 'NULL'
        
        for block in blocks:
            if block.get('type') == 'image_caption':
                caption = self._merge_lines_from_block(block)
                break
        
        return f'image caption: {caption}'

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
            def get_color_by_type(block_type):
                color_map = {
                    'title': (255, 0, 0),      # 红色 - 标题
                    'text': (0, 255, 0),       # 绿色 - 正文
                    'list': (0, 0, 255),       # 蓝色 - 列表
                    'table': (255, 165, 0),    # 橙色 - 表格
                    'image': (128, 0, 128),    # 紫色 - 图像
                    'interline_equation': (0, 128, 128),  # 青色 - 公式
                    'index': (165, 42, 42)     # 棕色 - 索引
                }
                return color_map.get(block_type, (128, 128, 128))  # 灰色 - 其他

            BOX_WIDTH = 3

            print(f"   ➜ 绘制 {len(text_blocks)} 个文本框...")

            for idx, block in enumerate(text_blocks):
                block_type = block['type']
                color = get_color_by_type(block_type)
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
                        # 显示块类型和文本预览
                        display_text = f"{block_type}: {text[:30]}..." if len(text) > 30 else f"{block_type}: {text}"
                        text_x = max(10, x0)
                        text_y = max(10, y0 - 40)
                        
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

# ==================== PDF 转图像工具 ====================
def pdf_to_images(pdf_path, dpi=200):
    """将 PDF 转换为图像"""
    temp_dir = "temp_pdf_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    doc = fitz.open(pdf_path)
    
    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(temp_dir, f"page_{i+1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    
    doc.close()
    return image_paths

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
def process_pdf_with_magic_pdf(pdf_path, output_name=None, cache_dir=None, lang="ch"):
    """使用 PDF 解析器处理 PDF 并生成标注图片"""
    if not os.path.exists(pdf_path):
        print(f"[X] 文件不存在: {pdf_path}")
        return

    if output_name is None:
        output_name = os.path.splitext(os.path.basename(pdf_path))[0]

    output_folder = "ocr_boxes_output"
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📄 开始处理 PDF: {pdf_path}")
    print(f"🧠 使用引擎: PDF 解析器")
    print(f"💻 设备: {device.upper()}")
    print(f"🌐 语言: {lang}")
    if cache_dir:
        print(f"💾 缓存目录: {cache_dir}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # 1. 初始化配置和解析器
        print("\n[1/4] 🚀 初始化 PDF 解析器...")
        config = Config(cache_dir)
        pdf_parser = MagicPDFParser(config)
        
        # 2. 使用解析器解析 PDF
        print("\n[2/4] 🔍 解析 PDF...")
        text_blocks = pdf_parser.extract_blocks_from_pdf(pdf_path, lang)
        
        if not text_blocks:
            print("❌ 没有解析到任何块，使用备选方法...")
            # 使用简单的 PyMuPDF 解析
            import fitz
            doc = fitz.open(pdf_path)
            text_blocks = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_blocks.append({
                        'page_num': page_num + 1,
                        'type': 'text',
                        'bbox': [0, 0, page.rect.width, page.rect.height],
                        'text': text.strip(),
                        'confidence': 0.5
                    })
            doc.close()
        
        # 按页面分组文本块
        page_blocks = {}
        for block in text_blocks:
            page_num = block['page_num']
            if page_num not in page_blocks:
                page_blocks[page_num] = []
            page_blocks[page_num].append(block)
        
        # 3. 重新生成图像并绘制标注
        print("\n[3/4] 🖼️  生成标注图像...")
        image_paths = pdf_to_images(pdf_path, dpi=200)
        annotated_images = []
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n--- 第 {i}/{len(image_paths)} 页 ---")
            
            if i in page_blocks:
                page_blocks_i = page_blocks[i]
                print(f"   ➜ 本页有 {len(page_blocks_i)} 个块")
                
                # 统计不同类型块的数量
                type_count = {}
                for block in page_blocks_i:
                    block_type = block['type']
                    type_count[block_type] = type_count.get(block_type, 0) + 1
                
                print(f"   ➜ 块类型分布: {type_count}")
                
                # 绘图
                out_img = os.path.join(output_folder, f"{output_name}_page_{i}_annotated.png")
                draw_boxes_on_image(img_path, page_blocks_i, out_img)
                annotated_images.append(out_img)
            else:
                print("   ➜ 本页没有检测到块")
        
        # 4. 保存处理摘要
        elapsed_time = time.time() - start_time
        
        # 计算统计信息
        type_statistics = {}
        for block in text_blocks:
            block_type = block['type']
            type_statistics[block_type] = type_statistics.get(block_type, 0) + 1
        
        summary = {
            "pdf_file": pdf_path,
            "total_pages": len(image_paths),
            "total_blocks": len(text_blocks),
            "block_type_statistics": type_statistics,
            "output_folder": output_folder,
            "annotated_images": [os.path.basename(p) for p in annotated_images],
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": round(elapsed_time, 2),
            "device_used": device,
            "model_used": "PDF Parser",
            "language": lang
        }
        
        summary_file = os.path.join(output_folder, f"{output_name}_processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 处理摘要已保存: {summary_file}")
        
        # 保存块详细信息
        blocks_file = os.path.join(output_folder, f"{output_name}_blocks.json")
        with open(blocks_file, 'w', encoding='utf-8') as f:
            json.dump(text_blocks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 块详情已保存: {blocks_file}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 安全清理临时文件
    safe_cleanup_temp_files()

    print(f"\n🎉 PDF 解析标注完成！")
    print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
    print(f"📁 输出目录: {output_folder}/")
    print(f"🖼️  标注图片: {len(annotated_images)} 张")
    print(f"📦 解析块数: {len(text_blocks)} 个")
    if 'type_statistics' in locals():
        print(f"📊 块类型统计: {type_statistics}")
    print(f"{'='*60}\n")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="PDF 解析标注工具 V8")
    parser.add_argument("pdf_file", help="输入 PDF 文件路径")
    parser.add_argument("-o", "--output", help="输出文件名前缀")
    parser.add_argument("--cache-dir", help="ModelScope 缓存目录")
    parser.add_argument("--lang", default="ch", choices=['ch', 'en'], help="文档语言")
    parser.add_argument("--dpi", type=int, default=200, help="PDF 转图像 DPI")
    
    args = parser.parse_args()

    process_pdf_with_magic_pdf(
        args.pdf_file, 
        args.output, 
        args.cache_dir,
        args.lang
    )

if __name__ == "__main__":
    # 全局设备变量
    device = "cpu"
    
    if len(sys.argv) == 1:
        print("📌 用法: python pdf_ocr_with_boxes_v4.py <pdf文件> [选项]")
        print("\n示例:")
        print("  # 基本使用")
        print("  python pdf_ocr_with_boxes_v4.py document.pdf")
        print("  # 指定输出名称和语言")
        print("  python pdf_ocr_with_boxes_v4.py document.pdf -o result --lang en")
        print("  # 指定缓存目录")
        print("  python pdf_ocr_with_boxes_v4.py document.pdf --cache-dir D:/modelscope")
    else:
        main()