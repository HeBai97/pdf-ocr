from transformers import AutoModel, AutoTokenizer
import torch
import os
from PIL import Image
import sys
from io import StringIO
import re
import tempfile
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 尝试导入IPEX
try:
    import intel_extension_for_pytorch as ipex
    ipex_available = True
except (ImportError, OSError) as e:
    ipex_available = False
    print(f"IPEX不可用: {type(e).__name__}")

# 尝试导入pytesseract用于快速方向检测
try:
    import pytesseract
    tesseract_available = True
except ImportError:
    tesseract_available = False
    print("pytesseract不可用，将使用传统的4次旋转尝试方法")

# 尝试使用XPU（Intel Arc），如果不可用则使用CPU
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = 'xpu'
    print(f"使用Intel XPU: {torch.xpu.get_device_name(0)}")
    if not ipex_available:
        print("警告: IPEX未安装，性能可能受影响")
else:
    device = 'cpu'
    print("XPU不可用，使用CPU模式")

# 快速检测图片方向（使用Tesseract OSD）
def detect_orientation_fast(image_path):
    """使用Tesseract快速检测图片方向（毫秒级）"""
    if not tesseract_available:
        return None
    
    try:
        image = Image.open(image_path)
        
        # 使用Tesseract的OSD（Orientation and Script Detection）
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        
        rotation = osd.get('rotate', 0)  # 需要旋转的角度
        confidence = osd.get('orientation_conf', 0)  # 置信度
        
        print(f"  → Tesseract检测: 需要旋转 {rotation}度 (置信度: {confidence:.2f})")
        
        # 如果置信度较高，返回检测结果
        if confidence > 1.0:  # Tesseract的置信度阈值
            return rotation
        else:
            print(f"  → 置信度较低，将尝试所有角度")
            return None
            
    except Exception as e:
        print(f"  → Tesseract检测失败: {e}")
        return None

# 从EXIF获取方向信息
def get_exif_orientation(image_path):
    """从图片EXIF数据获取方向信息（最快）"""
    try:
        from PIL.ExifTags import TAGS
        image = Image.open(image_path)
        exif = image._getexif()
        
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'Orientation':
                    # EXIF orientation值映射
                    orientation_map = {1: 0, 3: 180, 6: 270, 8: 90}
                    rotation = orientation_map.get(value, 0)
                    if rotation != 0:
                        print(f"  → EXIF方向: 需要旋转 {rotation}度")
                        return rotation
    except:
        pass
    
    return None

# 智能检测图片方向（组合多种方法）
def smart_detect_orientation(image_path):
    """智能检测图片方向，返回需要旋转的角度"""
    print("🔍 正在快速检测图片方向...")
    
    # 方法1: 检查EXIF信息（瞬时）
    exif_rotation = get_exif_orientation(image_path)
    if exif_rotation is not None:
        return exif_rotation
    
    # 方法2: 使用Tesseract OSD（毫秒级）
    tesseract_rotation = detect_orientation_fast(image_path)
    if tesseract_rotation is not None:
        return tesseract_rotation
    
    # 如果所有快速方法都失败，返回None（使用回退方案）
    print("  → 快速检测未成功，将尝试所有旋转角度")
    return None

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

# 尝试不同角度旋转图片
def try_rotate_image(image_path):
    """尝试不同角度旋转图片，返回可能的旋转版本"""
    img = Image.open(image_path)
    
    # 返回原图和3个旋转版本（90度、180度、270度）
    rotations = {
        '0度': img,
        '90度': img.rotate(-90, expand=True),
        '180度': img.rotate(180, expand=True),
        '270度': img.rotate(-270, expand=True)
    }
    
    return rotations

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

# ==================== 双层PDF生成器 ====================
class DoubleLayerPDFGenerator:
    """使用 DeepSeek-OCR 生成双层PDF（底层图像 + 上层透明可搜索文本）"""
    
    def __init__(self, model, tokenizer, device):
        """
        初始化双层PDF生成器
        
        Args:
            model: DeepSeek-OCR 模型实例
            tokenizer: DeepSeek-OCR tokenizer
            device: 运行设备 ('xpu' 或 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.font_name = "Helvetica"  # 默认字体
        self.dpi = 300
        
        # 尝试注册中文字体
        try:
            # 尝试常见的中文字体路径
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
                "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont("ChineseFont", font_path))
                    self.font_name = "ChineseFont"
                    print(f"✓ 已加载中文字体: {font_path}")
                    break
            else:
                print("⚠ 未找到中文字体，使用默认字体（可能无法显示中文）")
        except Exception as e:
            print(f"⚠ 字体加载失败: {e}，使用默认字体")
    
    def ocr_image_with_boxes(self, image_path):
        """
        对图片执行OCR并返回带位置信息的结果
        
        Returns:
            tuple: (texts, boxes) - 文本列表和对应的边界框坐标
        """
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        
        # 捕获 stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # 执行OCR（启用grounding以获取位置信息）
            res = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path='./',
                base_size=1024,
                image_size=1024,
                crop_mode=False,
                save_results=False,
                test_compress=True
            )
        finally:
            sys.stdout = old_stdout
        
        # 获取输出文本
        captured_text = captured_output.getvalue()
        
        # 解析文本和坐标
        texts, boxes = self.parse_grounding_output(captured_text)
        
        return texts, boxes
    
    def parse_grounding_output(self, ocr_output):
        """
        解析 DeepSeek-OCR 的 grounding 输出，提取文本和坐标
        
        DeepSeek-OCR 的输出格式类似：
        <|det|>x1,y1,x2,y2,x3,y3,x4,y4<|/det|>文本内容
        """
        texts = []
        boxes = []
        
        # 正则表达式匹配 <|det|>坐标<|/det|>文本 模式
        pattern = r'<\|det\|>([\d,]+)<\|/det\|>([^\n<]+)'
        matches = re.finditer(pattern, ocr_output)
        
        for match in matches:
            coords_str = match.group(1)
            text = match.group(2).strip()
            
            if not text:
                continue
            
            try:
                # 解析坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
                coords = [float(x) for x in coords_str.split(',')]
                if len(coords) == 8:
                    # 转换为四个点的格式
                    box = [
                        [coords[0], coords[1]],  # 左上
                        [coords[2], coords[3]],  # 右上
                        [coords[4], coords[5]],  # 右下
                        [coords[6], coords[7]]   # 左下
                    ]
                    texts.append(text)
                    boxes.append(box)
            except (ValueError, IndexError) as e:
                print(f"⚠ 坐标解析失败: {e}")
                continue
        
        print(f"✓ 解析到 {len(texts)} 个文本框")
        return texts, boxes
    
    def generate_double_layer_pdf(self, image_paths, output_pdf_path):
        """
        生成双层PDF
        
        Args:
            image_paths: 图片路径列表（每页一张图）
            output_pdf_path: 输出PDF路径
        """
        if not image_paths:
            print("❌ 没有要处理的图片")
            return False
        
        print(f"\n{'='*60}")
        print("开始生成双层PDF...")
        print(f"{'='*60}")
        
        c = canvas.Canvas(output_pdf_path)
        
        for page_num, img_path in enumerate(image_paths, 1):
            print(f"\n正在处理第 {page_num}/{len(image_paths)} 页...")
            
            # 打开图片
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                print(f"  图片尺寸: {img_width} x {img_height}")
                
                # 设置PDF页面尺寸
                c.setPageSize((img_width, img_height))
                
                # 绘制底层图像
                c.drawImage(img_path, 0, 0, width=img_width, height=img_height)
                print(f"  ✓ 已添加底层图像")
                
                # 执行OCR获取文本和位置
                print(f"  正在执行OCR识别...")
                texts, boxes = self.ocr_image_with_boxes(img_path)
                
                # 绘制透明文本层
                if texts and boxes:
                    print(f"  正在添加 {len(texts)} 个文本框...")
                    for text, box in zip(texts, boxes):
                        self.draw_transparent_text(c, text, box, img_height)
                    print(f"  ✓ 已添加透明文本层")
                else:
                    print(f"  ⚠ 该页没有识别到文本")
                
                # 完成当前页
                c.showPage()
        
        # 保存PDF
        c.save()
        print(f"\n{'='*60}")
        print(f"✓ 双层PDF已生成: {output_pdf_path}")
        print(f"{'='*60}")
        return True
    
    def draw_transparent_text(self, c, text, box, img_height):
        """
        在PDF上绘制透明文本（仅用于搜索，不可见）
        
        Args:
            c: ReportLab canvas对象
            text: 要绘制的文本
            box: 边界框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            img_height: 图片高度（用于Y轴坐标转换）
        """
        if not text or len(box) < 4:
            return
        
        # 提取边界框坐标
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        
        # 计算边界框
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # PDF坐标系统：Y轴从底部向上，需要翻转
        pdf_min_y = img_height - max_y
        pdf_max_y = img_height - min_y
        
        box_width = max_x - min_x
        box_height = pdf_max_y - pdf_min_y
        
        if box_width <= 0 or box_height <= 0:
            return
        
        # 计算字体大小
        font_size = self.calculate_font_size(c, text, box_width, box_height)
        
        # 计算文本位置（垂直居中）
        text_x = min_x
        text_y = pdf_min_y + (box_height - font_size) / 2
        
        # 绘制透明文本（renderMode=3: 不绘制图形，只保留文本索引用于搜索）
        text_obj = c.beginText()
        text_obj.setTextRenderMode(3)  # 不可见但可搜索
        text_obj.setFont(self.font_name, font_size)
        text_obj.setTextOrigin(text_x, text_y)
        
        # 计算字符间距以填充宽度
        text_width = c.stringWidth(text, self.font_name, font_size)
        if len(text) > 1 and text_width < box_width:
            extra_space = (box_width - text_width) / (len(text) - 1)
            text_obj.setCharSpace(extra_space)
        
        text_obj.textLine(text)
        c.drawText(text_obj)
    
    def calculate_font_size(self, c, text, box_width, box_height):
        """
        自适应计算字体大小
        
        Args:
            c: ReportLab canvas对象
            text: 文本内容
            box_width: 边界框宽度
            box_height: 边界框高度
        
        Returns:
            float: 合适的字体大小
        """
        if not text:
            return 8
        
        # 基于高度的字体大小
        font_size_h = box_height * 0.9
        
        # 基于宽度的字体大小
        try:
            text_width = c.stringWidth(text, self.font_name, font_size_h)
            if text_width > 0:
                scale_ratio = box_width / text_width
                font_size_w = font_size_h * scale_ratio
            else:
                font_size_w = font_size_h
        except Exception:
            # 估算：平均每个字符宽度为字体大小的0.55倍
            avg_char_width = 0.55
            font_size_w = box_width / max(len(text), 1) / avg_char_width
        
        # 取两者较小值
        font_size = min(font_size_h, font_size_w)
        
        # 限制在合理范围内
        font_size = max(6, min(48, font_size))
        
        return round(font_size)

# 创建全局PDF生成器实例
pdf_generator = DoubleLayerPDFGenerator(model, tokenizer, device)

# 执行单次OCR识别
def run_single_ocr(image_path, rotation_angle=0, save_name=None):
    """对单张图片执行一次OCR识别"""
    import time
    import shutil
    
    # 打开并旋转图片
    img = Image.open(image_path)
    if rotation_angle == 90:
        img = img.rotate(-90, expand=True)
    elif rotation_angle == 180:
        img = img.rotate(180, expand=True)
    elif rotation_angle == 270:
        img = img.rotate(-270, expand=True)
    
    # 保存临时图片
    temp_path = "temp_ocr_image.png"
    img.save(temp_path)
    
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    start_time = time.time()
    
    # 捕获 stdout 来获取模型输出
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # 执行OCR
        res = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=temp_path, 
            output_path='./', 
            base_size=1024,
            image_size=1024, 
            crop_mode=False,
            save_results=False,
            test_compress=True
        )
    finally:
        # 恢复 stdout
        sys.stdout = old_stdout
    
    # 获取捕获的输出
    captured_text = captured_output.getvalue()
    
    # 清理输出文本
    cleaned_text = clean_ocr_output(captured_text)
    
    elapsed_time = time.time() - start_time
    
    # 保存处理后的图片到images文件夹
    if save_name:
        os.makedirs('images', exist_ok=True)
        save_path = os.path.join('images', f"{save_name}_{rotation_angle}度.png")
        shutil.copy(temp_path, save_path)
        print(f"  → 已保存图片: {save_path}")
    
    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return cleaned_text, elapsed_time

# 处理单张图片的函数（优化版）
def process_image(image_path, output_name):
    """处理单张图片，智能检测方向后执行OCR"""
    print(f"\n{'='*60}")
    print(f"正在处理图片: {image_path}")
    print('='*60)
    
    import time
    
    # 尝试快速检测图片方向
    detected_rotation = smart_detect_orientation(image_path)
    
    if detected_rotation is not None:
        # 快速检测成功，只对检测到的角度执行一次OCR
        print(f"\n✅ 使用检测到的方向: {detected_rotation}度")
        print(f"⚡ 执行单次OCR识别...")
        
        result_text, elapsed = run_single_ocr(image_path, detected_rotation, save_name=output_name)
        
        print(f"✓ 识别完成！耗时: {elapsed:.2f} 秒, 文本长度: {len(result_text)} 字符")
        
        # 转换角度名称
        rotation_map = {0: '0度', 90: '90度', 180: '180度', 270: '270度'}
        rotation_name = rotation_map.get(detected_rotation, '0度')
        
        return result_text, rotation_name
    
    else:
        # 快速检测失败，回退到传统的4次旋转尝试方法
        print(f"\n⚠️  回退到传统方法: 尝试所有旋转角度...")
        
        # 获取不同旋转角度的图片
        rotations = try_rotate_image(image_path)
        
        best_result = None
        best_length = 0
        best_rotation = None
        
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        temp_folder = "temp_rotations"
        os.makedirs(temp_folder, exist_ok=True)
        os.makedirs('images', exist_ok=True)
        
        # 尝试不同的旋转角度
        for rotation_name, rotated_img in rotations.items():
            print(f"\n尝试 {rotation_name} 旋转...")
            
            # 保存临时旋转图片
            temp_path = os.path.join(temp_folder, f"temp_{rotation_name}.png")
            rotated_img.save(temp_path)
            
            # 保存到images文件夹用于二次校验
            save_path = os.path.join('images', f"{output_name}_{rotation_name}.png")
            rotated_img.save(save_path)
            
            start_time = time.time()
            
            # 捕获 stdout 来获取模型输出
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # 执行OCR
                res = model.infer(
                    tokenizer, 
                    prompt=prompt, 
                    image_file=temp_path, 
                    output_path='./', 
                    base_size=1024,
                    image_size=1024, 
                    crop_mode=False,
                    save_results=False,
                    test_compress=True
                )
            finally:
                # 恢复 stdout
                sys.stdout = old_stdout
            
            # 获取捕获的输出
            captured_text = captured_output.getvalue()
            
            # 清理输出文本
            cleaned_text = clean_ocr_output(captured_text)
            
            elapsed_time = time.time() - start_time
            print(f"识别耗时: {elapsed_time:.2f} 秒, 文本长度: {len(cleaned_text)} 字符")
            print(f"  → 已保存图片: {save_path}")
            
            # 选择识别文本最长的结果（通常文本最长说明识别效果最好）
            if len(cleaned_text) > best_length:
                best_length = len(cleaned_text)
                best_result = cleaned_text
                best_rotation = rotation_name
        
        # 清理临时文件
        import shutil
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        
        print(f"\n✓ 最佳旋转角度: {best_rotation}, 文本长度: {best_length} 字符")
        
        return best_result, best_rotation

# ==================== 便捷函数：直接生成双层PDF ====================
def generate_pdf_from_images(image_paths, output_pdf_path='output_searchable.pdf'):
    """
    便捷函数：直接从图片列表生成双层PDF
    
    Args:
        image_paths: 图片路径列表（字符串列表）
        output_pdf_path: 输出PDF路径
    
    Returns:
        bool: 是否成功
    
    使用示例:
        # 方式1: 从文件列表生成
        generate_pdf_from_images([
            '48b9bb8b3bef55124e97520838d68ce1.jpg',
            '8513578d2d071e55893ef0d9f36ba232.jpg'
        ], 'student_answers.pdf')
        
        # 方式2: 从images文件夹中已处理的图片生成
        import glob
        processed_images = sorted(glob.glob('images/*.png'))
        generate_pdf_from_images(processed_images, 'output.pdf')
    """
    print(f"\n{'='*60}")
    print("使用便捷函数生成双层PDF")
    print(f"{'='*60}")
    print(f"输入图片数量: {len(image_paths)}")
    print(f"输出PDF路径: {output_pdf_path}")
    
    # 检查图片是否存在
    valid_paths = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            valid_paths.append(img_path)
            print(f"  ✓ {img_path}")
        else:
            print(f"  ✗ {img_path} (不存在)")
    
    if not valid_paths:
        print("\n❌ 没有有效的图片文件！")
        return False
    
    print(f"\n找到 {len(valid_paths)} 个有效图片文件")
    
    # 生成PDF
    return pdf_generator.generate_double_layer_pdf(valid_paths, output_pdf_path)

# 主程序
if __name__ == "__main__":
    # 要处理的图片列表
    images = [
        ("48b9bb8b3bef55124e97520838d68ce1.jpg", "学生答卷_第1张"),
        ("8513578d2d071e55893ef0d9f36ba232.jpg", "学生答卷_第2张")
    ]
    
    all_results = []
    
    for image_path, name in images:
        if os.path.exists(image_path):
            result, rotation = process_image(image_path, name)
            all_results.append({
                'name': name,
                'path': image_path,
                'rotation': rotation,
                'content': result
            })
        else:
            print(f"警告: 图片文件不存在 - {image_path}")
    
    # 保存结果到Markdown
    output_md = 'student_answers_output.md'
    
    with open(output_md, 'w', encoding='utf-8') as f:
        for idx, result in enumerate(all_results, 1):
            f.write(result['content'])
    
    print(f"\n{'='*60}")
    print("处理完成！")
    print(f"Markdown文档已保存至: {output_md}")
    print('='*60)
    
    # 显示摘要
    print("\n识别摘要:")
    for result in all_results:
        print(f"  - {result['name']}: {result['rotation']} (文本长度: {len(result['content'])} 字符)")
    
    # ==================== 生成双层PDF ====================
    print("\n" + "="*60)
    generate_pdf = input("是否生成双层PDF（可搜索文本）？(y/n): ").strip().lower()
    
    if generate_pdf == 'y':
        # 收集已旋转的图片路径
        image_paths_for_pdf = []
        for result in all_results:
            # 使用已经保存的正确旋转角度的图片
            rotation_name = result['rotation']
            img_path = os.path.join('images', f"{result['name']}_{rotation_name}.png")
            if os.path.exists(img_path):
                image_paths_for_pdf.append(img_path)
            else:
                print(f"⚠ 警告: 找不到图片 {img_path}")
        
        if image_paths_for_pdf:
            output_pdf = 'student_answers_searchable.pdf'
            success = pdf_generator.generate_double_layer_pdf(image_paths_for_pdf, output_pdf)
            if success:
                print(f"\n✓ 双层PDF生成成功！")
                print(f"  - 文件路径: {output_pdf}")
                print(f"  - 功能: 可搜索、可复制文本")
        else:
            print("❌ 没有可用的图片生成PDF")

