"""
PDF OCR文本框可视化工具 V2

改进版：使用PaddleOCR获取文本框坐标（更可靠）

使用方法:
    python pdf_ocr_with_boxes_v2.py <pdf文件路径>
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import sys
from io import StringIO
import re
from reportlab.pdfgen import canvas
from paddleocr import PaddleOCR
import numpy as np

# 尝试导入IPEX
try:
    import intel_extension_for_pytorch as ipex
    ipex_available = True
except (ImportError, OSError) as e:
    ipex_available = False

# 设备检测
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = 'xpu'
    print(f"使用Intel XPU: {torch.xpu.get_device_name(0)}")
else:
    device = 'cpu'
    print("使用CPU模式")

# ==================== 配置 ====================
class Config:
    """配置参数"""
    
    # === 颜色配置 ===
    COLORS = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 165, 0),    # 橙色
        (255, 0, 255),    # 品红
        (0, 255, 255),    # 青色
        (255, 255, 0),    # 黄色
        (128, 0, 128),    # 紫色
    ]
    # 说明：文本框会循环使用这些颜色进行标注
    
    # === 文本框边框配置 ===
    BOX_WIDTH = 1
    # 说明：文本框边框线条的宽度（像素）
    # 建议值：1-5，值越大边框越粗
    # 影响：过粗会遮挡文字，过细不明显
    
    BOX_OPACITY = 180
    # 说明：文本框边框的不透明度（0-255）
    # 建议值：150-220，0完全透明，255完全不透明
    # 影响：透明度太高会遮挡原文，太低看不清边框
    
    # === 文本标签配置 ===
    SHOW_TEXT_LABEL = False
    # 说明：是否在文本框上方显示识别出的文字标签
    # True=显示，False=不显示（只显示彩色框）
    
    LABEL_FONT_SIZE = 12
    # 说明：文本标签的字体大小（像素）
    # 建议值：12-24，根据PDF分辨率调整
    # 影响：字体太大标签会重叠，太小看不清
    
    LABEL_OFFSET_Y = 8
    # 说明：文本标签距离文本框顶部的垂直距离（像素）
    # 建议值：5-15，值越大标签离框越远
    # 影响：太小标签会覆盖原文，太大可能超出图片边界
    
    LABEL_OFFSET_X = 2
    # 说明：文本标签距离文本框左边的水平偏移（像素）
    # 建议值：0-10，用于微调标签左右位置
    # 影响：调整标签与文本框的对齐关系
    
    LABEL_PADDING = 2
    # 说明：文本标签内部的内边距（像素）
    # 建议值：1-5，文字与标签背景边缘的距离
    # 影响：值越大标签背景框越大
    
    LABEL_BG_OPACITY = 220
    # 说明：文本标签背景的不透明度（0-255）
    # 建议值：180-240，0完全透明，255完全不透明
    # 影响：背景透明度，太高遮挡原文，太低看不清文字
    
    # === 图像处理配置 ===
    DPI = 180
    # 说明：PDF转图片时的分辨率（DPI = Dots Per Inch）
    # 建议值：150-300，值越高图片越清晰但文件越大
    # 影响：150适合快速预览，300适合高质量输出
    # 重要：日语识别建议使用 200-300 DPI，太低会影响小字识别准确率
    
    OUTPUT_FOLDER = "ocr_boxes_output"
    # 说明：输出文件的保存文件夹路径
    
    # === OCR识别配置 ===
    LANG = 'japan'
    # 说明：OCR识别语言
    # 可选值：'japan'=日语, 'ch'=中文, 'en'=英文
    # 影响：直接影响识别准确率，务必选择正确的语言
    
    OCR_MIN_CONFIDENCE = 0.5
    # 说明：OCR识别的最低置信度阈值（0.0-1.0）
    # 建议值：0.3-0.7，值越高过滤越严格
    # 影响：提高此值会过滤掉低质量识别结果，但可能漏掉一些文字
    # 如果发现很多错误识别，可以提高到 0.6-0.7
    
    SHOW_CONFIDENCE = False
    # 说明：是否在标签上显示识别置信度
    # True=显示置信度分数（如：テスト[0.95]），False=只显示文字
    
    # === OCR检测精度调整参数（重要！影响框的准确性）===
    OCR_DET_DB_THRESH = 0.3
    # 说明：文本检测的二值化阈值（0.0-1.0）
    # 建议值：0.2-0.5，值越小检测越敏感，能检测到更模糊的文字
    # 影响：太小会误检噪点，太大会漏检文字
    # 默认0.3适合大多数情况
    
    OCR_DET_DB_BOX_THRESH = 0.5
    # 说明：文本框的置信度阈值（0.0-1.0）
    # 建议值：0.4-0.7，值越高过滤越严格
    # 影响：太小会保留低质量框，太大会丢失文字
    # 如果发现标注框位置不准确，可以提高到0.6
    
    OCR_DET_DB_UNCLIP_RATIO = 1.2
    # 说明：文本框扩张比例（通常1.2-2.0）
    # [!] 这个参数很关键！直接影响标注框大小
    # 建议值：
    #   - 1.2-1.4：框会紧贴文字（适合密集文本）
    #   - 1.5-1.6：适中（默认推荐）
    #   - 1.7-2.0：框会包含较多边距（适合稀疏文本）
    # 影响：如果你觉得"框比文字大很多"，降低此值到1.3-1.4
    #       如果觉得"框切到文字"，增加此值到1.7-1.8
    
    OCR_USE_DILATION = False
    # 说明：是否使用膨胀算法扩展文本区域
    # True=使用（框会稍大一些），False=不使用（框更紧凑）
    # 影响：关闭可以让框更贴合文字，但可能会切到边缘
    # 用于调试，查看哪些文字识别质量低
    
    # === 坐标偏移调整（修复框位置偏移问题）===
    BOX_OFFSET_X = 0
    # 说明：标注框X轴（水平）偏移量（像素）
    # 建议值：-50 到 +50
    # 正值=向右移动，负值=向左移动
    # 如果框往左偏，增加此值；如果框往右偏，减少此值
    
    BOX_OFFSET_Y = 0
    # 说明：标注框Y轴（垂直）偏移量（像素）
    # 建议值：-50 到 +50
    # 正值=向下移动，负值=向上移动
    # 如果框往上偏，增加此值；如果框往下偏，减少此值

config = Config()

# ==================== OCR工具 ====================

def pdf_to_images(pdf_path, output_folder="temp_pdf_images", dpi=150):
    """将PDF转换为图片"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    
    print(f"PDF共有 {len(pdf_document)} 页")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        print(f"  => 已转换第 {page_num + 1} 页")
    
    pdf_document.close()
    return image_paths

def ocr_with_paddleocr(image_path, paddle_ocr, debug=False):
    """
    使用PaddleOCR获取文本和边界框
    
    Returns:
        list: [(text, box), ...] 其中box = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # 调试：显示输入图片信息
    if debug:
        print(f"  [调试] 输入图片尺寸: {img.size} (宽x高)")
        print(f"  [调试] 图片模式: {img.mode}")
        print(f"  [调试] 数组形状: {img_array.shape}")
    
    # 执行OCR - 兼容新旧版本API
    result = None
    api_used = "unknown"
    
    try:
        # 尝试新版本API (使用predict方法)
        result = paddle_ocr.predict(img_array)
        api_used = "predict()"
    except AttributeError:
        # 如果predict方法不存在，使用旧版本的ocr方法
        try:
            result = paddle_ocr.ocr(img_array, cls=False)
            api_used = "ocr(cls=False)"
        except TypeError:
            # 如果cls参数不支持，不传递cls参数
            result = paddle_ocr.ocr(img_array)
            api_used = "ocr()"
    
    # 调试信息
    if debug:
        print(f"  [调试] 使用API: {api_used}")
        print(f"  [调试] result类型: {type(result)}")
        if result:
            print(f"  [调试] result长度: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            if isinstance(result, list) and len(result) > 0:
                print(f"  [调试] result[0]类型: {type(result[0])}")
                # 检查是否是OCRResult对象
                if hasattr(result[0], 'rec_texts'):
                    print(f"  [调试] OCRResult对象 - rec_texts数量: {len(result[0].rec_texts) if hasattr(result[0].rec_texts, '__len__') else 'N/A'}")
                    if hasattr(result[0].rec_texts, '__len__') and len(result[0].rec_texts) > 0:
                        print(f"  [调试] 第1个文本: '{result[0].rec_texts[0]}'")
                # 检查是否是字典
                elif isinstance(result[0], dict):
                    print(f"  [调试] 字典格式 - keys: {list(result[0].keys())}")
                    if 'rec_texts' in result[0] and result[0]['rec_texts']:
                        print(f"  [调试] 第1个文本: '{result[0]['rec_texts'][0]}'")
                # 检查是否是列表
                elif isinstance(result[0], list) and len(result[0]) > 0:
                    print(f"  [调试] result[0][0]: {result[0][0]}")
    
    text_boxes = []
    
    if not result:
        return text_boxes
    
    # 解析结果 - 兼容新旧版本的PaddleOCR
    try:
        if isinstance(result, list) and len(result) > 0:
            ocr_result = result[0]
            
            # 新版本：字典格式（PaddleX格式，包含rec_texts, rec_scores, rec_polys键）
            if isinstance(ocr_result, dict) and 'rec_texts' in ocr_result:
                if debug:
                    print(f"  [调试] 使用新版本字典格式（PaddleX）")
                
                rec_texts = ocr_result.get('rec_texts', [])
                rec_scores = ocr_result.get('rec_scores', [1.0] * len(rec_texts))
                rec_polys = ocr_result.get('rec_polys', [])
                
                for idx, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                    # 过滤低置信度和空文本
                    if text and len(text.strip()) > 0 and score > config.OCR_MIN_CONFIDENCE:
                        # 转换numpy数组为列表格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        if hasattr(poly, 'tolist'):
                            box = poly.tolist()
                        else:
                            box = poly
                        
                        # 调试第一个结果
                        if debug and len(text_boxes) == 0:
                            print(f"  [调试] 第1个文本: '{text}', 置信度: {score:.3f}")
                            print(f"  [调试] 第1个框坐标: {box}")
                            if len(box) >= 4:
                                xs = [pt[0] for pt in box]
                                ys = [pt[1] for pt in box]
                                print(f"  [调试] X范围: [{min(xs):.1f}, {max(xs):.1f}], Y范围: [{min(ys):.1f}, {max(ys):.1f}]")
                                print(f"  [调试] 框宽度: {max(xs)-min(xs):.1f}, 框高度: {max(ys)-min(ys):.1f}")
                        
                        text_boxes.append((text, box))
            
            # 中版本：OCRResult对象（有rec_texts, rec_scores, rec_polys属性）
            elif hasattr(ocr_result, 'rec_texts') and hasattr(ocr_result, 'rec_polys'):
                if debug:
                    print(f"  [调试] 使用OCRResult对象格式")
                
                rec_texts = ocr_result.rec_texts
                rec_scores = ocr_result.rec_scores if hasattr(ocr_result, 'rec_scores') else [1.0] * len(rec_texts)
                rec_polys = ocr_result.rec_polys
                
                for idx, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                    # 过滤低置信度和空文本
                    if text and len(text.strip()) > 0 and score > config.OCR_MIN_CONFIDENCE:
                        # 转换numpy数组为列表格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        if hasattr(poly, 'tolist'):
                            box = poly.tolist()
                        else:
                            box = poly
                        
                        # 调试第一个结果
                        if debug and len(text_boxes) == 0:
                            print(f"  [调试] 第1个文本: '{text}', 置信度: {score:.3f}")
                            print(f"  [调试] 第1个框坐标: {box}")
                            if len(box) >= 4:
                                xs = [pt[0] for pt in box]
                                ys = [pt[1] for pt in box]
                                print(f"  [调试] X范围: [{min(xs):.1f}, {max(xs):.1f}], Y范围: [{min(ys):.1f}, {max(ys):.1f}]")
                                print(f"  [调试] 框宽度: {max(xs)-min(xs):.1f}, 框高度: {max(ys)-min(ys):.1f}")
                        
                        text_boxes.append((text, box))
            
            # 旧版本：列表格式 [[[box], (text, score)], ...]
            elif isinstance(ocr_result, list):
                if debug:
                    print(f"  [调试] 使用旧版本列表格式")
                
                for idx, line in enumerate(ocr_result):
                    if line and len(line) >= 2:
                        try:
                            box = line[0]
                            text_info = line[1]
                            
                            # 提取文本和置信度
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = float(text_info[1])
                            elif isinstance(text_info, (list, tuple)) and len(text_info) == 1:
                                text = text_info[0]
                                confidence = 1.0
                            else:
                                text = str(text_info)
                                confidence = 1.0
                            
                            # 过滤低置信度和空文本
                            if text and len(text.strip()) > 0 and confidence > config.OCR_MIN_CONFIDENCE:
                                # 调试第一个结果
                                if debug and len(text_boxes) == 0:
                                    print(f"  [调试] 第1个文本: '{text}', 置信度: {confidence:.3f}")
                                    print(f"  [调试] 第1个框坐标: {box}")
                                    if len(box) >= 4:
                                        xs = [pt[0] for pt in box]
                                        ys = [pt[1] for pt in box]
                                        print(f"  [调试] X范围: [{min(xs):.1f}, {max(xs):.1f}], Y范围: [{min(ys):.1f}, {max(ys):.1f}]")
                                        print(f"  [调试] 框宽度: {max(xs)-min(xs):.1f}, 框高度: {max(ys)-min(ys):.1f}")
                                
                                text_boxes.append((text, box))
                            
                        except (IndexError, TypeError, ValueError) as e:
                            if debug:
                                print(f"  [调试] 解析第{idx}行失败: {e}")
                            continue
        
        if debug:
            print(f"  [调试] 成功解析 {len(text_boxes)} 个文本框")
            
    except Exception as e:
        print(f"  [!] 解析OCR结果时出错: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    return text_boxes

def draw_boxes_on_image(image_path, text_boxes, output_path):
    """在图片上绘制文本框"""
    img = Image.open(image_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 加载字体
    try:
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/System/Library/Fonts/PingFang.ttc",
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, config.LABEL_FONT_SIZE)
                    break
                except:
                    continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    print(f"  => 绘制 {len(text_boxes)} 个文本框...")
    
    # 绘制每个文本框
    for idx, (text, box) in enumerate(text_boxes):
        if len(box) < 4:
            continue
        
        # 选择颜色
        color = config.COLORS[idx % len(config.COLORS)]
        
        # 转换坐标并应用偏移量
        points = [(float(pt[0]) + config.BOX_OFFSET_X, float(pt[1]) + config.BOX_OFFSET_Y) for pt in box]
        
        # 绘制半透明填充
        fill_color = color + (50,)
        draw.polygon(points, fill=fill_color, outline=None)
        
        # 绘制边框
        outline_color = color + (config.BOX_OPACITY,)
        draw.line(points + [points[0]], fill=outline_color, width=config.BOX_WIDTH)
        
        # 绘制文本标签
        if config.SHOW_TEXT_LABEL and text:
            # 计算文本框的边界
            min_x = min(pt[0] for pt in points)
            min_y = min(pt[1] for pt in points)
            max_x = max(pt[0] for pt in points)
            
            # 截断过长的文本
            display_text = text[:15] + '...' if len(text) > 15 else text
            
            # 计算标签位置（在文本框上方）
            # Y坐标 = 文本框顶部 - 字体大小 - 垂直偏移量 - 内边距
            label_y = min_y - config.LABEL_FONT_SIZE - config.LABEL_OFFSET_Y - config.LABEL_PADDING
            
            # X坐标 = 文本框左边 + 水平偏移量
            label_x = min_x + config.LABEL_OFFSET_X
            
            # 边界保护：防止标签超出图片边界
            label_y = max(config.LABEL_PADDING, label_y)  # 不能超出顶部
            label_x = max(config.LABEL_PADDING, label_x)  # 不能超出左侧
            
            try:
                # 获取文本边界框（用于绘制背景）
                bbox = draw.textbbox((label_x, label_y), display_text, font=font)
                
                # 添加内边距到背景框
                padded_bbox = (
                    bbox[0] - config.LABEL_PADDING,
                    bbox[1] - config.LABEL_PADDING,
                    bbox[2] + config.LABEL_PADDING,
                    bbox[3] + config.LABEL_PADDING
                )
                
                # 再次检查右边界
                img_width = overlay.size[0]
                if padded_bbox[2] > img_width:
                    # 如果标签超出右边界，向左调整
                    shift = padded_bbox[2] - img_width + config.LABEL_PADDING
                    label_x -= shift
                    bbox = draw.textbbox((label_x, label_y), display_text, font=font)
                    padded_bbox = (
                        bbox[0] - config.LABEL_PADDING,
                        bbox[1] - config.LABEL_PADDING,
                        bbox[2] + config.LABEL_PADDING,
                        bbox[3] + config.LABEL_PADDING
                    )
                
                # 绘制标签背景
                draw.rectangle(padded_bbox, fill=color + (config.LABEL_BG_OPACITY,))
                
                # 绘制标签文字（白色）
                draw.text((label_x, label_y), display_text, fill=(255, 255, 255, 255), font=font)
            except Exception as e:
                # 降级方案：直接绘制文字，不带背景
                draw.text((label_x, label_y), display_text, fill=outline_color, font=font)
    
    # 合并图层
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    img.save(output_path, 'PNG')
    
    print(f"  => 已保存标注图片: {output_path}")
    return output_path

def images_to_pdf(image_paths, output_pdf_path):
    """将图片转换为PDF"""
    if not image_paths:
        return False
    
    c = canvas.Canvas(output_pdf_path)
    
    for img_path in image_paths:
        with Image.open(img_path) as img:
            w, h = img.size
            c.setPageSize((w, h))
            c.drawImage(img_path, 0, 0, width=w, height=h)
            c.showPage()
    
    c.save()
    print(f"=> PDF已生成: {output_pdf_path}")
    return True

# ==================== 主处理流程 ====================

def process_pdf_with_ocr_boxes(pdf_path, output_name=None):
    """处理PDF并标注文本框"""
    if not os.path.exists(pdf_path):
        print(f"[X] 文件不存在: {pdf_path}")
        return
    
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"开始处理PDF: {pdf_path}")
    print(f"{'='*60}\n")
    
    # 1. 初始化PaddleOCR
    print("[1/5] 初始化PaddleOCR...")
    
    # 确定设备类型（PaddleOCR新版本使用device参数）
    paddle_device = 'cpu'  # PaddleOCR暂不支持Intel XPU，统一使用CPU
    
    if device == 'xpu':
        print("=> 检测到Intel XPU，PaddleOCR将使用CPU模式（暂不支持XPU）")
    
    try:
        # 获取PaddleOCR版本信息
        import paddleocr
        paddle_version = paddleocr.__version__ if hasattr(paddleocr, '__version__') else 'unknown'
        print(f"=> PaddleOCR版本: {paddle_version}")
        
        # 显示语言配置
        lang_names = {'japan': '日语', 'ch': '中文', 'en': '英文'}
        lang_display = lang_names.get(config.LANG, config.LANG)
        print(f"=> 使用语言: {lang_display} (lang='{config.LANG}')")
        
        # 准备OCR参数（用于提高检测精度）
        # 注意：新版PaddleOCR (>=3.0) 参数名已更改
        ocr_params = {
            'lang': config.LANG,
            'text_det_thresh': config.OCR_DET_DB_THRESH,  # 新版参数名
            'text_det_box_thresh': config.OCR_DET_DB_BOX_THRESH,  # 新版参数名
            'text_det_unclip_ratio': config.OCR_DET_DB_UNCLIP_RATIO,  # 新版参数名
            'show_log': False,
        }
        
        print(f"=> OCR精度参数:")
        print(f"   - text_det_thresh: {config.OCR_DET_DB_THRESH} (二值化阈值)")
        print(f"   - text_det_box_thresh: {config.OCR_DET_DB_BOX_THRESH} (框置信度)")
        print(f"   - text_det_unclip_ratio: {config.OCR_DET_DB_UNCLIP_RATIO} (框扩张比例 ⭐关键!)")
        
        # 智能初始化 - 尝试多种配置方式
        paddle_ocr = None
        init_method = ""
        
        # 方法1: 完整参数配置（推荐）
        try:
            paddle_ocr = PaddleOCR(**ocr_params)
            init_method = "完整参数"
        except Exception as e1:
            # 方法2: 尝试添加device参数
            try:
                ocr_params['device'] = paddle_device
                paddle_ocr = PaddleOCR(**ocr_params)
                init_method = f"完整参数+device='{paddle_device}'"
            except Exception as e2:
                # 方法3: 尝试use_gpu参数（旧版本）
                try:
                    ocr_params.pop('device', None)
                    ocr_params['use_gpu'] = False
                    paddle_ocr = PaddleOCR(**ocr_params)
                    init_method = "完整参数+use_gpu=False"
                except Exception as e3:
                    # 方法4: 最简配置（降级方案）
                    try:
                        paddle_ocr = PaddleOCR(lang=config.LANG)
                        init_method = "最简配置(降级)"
                        print("[!] 无法使用精度参数，使用默认配置")
                    except Exception as e4:
                        # 所有方法都失败
                        print(f"[X] PaddleOCR初始化失败:")
                        print(f"   方法1 (完整参数): {e1}")
                        print(f"   方法2 (device): {e2}")
                        print(f"   方法3 (use_gpu): {e3}")
                        print(f"   方法4 (最简): {e4}")
                        print(f"\n提示: 首次使用{lang_display}模型需要下载，请确保网络连接正常")
                        print("\n请检查安装:")
                        print("  pip install paddlepaddle paddleocr")
                        print("\n或重新安装:")
                        print("  pip uninstall -y paddlepaddle paddleocr")
                        print("  pip install paddlepaddle paddleocr")
                        return
        
        print(f"=> OCR引擎初始化完成 ({init_method})\n")
            
    except Exception as e:
        print(f"[X] PaddleOCR初始化异常: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. PDF转图片
    print("[2/5] 步骤1: 将PDF转换为图片...")
    temp_folder = "temp_pdf_images"
    image_paths = pdf_to_images(pdf_path, temp_folder, dpi=config.DPI)
    print(f"=> 完成！共 {len(image_paths)} 页\n")
    
    # 3. OCR识别并标注
    print("[3/5] 步骤2: OCR识别并标注文本框...")
    annotated_images = []
    all_texts = []
    
    import time
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n--- 第 {i}/{len(image_paths)} 页 ---")
        
        start_time = time.time()
        
        # 使用PaddleOCR获取文本框（启用调试模式）
        debug_mode = (i == 1)  # 只在第一页显示调试信息
        text_boxes = ocr_with_paddleocr(image_path, paddle_ocr, debug=debug_mode)
        
        elapsed = time.time() - start_time
        print(f"  => OCR耗时: {elapsed:.2f}秒")
        print(f"  => 识别到 {len(text_boxes)} 个文本块")
        
        # 提取纯文本
        texts = [text for text, _ in text_boxes]
        cleaned_text = '\n'.join(texts)
        print(f"  => 文本长度: {len(cleaned_text)} 字符")
        
        # 绘制标注
        output_image_path = os.path.join(
            config.OUTPUT_FOLDER,
            f"{output_name}_page_{i}_annotated.png"
        )
        draw_boxes_on_image(image_path, text_boxes, output_image_path)
        annotated_images.append(output_image_path)
        
        # 保存文本
        all_texts.append(f"# 第 {i} 页\n\n{cleaned_text}\n\n")
        
        # 显示部分文本
        if texts:
            print(f"  => 前5个文本块:")
            for idx, text in enumerate(texts[:5], 1):
                print(f"     [{idx}] {text[:30]}{'...' if len(text) > 30 else ''}")
    
    print(f"\n=> OCR识别完成！\n")
    
    # 4. 生成PDF
    print("[4/5] 步骤3: 生成标注PDF...")
    output_pdf = os.path.join(config.OUTPUT_FOLDER, f"{output_name}_annotated.pdf")
    images_to_pdf(annotated_images, output_pdf)
    print()
    
    # 5. 保存文本
    print("[5/5] 步骤4: 保存识别文本...")
    output_text = os.path.join(config.OUTPUT_FOLDER, f"{output_name}_ocr_text.md")
    with open(output_text, 'w', encoding='utf-8') as f:
        f.writelines(all_texts)
    print(f"=> 文本已保存: {output_text}\n")
    
    # 6. 清理临时文件
    print("清理临时文件...")
    import shutil
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
        print("=> 已清理临时文件\n")
    
    # 总结
    print(f"{'='*60}")
    print("=> 处理完成！")
    print(f"{'='*60}")
    print(f"输出文件夹: {config.OUTPUT_FOLDER}/")
    print(f"   - 标注PDF: {output_name}_annotated.pdf")
    print(f"   - 识别文本: {output_name}_ocr_text.md")
    print(f"   - 标注图片: {output_name}_page_*_annotated.png ({len(annotated_images)}张)")
    print(f"{'='*60}\n")

# ==================== 命令行入口 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF OCR文本框可视化工具 V2 (PaddleOCR)')
    parser.add_argument('pdf_file', help='输入PDF文件路径')
    parser.add_argument('-o', '--output', help='输出文件名前缀')
    parser.add_argument('--lang', choices=['japan', 'ch', 'en'], default='japan', 
                        help='OCR语言 (japan=日语, ch=中文, en=英文，默认: japan)')
    parser.add_argument('--no-labels', action='store_true', help='不显示文本标签')
    parser.add_argument('--dpi', type=int, default=150, help='PDF转图片DPI')
    parser.add_argument('--box-width', type=int, default=3, help='方框线条宽度')
    parser.add_argument('--offset-x', type=int, default=0, help='标注框X轴偏移量（正数向右，负数向左）')
    parser.add_argument('--offset-y', type=int, default=0, help='标注框Y轴偏移量（正数向下，负数向上）')
    
    args = parser.parse_args()
    
    # 应用配置
    if args.lang:
        config.LANG = args.lang
    if args.no_labels:
        config.SHOW_TEXT_LABEL = False
    if args.dpi:
        config.DPI = args.dpi
    if args.box_width:
        config.BOX_WIDTH = args.box_width
    if args.offset_x:
        config.BOX_OFFSET_X = args.offset_x
    if args.offset_y:
        config.BOX_OFFSET_Y = args.offset_y
    
    process_pdf_with_ocr_boxes(args.pdf_file, args.output)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        default_pdf = 'japanese_test.pdf'
        if os.path.exists(default_pdf):
            print(f"使用默认文件: {default_pdf}\n")
            process_pdf_with_ocr_boxes(default_pdf)
        else:
            print("用法: python pdf_ocr_with_boxes_v2.py <pdf文件>")
    else:
        main()

