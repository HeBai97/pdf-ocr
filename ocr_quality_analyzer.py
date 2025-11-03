"""
OCR识别质量分析工具

用于诊断PaddleOCR识别质量问题：
- 显示每个文本框的置信度
- 统计低置信度识别
- 分析框位置偏移问题
"""

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def analyze_ocr_quality(image_path, lang='japan', min_confidence=0.3):
    """分析OCR识别质量"""
    
    print(f"\n{'='*70}")
    print(f"分析图片: {image_path}")
    print(f"{'='*70}\n")
    
    # 初始化OCR
    print("初始化PaddleOCR...")
    ocr = PaddleOCR(lang=lang)
    
    # 读取图片
    img = Image.open(image_path)
    img_array = np.array(img)
    
    print(f"图片尺寸: {img.size[0]}x{img.size[1]}\n")
    
    # 执行OCR
    print("执行OCR识别...")
    try:
        result = ocr.predict(img_array)
    except AttributeError:
        try:
            result = ocr.ocr(img_array, cls=False)
        except TypeError:
            result = ocr.ocr(img_array)
    
    # 解析结果
    text_boxes = []
    
    if result and isinstance(result, list) and len(result) > 0:
        ocr_result = result[0]
        
        # 支持多种格式
        if isinstance(ocr_result, dict) and 'rec_texts' in ocr_result:
            rec_texts = ocr_result.get('rec_texts', [])
            rec_scores = ocr_result.get('rec_scores', [1.0] * len(rec_texts))
            rec_polys = ocr_result.get('rec_polys', [])
            
            for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                if text and len(text.strip()) > 0:
                    text_boxes.append({
                        'text': text,
                        'confidence': float(score),
                        'box': poly.tolist() if hasattr(poly, 'tolist') else poly
                    })
        
        elif hasattr(ocr_result, 'rec_texts') and hasattr(ocr_result, 'rec_polys'):
            rec_texts = ocr_result.rec_texts
            rec_scores = ocr_result.rec_scores if hasattr(ocr_result, 'rec_scores') else [1.0] * len(rec_texts)
            rec_polys = ocr_result.rec_polys
            
            for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                if text and len(text.strip()) > 0:
                    text_boxes.append({
                        'text': text,
                        'confidence': float(score),
                        'box': poly.tolist() if hasattr(poly, 'tolist') else poly
                    })
        
        elif isinstance(ocr_result, list):
            for line in ocr_result:
                if line and len(line) >= 2:
                    box = line[0]
                    text_info = line[1]
                    
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = float(text_info[1])
                    else:
                        text = str(text_info)
                        confidence = 1.0
                    
                    if text and len(text.strip()) > 0:
                        text_boxes.append({
                            'text': text,
                            'confidence': confidence,
                            'box': box
                        })
    
    # 统计分析
    print(f"\n{'='*70}")
    print(f"识别结果统计")
    print(f"{'='*70}\n")
    
    total = len(text_boxes)
    print(f"总识别数量: {total}")
    
    if total == 0:
        print("⚠️  未识别到任何文字！")
        return
    
    # 按置信度分类
    high_conf = [t for t in text_boxes if t['confidence'] >= 0.8]
    medium_conf = [t for t in text_boxes if 0.5 <= t['confidence'] < 0.8]
    low_conf = [t for t in text_boxes if t['confidence'] < 0.5]
    
    print(f"\n置信度分布:")
    print(f"  高 (≥0.8):  {len(high_conf):3d} 个 ({len(high_conf)/total*100:5.1f}%)")
    print(f"  中 (0.5-0.8): {len(medium_conf):3d} 个 ({len(medium_conf)/total*100:5.1f}%)")
    print(f"  低 (<0.5):  {len(low_conf):3d} 个 ({len(low_conf)/total*100:5.1f}%)")
    
    # 按文本长度分类
    single_char = [t for t in text_boxes if len(t['text'].strip()) == 1]
    short_text = [t for t in text_boxes if 2 <= len(t['text'].strip()) <= 5]
    long_text = [t for t in text_boxes if len(t['text'].strip()) > 5]
    
    print(f"\n文本长度分布:")
    print(f"  单字符:   {len(single_char):3d} 个")
    print(f"  短文本(2-5字): {len(short_text):3d} 个")
    print(f"  长文本(>5字):  {len(long_text):3d} 个")
    
    # 显示低置信度文本
    if low_conf:
        print(f"\n{'='*70}")
        print(f"⚠️  低置信度文本 (置信度 < 0.5) - 可能识别错误")
        print(f"{'='*70}\n")
        
        for idx, item in enumerate(sorted(low_conf, key=lambda x: x['confidence']), 1):
            print(f"{idx:2d}. [{item['confidence']:.3f}] {item['text'][:30]}")
            if idx >= 20:  # 只显示前20个
                print(f"    ... 还有 {len(low_conf) - 20} 个")
                break
    
    # 显示中等置信度文本
    if medium_conf:
        print(f"\n{'='*70}")
        print(f"⚠️  中等置信度文本 (0.5 ≤ 置信度 < 0.8) - 需要注意")
        print(f"{'='*70}\n")
        
        for idx, item in enumerate(sorted(medium_conf, key=lambda x: x['confidence']), 1):
            print(f"{idx:2d}. [{item['confidence']:.3f}] {item['text'][:30]}")
            if idx >= 20:
                print(f"    ... 还有 {len(medium_conf) - 20} 个")
                break
    
    # 文本框尺寸分析
    print(f"\n{'='*70}")
    print(f"文本框尺寸分析")
    print(f"{'='*70}\n")
    
    box_sizes = []
    for item in text_boxes:
        box = item['box']
        if len(box) >= 4:
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            box_sizes.append((width, height))
    
    if box_sizes:
        widths = [w for w, h in box_sizes]
        heights = [h for w, h in box_sizes]
        
        print(f"宽度: 最小={min(widths):.0f}px, 最大={max(widths):.0f}px, 平均={np.mean(widths):.0f}px")
        print(f"高度: 最小={min(heights):.0f}px, 最大={max(heights):.0f}px, 平均={np.mean(heights):.0f}px")
        
        # 找出异常尺寸的框
        small_boxes = [(w, h) for w, h in box_sizes if w < 20 or h < 20]
        large_boxes = [(w, h) for w, h in box_sizes if w > 500 or h > 100]
        
        if small_boxes:
            print(f"\n⚠️  检测到 {len(small_boxes)} 个过小的文本框（可能识别错误）")
        if large_boxes:
            print(f"⚠️  检测到 {len(large_boxes)} 个过大的文本框（可能合并了多个文本）")
    
    # 生成带置信度标签的可视化图片
    print(f"\n{'='*70}")
    print(f"生成可视化图片...")
    print(f"{'='*70}\n")
    
    output_path = image_path.replace('.', '_quality_analysis.')
    visualize_with_confidence(image_path, text_boxes, output_path, min_confidence)
    
    print(f"\n✓ 分析完成！")
    print(f"  可视化图片已保存: {output_path}\n")
    
    return text_boxes


def visualize_with_confidence(image_path, text_boxes, output_path, min_confidence=0.3):
    """生成带置信度的可视化图片"""
    
    img = Image.open(image_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 加载字体
    try:
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 14)
                    break
                except:
                    continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 根据置信度选择颜色
    for item in text_boxes:
        text = item['text']
        confidence = item['confidence']
        box = item['box']
        
        if len(box) < 4:
            continue
        
        # 根据置信度选择颜色
        if confidence >= 0.8:
            color = (0, 255, 0)  # 绿色 - 高置信度
        elif confidence >= 0.5:
            color = (255, 165, 0)  # 橙色 - 中等置信度
        else:
            color = (255, 0, 0)  # 红色 - 低置信度
        
        # 转换坐标
        points = [(float(pt[0]), float(pt[1])) for pt in box]
        
        # 绘制半透明填充
        fill_color = color + (30,)
        draw.polygon(points, fill=fill_color, outline=None)
        
        # 绘制边框
        outline_color = color + (180,)
        draw.line(points + [points[0]], fill=outline_color, width=2)
        
        # 绘制文本标签（包含置信度）
        min_x = min(pt[0] for pt in points)
        min_y = min(pt[1] for pt in points)
        
        display_text = f"{text[:10]} [{confidence:.2f}]" if len(text) > 10 else f"{text} [{confidence:.2f}]"
        
        label_y = max(2, min_y - 18)
        label_x = max(2, min_x)
        
        try:
            bbox = draw.textbbox((label_x, label_y), display_text, font=font)
            padded_bbox = (bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1)
            draw.rectangle(padded_bbox, fill=color + (220,))
            draw.text((label_x, label_y), display_text, fill=(255, 255, 255, 255), font=font)
        except:
            draw.text((label_x, label_y), display_text, fill=outline_color, font=font)
    
    # 添加图例
    legend_y = 10
    legend_x = img.size[0] - 250
    
    legend_items = [
        ("高置信度 (≥0.8)", (0, 255, 0)),
        ("中等置信度 (0.5-0.8)", (255, 165, 0)),
        ("低置信度 (<0.5)", (255, 0, 0)),
    ]
    
    for text, color in legend_items:
        # 背景
        bbox = draw.textbbox((legend_x + 30, legend_y), text, font=font)
        padded_bbox = (legend_x - 5, bbox[1] - 2, bbox[2] + 5, bbox[3] + 2)
        draw.rectangle(padded_bbox, fill=(255, 255, 255, 200))
        
        # 色块
        draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 15], fill=color + (200,), outline=color + (255,))
        
        # 文字
        draw.text((legend_x + 30, legend_y), text, fill=(0, 0, 0, 255), font=font)
        legend_y += 25
    
    # 合并图层
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    img.save(output_path, 'PNG')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR识别质量分析工具')
    parser.add_argument('image', help='输入图片路径')
    parser.add_argument('--lang', default='japan', choices=['japan', 'ch', 'en'],
                       help='OCR语言 (默认: japan)')
    parser.add_argument('--min-conf', type=float, default=0.3,
                       help='最低置信度阈值 (默认: 0.3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ 文件不存在: {args.image}")
        return
    
    analyze_ocr_quality(args.image, args.lang, args.min_conf)


if __name__ == "__main__":
    # 如果直接运行，分析 temp_pdf_images 文件夹中的第一张图片
    import sys
    
    if len(sys.argv) == 1:
        test_image = "temp_pdf_images/page_1.png"
        if os.path.exists(test_image):
            print(f"使用默认图片: {test_image}\n")
            analyze_ocr_quality(test_image)
        else:
            print("用法: python ocr_quality_analyzer.py <图片文件>")
            print("\n示例:")
            print("  python ocr_quality_analyzer.py temp_pdf_images/page_1.png")
            print("  python ocr_quality_analyzer.py image.jpg --lang ch")
    else:
        main()

