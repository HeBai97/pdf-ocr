"""
PaddleOCR 批量测试脚本

用于批量处理文件夹下的所有图片
"""

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import os
import glob

def extract_texts_from_result(result):
    """从OCR结果中提取文本"""
    texts = []
    
    if not result or not isinstance(result, list) or len(result) == 0:
        return texts
    
    ocr_result = result[0]
    
    # 新版本：字典格式（包含rec_texts, rec_scores, rec_polys键）
    if isinstance(ocr_result, dict) and 'rec_texts' in ocr_result:
        rec_texts = ocr_result.get('rec_texts', [])
        rec_scores = ocr_result.get('rec_scores', [1.0] * len(rec_texts))
        
        for text, score in zip(rec_texts, rec_scores):
            if text and score > 0.5:
                texts.append(text)
    
    # 中版本：OCRResult对象
    elif hasattr(ocr_result, 'rec_texts'):
        rec_texts = ocr_result.rec_texts
        rec_scores = ocr_result.rec_scores if hasattr(ocr_result, 'rec_scores') else [1.0] * len(rec_texts)
        
        for text, score in zip(rec_texts, rec_scores):
            if text and score > 0.5:
                texts.append(text)
    
    # 旧版本：列表格式
    elif isinstance(ocr_result, list):
        for item in ocr_result:
            if item and len(item) >= 2:
                text_info = item[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                    text = text_info[0]
                    confidence = text_info[1] if len(text_info) >= 2 else 1.0
                    if text and confidence > 0.5:
                        texts.append(text)
    
    return texts

def test_paddleocr():
    """批量测试PaddleOCR识别所有图片"""
    
    # 查找测试图片
    test_images = []
    
    # 查找ocr_boxes_output文件夹中的图片
    if os.path.exists('ocr_boxes_output'):
        for f in os.listdir('ocr_boxes_output'):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join('ocr_boxes_output', f))
    
    # 如果没有，查找当前目录下的所有图片
    if not test_images:
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            found = glob.glob(ext)
            test_images.extend(found)
    
    if not test_images:
        print("❌ 未找到测试图片")
        print("请先运行: python pdf_ocr_with_boxes_v2.py")
        return
    
    # 排序图片列表
    test_images.sort()
    
    print(f"📂 找到 {len(test_images)} 张图片")
    print("="*60)
    
    # 初始化PaddleOCR（只初始化一次）
    print("初始化PaddleOCR...")
    
    # 自动检测语言（根据文件名判断）
    lang = 'japan'  # 默认日语
    sample_image = test_images[0]
    if 'japanese' in sample_image.lower() or 'japan' in sample_image.lower():
        lang = 'japan'
    elif 'chinese' in sample_image.lower() or 'cn' in sample_image.lower():
        lang = 'ch'
    
    print(f"使用语言模型: {lang}")
    
    # 尝试多种初始化方式
    ocr = None
    init_method = ""
    
    try:
        ocr = PaddleOCR(lang=lang)
        init_method = f"最简配置 (lang='{lang}')"
        print(f"✓ 初始化成功 - {init_method}\n")
    except Exception as e1:
        try:
            ocr = PaddleOCR(lang=lang, device='cpu')
            init_method = f"device='cpu' (lang='{lang}')"
            print(f"✓ 初始化成功 - {init_method}\n")
        except Exception as e2:
            try:
                ocr = PaddleOCR(lang=lang, use_gpu=False)
                init_method = f"use_gpu=False (lang='{lang}')"
                print(f"✓ 初始化成功 - {init_method}\n")
            except Exception as e3:
                print("\n❌ 所有初始化方法都失败了")
                print(f"错误: {e3}")
                return
    
    # 批量处理所有图片
    all_results = []
    
    for idx, test_image in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] 处理: {test_image}")
        print("-" * 60)
        
        try:
            # 读取图片
            img = Image.open(test_image)
            img_array = np.array(img)
            print(f"  图片尺寸: {img.size}, 模式: {img.mode}")
            
            # 执行OCR
            result = None
            try:
                result = ocr.predict(img_array)
            except AttributeError:
                result = ocr.ocr(img_array)
            
            # 提取文本
            texts = extract_texts_from_result(result)
            
            print(f"  ✓ 识别到 {len(texts)} 个文本块")
            
            # 保存结果
            all_results.append({
                'image': test_image,
                'texts': texts,
                'count': len(texts)
            })
            
            # 显示前3个文本
            if texts:
                for i, text in enumerate(texts[:3], 1):
                    print(f"    {i}. {text}")
                if len(texts) > 3:
                    print(f"    ... 还有 {len(texts) - 3} 个文本")
        
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            all_results.append({
                'image': test_image,
                'texts': [],
                'count': 0,
                'error': str(e)
            })
    
    # 保存所有结果到文件
    print("\n" + "="*60)
    print("保存结果...")
    print("="*60)
    
    output_file = "paddleocr_batch_result.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PaddleOCR 批量识别结果\n")
        f.write("="*60 + "\n\n")
        f.write(f"总共处理: {len(test_images)} 张图片\n")
        f.write(f"使用方法: {init_method}\n\n")
        
        total_texts = sum(r['count'] for r in all_results)
        f.write(f"总共识别: {total_texts} 个文本块\n\n")
        f.write("="*60 + "\n\n")
        
        for idx, result in enumerate(all_results, 1):
            f.write(f"\n[{idx}] {result['image']}\n")
            f.write("-" * 60 + "\n")
            
            if 'error' in result:
                f.write(f"❌ 错误: {result['error']}\n")
            else:
                f.write(f"识别到 {result['count']} 个文本块:\n\n")
                for i, text in enumerate(result['texts'], 1):
                    f.write(f"{i}. {text}\n")
            
            f.write("\n")
    
    print(f"\n✓ 结果已保存到: {output_file}")
    
    # 打印统计信息
    success_count = sum(1 for r in all_results if 'error' not in r and r['count'] > 0)
    total_texts = sum(r['count'] for r in all_results)
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"成功处理: {success_count}/{len(test_images)} 张图片")
    print(f"总共识别: {total_texts} 个文本块")
    print("="*60)

if __name__ == "__main__":
    try:
        test_paddleocr()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

