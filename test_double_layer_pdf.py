"""
双层PDF生成功能测试脚本

此脚本用于快速测试双层PDF生成功能，无需运行完整的OCR处理流程。

使用方法:
    python test_double_layer_pdf.py
"""

import os
import glob
from parse_student_answers import pdf_generator, generate_pdf_from_images

def test_from_existing_images():
    """测试从已处理的images文件夹中的图片生成PDF"""
    print("\n" + "="*60)
    print("测试1: 从images文件夹生成PDF")
    print("="*60)
    
    # 查找images文件夹中的所有图片
    image_folder = 'images'
    if not os.path.exists(image_folder):
        print(f"❌ {image_folder} 文件夹不存在")
        print("   请先运行 parse_student_answers.py 进行OCR识别")
        return False
    
    # 获取所有png图片并排序
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    
    if not image_files:
        print(f"❌ {image_folder} 文件夹中没有图片")
        return False
    
    print(f"找到 {len(image_files)} 张图片:")
    for img in image_files:
        print(f"  - {img}")
    
    # 询问用户选择
    print("\n请选择要生成PDF的图片:")
    print("1. 使用所有图片")
    print("2. 只使用名称中包含'270度'或'0度'的图片（最佳旋转角度）")
    print("3. 手动选择")
    
    choice = input("请输入选项 (1/2/3，默认为2): ").strip() or "2"
    
    selected_images = []
    
    if choice == "1":
        selected_images = image_files
    elif choice == "2":
        # 智能选择：每个文档选择一个最佳角度
        doc_images = {}
        for img in image_files:
            # 提取文档名称（如"学生答卷_第1张"）
            basename = os.path.basename(img)
            doc_name = basename.rsplit('_', 1)[0]  # 去掉角度部分
            
            if doc_name not in doc_images:
                doc_images[doc_name] = []
            doc_images[doc_name].append(img)
        
        # 为每个文档选择第一个（通常是最佳角度）
        for doc_name in sorted(doc_images.keys()):
            # 优先选择270度或0度的版本
            candidates = doc_images[doc_name]
            best = None
            for candidate in candidates:
                if '270度' in candidate or '0度' in candidate:
                    best = candidate
                    break
            if best is None:
                best = candidates[0]
            selected_images.append(best)
    
    elif choice == "3":
        print("\n可用图片:")
        for idx, img in enumerate(image_files, 1):
            print(f"  {idx}. {os.path.basename(img)}")
        
        indices = input("请输入图片编号（用逗号分隔，如 1,3,5）: ").strip()
        try:
            selected_indices = [int(x.strip()) - 1 for x in indices.split(',')]
            selected_images = [image_files[i] for i in selected_indices if 0 <= i < len(image_files)]
        except (ValueError, IndexError):
            print("❌ 无效的输入")
            return False
    
    if not selected_images:
        print("❌ 没有选择任何图片")
        return False
    
    print(f"\n已选择 {len(selected_images)} 张图片:")
    for img in selected_images:
        print(f"  ✓ {img}")
    
    # 生成PDF
    output_pdf = input("\n输入输出PDF文件名（默认: test_output.pdf）: ").strip() or "test_output.pdf"
    
    if not output_pdf.endswith('.pdf'):
        output_pdf += '.pdf'
    
    success = generate_pdf_from_images(selected_images, output_pdf)
    
    if success:
        print(f"\n{'='*60}")
        print(f"✅ 测试成功！")
        print(f"{'='*60}")
        print(f"PDF文件已生成: {output_pdf}")
        print(f"文件大小: {os.path.getsize(output_pdf) / 1024:.2f} KB")
        print(f"\n请使用PDF阅读器打开文件测试:")
        print(f"  1. 文本搜索功能 (Ctrl+F)")
        print(f"  2. 文本复制功能")
        print(f"  3. 确认图片清晰度")
        return True
    else:
        print("\n❌ PDF生成失败")
        return False

def test_from_original_images():
    """测试从原始图片生成PDF（需要重新OCR）"""
    print("\n" + "="*60)
    print("测试2: 从原始图片生成PDF")
    print("="*60)
    
    # 查找项目根目录的jpg图片
    jpg_files = glob.glob('*.jpg')
    
    if not jpg_files:
        print("❌ 当前目录没有找到jpg图片")
        return False
    
    print(f"找到 {len(jpg_files)} 张原始图片:")
    for img in jpg_files:
        print(f"  - {img}")
    
    confirm = input("\n是否使用这些图片生成PDF？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return False
    
    output_pdf = "test_from_originals.pdf"
    
    print("\n⚠️  注意: 此操作会对每张图片执行OCR识别，可能需要较长时间...")
    
    success = generate_pdf_from_images(jpg_files, output_pdf)
    
    if success:
        print(f"\n✅ PDF生成成功: {output_pdf}")
        return True
    else:
        print("\n❌ PDF生成失败")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("双层PDF生成功能测试")
    print("="*60)
    
    print("\n可用测试:")
    print("1. 从已处理的images文件夹生成PDF（推荐，快速）")
    print("2. 从原始jpg图片生成PDF（需要重新OCR，较慢）")
    print("3. 退出")
    
    choice = input("\n请选择测试 (1/2/3): ").strip()
    
    if choice == "1":
        test_from_existing_images()
    elif choice == "2":
        test_from_original_images()
    elif choice == "3":
        print("退出测试")
    else:
        print("无效的选择")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

