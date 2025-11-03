# PDF OCR 工具集

一套功能强大的PDF和图片OCR识别工具，支持多语言识别、文本框可视化、质量分析和双层PDF生成。

## 🌟 主要特性

- **多模型支持**：PaddleOCR 和 DeepSeek-OCR
- **多语言识别**：日语、中文、英语
- **硬件加速**：支持 Intel XPU (Arc GPU) 和 CPU
- **可视化标注**：自动标注识别文本框
- **质量分析**：OCR识别质量诊断工具
- **双层PDF**：生成可搜索、可复制文本的PDF
- **智能方向检测**：自动检测图片旋转角度

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

- `transformers` - HuggingFace Transformers（用于DeepSeek-OCR）
- `torch` - PyTorch深度学习框架
- `paddleocr` - 百度PaddleOCR
- `PyMuPDF` - PDF处理
- `Pillow` - 图像处理
- `reportlab` - PDF生成
- `pytesseract` - 图片方向检测（可选）
- `intel_extension_for_pytorch` - Intel XPU加速（可选）

## 🚀 快速开始

### 1. PDF文本框可视化（PaddleOCR）

使用PaddleOCR识别PDF中的文本并可视化标注文本框位置。

```bash
# 基础用法
python pdf_ocr_with_boxes_v2.py <pdf文件路径>

# 示例
python pdf_ocr_with_boxes_v2.py japanese_test.pdf

# 指定语言
python pdf_ocr_with_boxes_v2.py document.pdf --lang ch

# 调整DPI和显示选项
python pdf_ocr_with_boxes_v2.py document.pdf --dpi 300 --no-labels
```

**输出文件**：
- `ocr_boxes_output/<文件名>_annotated.pdf` - 标注了文本框的PDF
- `ocr_boxes_output/<文件名>_ocr_text.md` - 识别的文本内容
- `ocr_boxes_output/<文件名>_page_*_annotated.png` - 每页的标注图片

**参数说明**：
- `--lang`: 识别语言 (`japan`/`ch`/`en`，默认: `japan`)
- `--dpi`: PDF转图片DPI (默认: 150，日语推荐200-300)
- `--no-labels`: 不显示文本标签（只显示彩色框）
- `--box-width`: 文本框边框宽度 (默认: 3)
- `--offset-x`: 文本框X轴偏移量
- `--offset-y`: 文本框Y轴偏移量

### 2. PDF转Markdown（DeepSeek-OCR）

使用DeepSeek-OCR模型将PDF转换为Markdown格式文档。

```bash
python pdf_to_markdown.py
```

**特点**：
- 高质量文档识别
- 自动转换为Markdown格式
- 保留文档结构
- 支持Intel XPU加速

**输出文件**：
- `japanese_test_output.md` - Markdown格式的识别结果

### 3. OCR质量分析工具

诊断PaddleOCR的识别质量问题，显示置信度分布和可能的错误。

```bash
# 分析单张图片
python ocr_quality_analyzer.py <图片路径>

# 示例
python ocr_quality_analyzer.py temp_pdf_images/page_1.png

# 指定语言
python ocr_quality_analyzer.py image.jpg --lang ch

# 设置最低置信度阈值
python ocr_quality_analyzer.py image.jpg --min-conf 0.5
```

**功能**：
- 显示每个文本框的置信度
- 统计置信度分布（高/中/低）
- 分析文本长度分布
- 检测异常尺寸文本框
- 生成带置信度标签的可视化图片

**输出**：
- 终端显示详细分析报告
- 生成 `<文件名>_quality_analysis.png` 可视化图片

### 4. 学生答卷处理与双层PDF生成

处理学生答卷图片，识别文字并生成可搜索的双层PDF。

```bash
python parse_student_answers.py
```

**功能**：
- 智能检测图片方向（支持EXIF和Tesseract）
- 自动旋转图片到正确方向
- OCR识别文本内容
- 生成可搜索的双层PDF

**输出文件**：
- `student_answers_output.md` - 识别的文本内容
- `student_answers_searchable.pdf` - 双层PDF（底层图像+透明文本层）
- `images/` - 处理后的图片文件

**便捷函数使用**：

```python
from parse_student_answers import generate_pdf_from_images

# 从文件列表生成双层PDF
generate_pdf_from_images([
    'image1.jpg',
    'image2.jpg'
], 'output.pdf')

# 从已处理的images文件夹生成
import glob
processed_images = sorted(glob.glob('images/*.png'))
generate_pdf_from_images(processed_images, 'output.pdf')
```

### 5. 双层PDF生成测试

快速测试双层PDF生成功能，无需重新运行OCR。

```bash
python test_double_layer_pdf.py
```

**功能**：
- 从已处理的`images`文件夹生成PDF
- 从原始图片生成PDF（需要重新OCR）
- 智能选择最佳旋转角度的图片
- 手动选择特定图片

### 6. PaddleOCR批量测试

批量处理多张图片并保存识别结果。

```bash
python test_paddleocr_debug.py
```

**功能**：
- 自动查找测试图片（支持jpg/png/jpeg）
- 批量OCR识别
- 保存所有结果到文本文件
- 显示统计信息

**输出文件**：
- `paddleocr_batch_result.txt` - 批量识别结果

## ⚙️ 配置说明

### pdf_ocr_with_boxes_v2.py 配置

主要配置项在 `Config` 类中（第38-163行）：

**OCR识别配置**：
```python
LANG = 'japan'  # 识别语言: 'japan'/'ch'/'en'
OCR_MIN_CONFIDENCE = 0.5  # 最低置信度阈值 (0.0-1.0)
DPI = 180  # PDF转图片DPI（日语推荐200-300）
```

**文本框标注配置**：
```python
SHOW_TEXT_LABEL = False  # 是否显示文本标签
BOX_WIDTH = 1  # 边框线条宽度（像素）
BOX_OPACITY = 180  # 边框不透明度（0-255）
```

**精度调整参数（重要）**：
```python
OCR_DET_DB_THRESH = 0.3  # 二值化阈值（0.2-0.5）
OCR_DET_DB_BOX_THRESH = 0.5  # 文本框置信度阈值（0.4-0.7）
OCR_DET_DB_UNCLIP_RATIO = 1.2  # 文本框扩张比例（1.2-2.0）
```

**坐标偏移调整**：
```python
BOX_OFFSET_X = 0  # X轴偏移量（像素）
BOX_OFFSET_Y = 0  # Y轴偏移量（像素）
```

## 🔧 Intel Arc GPU 支持

如果使用Intel Arc GPU，需要先修复DeepSeek-OCR模型：

```bash
python fix_xpu_for_arc.py
```

**功能**：
- 自动将 `.cuda()` 替换为 `.to(device)`
- 修复 `torch.autocast` 调用
- 添加设备检测代码
- 自动备份原文件

## 📁 项目结构

```
pdf/
├── pdf_ocr_with_boxes_v2.py      # PaddleOCR文本框可视化工具
├── pdf_to_markdown.py             # DeepSeek-OCR转Markdown工具
├── ocr_quality_analyzer.py        # OCR质量分析工具
├── parse_student_answers.py       # 学生答卷处理&双层PDF生成
├── fix_xpu_for_arc.py            # Intel XPU支持修复脚本
├── test_double_layer_pdf.py      # 双层PDF测试脚本
├── test_paddleocr_debug.py       # PaddleOCR批量测试脚本
├── requirements.txt               # Python依赖
├── README.md                      # 项目文档
├── japanese_test.pdf              # 测试PDF文件
├── images/                        # 处理后的图片输出文件夹
└── ocr_boxes_output/             # OCR标注结果输出文件夹
```

## 📝 使用示例

### 示例1：处理日语PDF

```bash
# 1. 转换PDF为标注图片和文本
python pdf_ocr_with_boxes_v2.py japanese_test.pdf --lang japan --dpi 300

# 2. 分析识别质量
python ocr_quality_analyzer.py ocr_boxes_output/japanese_test_page_1_annotated.png --lang japan

# 3. 转换为Markdown格式
python pdf_to_markdown.py
```

### 示例2：处理学生答卷

```bash
# 1. OCR识别并生成双层PDF
python parse_student_answers.py

# 2. 或快速测试已处理的图片
python test_double_layer_pdf.py
```

### 示例3：批量处理图片

```bash
# 1. 将所有图片放在images文件夹
# 2. 运行批量测试
python test_paddleocr_debug.py
```

## 🎯 使用场景

1. **文档数字化**：将纸质文档扫描后进行OCR识别和归档
2. **PDF文本提取**：从PDF中提取文本并转换为可编辑格式
3. **考卷识别**：批量处理学生答卷并生成可搜索PDF
4. **质量检查**：分析OCR识别质量，优化参数配置
5. **多语言文档处理**：支持日语、中文、英语文档识别

## 🐛 常见问题

### 1. 文本框位置不准确

调整以下参数：
- 提高 `DPI` 到 200-300
- 调整 `OCR_DET_DB_UNCLIP_RATIO` (1.2-2.0)
- 使用 `--offset-x` 和 `--offset-y` 微调位置

### 2. 识别率低

尝试：
- 确认 `LANG` 参数正确
- 提高图片分辨率（`DPI`）
- 降低 `OCR_DET_DB_THRESH` 提高敏感度
- 使用 `ocr_quality_analyzer.py` 分析问题

### 3. Intel XPU 不可用

运行修复脚本：
```bash
python fix_xpu_for_arc.py
```

### 4. 内存不足

- 降低 `DPI` 值
- 设置 `base_size` 和 `image_size` 为较小值
- 使用CPU模式而非GPU

## 📄 输出文件说明

| 文件类型 | 说明 |
|---------|------|
| `*_annotated.pdf` | 标注了文本框的PDF文件 |
| `*_ocr_text.md` | 识别的纯文本内容（Markdown格式） |
| `*_page_*_annotated.png` | 每页的标注图片 |
| `*_quality_analysis.png` | 质量分析可视化图片 |
| `*_searchable.pdf` | 双层PDF（可搜索文本） |

## 🔬 技术细节

### PaddleOCR

- **版本兼容性**：支持新旧版本API（predict/ocr）
- **设备支持**：CPU、CUDA GPU
- **语言支持**：中文、英文、日文等80+种语言
- **精度调整**：支持多种检测参数微调

### DeepSeek-OCR

- **模型**：`deepseek-ai/DeepSeek-OCR`
- **设备支持**：CPU、XPU (Intel Arc)、CUDA GPU
- **优化**：支持IPEX优化和bfloat16精度
- **功能**：支持grounding模式（提取文本位置）

### 双层PDF技术

- **底层**：原始图像（保持清晰度）
- **上层**：透明文本层（renderMode=3，不可见但可搜索）
- **字体自适应**：根据文本框大小自动调整字体
- **坐标转换**：自动处理PDF坐标系转换

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📜 许可证

本项目使用的第三方库遵循其各自的许可证：
- PaddleOCR: Apache 2.0
- DeepSeek-OCR: MIT
- PyTorch: BSD

## ⚡ 性能提示

1. **使用GPU加速**：如果有支持的GPU，性能可提升5-10倍
2. **合理设置DPI**：150适合快速预览，300适合高质量输出
3. **批量处理**：使用批量脚本处理多个文件更高效
4. **图片预处理**：确保图片方向正确可减少处理时间

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件反馈

---

**更新日期**: 2025-11-03

**版本**: 2.0

