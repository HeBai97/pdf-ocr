"""
基于硅基流动 API 的 PDF 文档解析器
支持使用 Qwen3-VL-32B 等多模态模型进行 OCR 和文档理解

官方文档: https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
"""

import os
import json
import base64
from io import BytesIO
from typing import Optional, Dict, List, Union
from pathlib import Path

from PIL import Image
from openai import OpenAI
import fitz  # PyMuPDF
import dotenv

dotenv.load_dotenv()    # 加载环境变量
api_key = os.getenv("SILICONFLOW_API_KEY")
print (api_key)

if not api_key:
    raise ValueError("请设置环境变量 SILICONFLOW_API_KEY")

class SiliconFlowPDFParser:
    """
    硅基流动 PDF 解析器
    使用硅基流动提供的 VLM API 进行文档解析
    """
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    }
    
    def __init__(
        self,
        api_key: str = api_key,
        model: str = "qwen3-vl-32b",
        api_base: str = "https://api.siliconflow.cn/v1",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 8192,
        dpi: int = 200,
    ):
        """
        初始化硅基流动 PDF 解析器
        
        Args:
            api_key: 硅基流动 API 密钥（也可通过环境变量 SILICONFLOW_API_KEY 设置）
            model: 模型名称简称，可选: qwen3-vl-32b
            api_base: API 基础 URL
            temperature: 生成温度
            top_p: nucleus sampling 参数
            max_tokens: 最大生成 token 数
            dpi: PDF 转图像的 DPI
        """
        self.api_key = api_key
        # 设置模型
        if model in self.SUPPORTED_MODELS:
            self.model_name = self.SUPPORTED_MODELS[model]
            self.model_short = model
        else:
            # 直接使用完整模型名
            self.model_name = model
            self.model_short = model
        
        self.api_base = api_base
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.dpi = dpi
        
        # 修复 SSL 证书文件路径问题
        # 如果 SSL_CERT_FILE 指向不存在的文件，先清除它
        if 'SSL_CERT_FILE' in os.environ:
            ssl_cert_file = os.environ['SSL_CERT_FILE']
            if not os.path.exists(ssl_cert_file):
                print(f"⚠ 检测到无效的 SSL_CERT_FILE: {ssl_cert_file}")
                print(f"  临时清除该环境变量以使用系统默认证书")
                del os.environ['SSL_CERT_FILE']
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        
        print(f"✓ 初始化硅基流动 PDF 解析器")
        print(f"  API: {self.api_base}")
        print(f"  模型: {self.model_name}")
    
    def _image_to_base64(self, image: Union[str, Image.Image]) -> str:
        """
        将图像转换为 base64 编码
        
        Args:
            image: 图像路径或 PIL Image 对象
            
        Returns:
            base64 编码的图像字符串（data URI 格式）
        """
        if isinstance(image, str):
            # 从文件路径读取
            with open(image, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 检测图像格式
            if image.lower().endswith('.png'):
                mime_type = "image/png"
            elif image.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/jpeg"
        else:
            # PIL Image 对象
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            mime_type = "image/jpeg"
        
        return f"data:{mime_type};base64,{image_base64}"
    
    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        将 PDF 转换为图像列表
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            图像列表
        """
        images = []
        pdf_document = fitz.open(pdf_path)
        
        print(f"📄 PDF 共 {len(pdf_document)} 页")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # 设置缩放以达到目标 DPI
            zoom = self.dpi / 72  # 默认 PDF DPI 是 72
            mat = fitz.Matrix(zoom, zoom)
            
            # 渲染页面为图像
            pix = page.get_pixmap(matrix=mat)
            
            # 转换为 PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            images.append(img)
            
            print(f"  ✓ 转换第 {page_num + 1} 页 ({img.width}x{img.height})")
        
        pdf_document.close()
        return images
    
    def inference(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        使用硅基流动 API 进行推理
        
        Args:
            image: 图像路径或 PIL Image 对象
            prompt: 文本提示
            temperature: 生成温度（可选，覆盖默认值）
            top_p: nucleus sampling 参数（可选，覆盖默认值）
            max_tokens: 最大生成 token 数（可选，覆盖默认值）
            
        Returns:
            模型生成的文本响应
        """
        # 使用提供的参数或默认值
        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p
        max_t = max_tokens if max_tokens is not None else self.max_tokens
        
        # 将图像转换为 base64
        image_base64 = self._image_to_base64(image)
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        try:
            # 调用 API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                top_p=tp,
                max_tokens=max_t,
            )
            
            # 提取响应文本
            result = response.choices[0].message.content
            
            # 打印 token 使用情况
            if hasattr(response, 'usage'):
                usage = response.usage
                print(f"  Token 使用: 输入={usage.prompt_tokens}, 输出={usage.completion_tokens}, 总计={usage.total_tokens}")
            
            return result
            
        except Exception as e:
            print(f"❌ API 请求失败: {e}")
            return None
    
    def parse_ocr(
        self,
        image: Union[str, Image.Image],
        language: str = "auto"
    ) -> str:
        """
        提取图像中的文本（OCR）
        
        Args:
            image: 图像路径或 PIL Image 对象
            language: 语言提示（auto, 中文, English, 日本語等）
            
        Returns:
            提取的文本内容
        """
        if language == "auto":
            prompt = "请提取这张图片中的所有文本内容，保持原始语言和格式。"
        elif language == "中文" or language == "zh":
            prompt = "请提取这张图片中的所有中文文本内容，保持原始格式。"
        elif language == "English" or language == "en":
            prompt = "Please extract all English text from this image, maintaining the original format."
        elif language == "日本語" or language == "ja":
            prompt = "この画像からすべての日本語テキストを抽出してください。元の形式を維持してください。"
        else:
            prompt = f"请提取这张图片中的所有{language}文本内容，保持原始格式。"
        
        response = self.inference(image, prompt)
        return response if response else ""
    
    def parse_document_layout(
        self,
        image: Union[str, Image.Image]
    ) -> Dict:
        """
        解析文档布局信息
        
        Args:
            image: 图像路径或 PIL Image 对象
            
        Returns:
            解析结果字典
        """
        prompt = """请分析这张文档图片，提取以下信息：

1. **文档布局**：识别文档中的所有元素，包括：
   - 标题 (Title)
   - 段落文本 (Text)
   - 列表项 (List-item)
   - 表格 (Table)
   - 公式 (Formula)
   - 图片 (Picture)
   - 页眉页脚 (Page-header, Page-footer)
   - 脚注 (Footnote)

2. **边界框位置**：每个元素的位置，格式为 [x1, y1, x2, y2]

3. **文本内容**：
   - 普通文本：使用 Markdown 格式
   - 表格：使用 Markdown 表格格式
   - 公式：使用 LaTeX 格式
   - 图片：省略文本字段

4. **阅读顺序**：按照人类阅读习惯排序所有元素

请以 JSON 格式输出结果：
```json
[
    {
        "category": "元素类别",
        "bbox": [x1, y1, x2, y2],
        "text": "文本内容"
    }
]
```

注意：输出必须是原始文本，不要翻译。"""
        
        response = self.inference(image, prompt, max_tokens=16384)
        
        if response is None:
            return {"error": "API 请求失败"}
        
        # 尝试解析 JSON
        try:
            # 提取 JSON 代码块
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            return {"layout": result, "raw_response": response}
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 解析失败，返回原始文本")
            return {"text": response, "error": str(e)}
    
    def parse_table(
        self,
        image: Union[str, Image.Image],
        format: str = "markdown"
    ) -> str:
        """
        解析表格内容
        
        Args:
            image: 图像路径或 PIL Image 对象
            format: 输出格式，'html' 或 'markdown'
            
        Returns:
            表格内容
        """
        if format.lower() == "html":
            prompt = "请将这张图片中的表格转换为 HTML 格式，保持原始内容不要翻译。"
        else:
            prompt = "请将这张图片中的表格转换为 Markdown 格式，保持原始内容不要翻译。"
        
        response = self.inference(image, prompt)
        return response if response else ""
    
    def parse_pdf(
        self,
        pdf_path: str,
        mode: str = "ocr",
        language: str = "auto",
        save_images: bool = False,
        output_dir: str = "./output"
    ) -> Dict:
        """
        解析 PDF 文件
        
        Args:
            pdf_path: PDF 文件路径
            mode: 解析模式，'ocr'（纯文本）或 'layout'（布局分析）
            language: 语言提示（仅 OCR 模式有效）
            save_images: 是否保存转换的图像
            output_dir: 输出目录
            
        Returns:
            解析结果字典
        """
        print(f"\n📄 开始解析 PDF: {pdf_path}")
        print(f"   模式: {mode}")
        print(f"   语言: {language}")
        
        # 转换 PDF 为图像
        images = self._pdf_to_images(pdf_path)
        
        # 准备输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_name = Path(pdf_path).stem
        
        # 保存图像（如果需要）
        if save_images:
            images_dir = output_path / f"{pdf_name}_images"
            images_dir.mkdir(exist_ok=True)
            for i, img in enumerate(images):
                img_path = images_dir / f"page_{i+1}.jpg"
                img.save(img_path, "JPEG", quality=95)
            print(f"✓ 图像已保存到: {images_dir}")
        
        # 逐页解析
        results = []
        all_text = []
        
        for i, image in enumerate(images):
            page_num = i + 1
            print(f"\n处理第 {page_num}/{len(images)} 页...")
            
            if mode == "ocr":
                # OCR 模式
                text = self.parse_ocr(image, language=language)
                results.append({
                    "page": page_num,
                    "text": text
                })
                all_text.append(f"\n{'='*60}\n第 {page_num} 页\n{'='*60}\n\n{text}")
                
            elif mode == "layout":
                # 布局分析模式
                layout_result = self.parse_document_layout(image)
                results.append({
                    "page": page_num,
                    "layout": layout_result
                })
                
                # 提取文本用于合并输出
                if "layout" in layout_result:
                    page_text = "\n".join([
                        item.get("text", "") 
                        for item in layout_result["layout"] 
                        if "text" in item
                    ])
                    all_text.append(f"\n{'='*60}\n第 {page_num} 页\n{'='*60}\n\n{page_text}")
                elif "text" in layout_result:
                    all_text.append(f"\n{'='*60}\n第 {page_num} 页\n{'='*60}\n\n{layout_result['text']}")
        
        # 保存结果
        result_dict = {
            "pdf_name": pdf_name,
            "total_pages": len(images),
            "mode": mode,
            "language": language,
            "pages": results
        }
        
        # 保存 JSON
        json_path = output_path / f"{pdf_name}_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f"\n✓ JSON 结果已保存到: {json_path}")
        
        # 保存合并的文本
        text_path = output_path / f"{pdf_name}_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_text))
        print(f"✓ 文本结果已保存到: {text_path}")
        
        # 保存 Markdown（如果是布局模式）
        if mode == "layout":
            md_path = output_path / f"{pdf_name}_layout.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {pdf_name}\n\n")
                f.write("\n".join(all_text))
            print(f"✓ Markdown 结果已保存到: {md_path}")
        
        print(f"\n✅ PDF 解析完成！")
        return result_dict


def main():
    """
    命令行入口
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="硅基流动 PDF 文档解析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # OCR 模式（提取纯文本）
  python siliconflow_pdf_parser.py document.pdf --mode ocr --language auto
  
  # 布局分析模式
  python siliconflow_pdf_parser.py document.pdf --mode layout
  
  # 指定模型和 API 密钥
  python siliconflow_pdf_parser.py document.pdf \\
      --api_key YOUR_API_KEY \\
      --model qwen3-vl-32b \\
      --mode ocr
  
  # 保存中间图像
  python siliconflow_pdf_parser.py document.pdf \\
      --mode layout \\
      --save_images \\
      --output_dir ./my_output

获取 API 密钥: https://cloud.siliconflow.cn/account/ak
        """
    )
    
    parser.add_argument(
        "pdf_path",
        type=str,
        help="PDF 文件路径"
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="硅基流动 API 密钥（也可通过环境变量 SILICONFLOW_API_KEY 设置）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-32b",
        choices=["qwen3-vl-32b"],
        help="使用的模型"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="ocr",
        choices=["ocr", "layout"],
        help="解析模式: ocr（纯文本） 或 layout（布局分析）"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="语言提示（auto, 中文, English, 日本語等）"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF 转图像的 DPI（默认: 200）"
    )
    
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="保存 PDF 转换的图像"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ocr_boxes_output",
        help="输出目录（默认: ./output）"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.pdf_path).exists():
        print(f"❌ 文件不存在: {args.pdf_path}")
        return
    
    try:
        # 初始化解析器
        # 如果命令行没有提供 api_key，使用环境变量中的值
        final_api_key = args.api_key if args.api_key else api_key
        pdf_parser = SiliconFlowPDFParser(
            api_key=final_api_key,
            model=args.model,
            dpi=args.dpi,
        )
        
        # 解析 PDF
        result = pdf_parser.parse_pdf(
            pdf_path=args.pdf_path,
            mode=args.mode,
            language=args.language,
            save_images=args.save_images,
            output_dir=args.output_dir,
        )
        
        print(f"\n{'='*60}")
        print("解析统计:")
        print(f"  PDF 文件: {result['pdf_name']}")
        print(f"  总页数: {result['total_pages']}")
        print(f"  模式: {result['mode']}")
        print(f"  输出目录: {args.output_dir}")
        print(f"{'='*60}\n")
        
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 获取 API 密钥:")
        print("   1. 访问 https://cloud.siliconflow.cn/account/ak")
        print("   2. 注册/登录账户")
        print("   3. 创建 API 密钥")
        print("   4. 设置环境变量: export SILICONFLOW_API_KEY='your_key'")
        print("      或使用参数: --api_key YOUR_KEY")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

