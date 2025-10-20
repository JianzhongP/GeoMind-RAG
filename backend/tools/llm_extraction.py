# 阿秋
# 2025/09/18 14:29
import io
import base64
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from PIL import Image
import aiohttp
import json
from pathlib import Path
import os
from urllib.parse import urlparse

# ============ 配置部分 ============
load_dotenv(f"D:\Pycharm_project\my_multimodal_RAG\\backend\.env", override=True)
API_KEY = os.getenv("QWEN_API_KEY")
MODEL_NAME = os.getenv("QWEN_MULTIMODAL_MODAL_NAME")
MODEL_URL = os.getenv("QWEN_BASE_URL")

# 批处理配置
PAGES_PER_REQUEST = 2
CONCURRENT_REQUESTS = 1


@dataclass
class ExtractionResult:
    """提取结果数据类"""
    filename: str = ""  # 添加默认值
    markdown_content: str = ""
    tables: List[Dict[str, Any]] = None
    formulas: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    token_usage: Dict[str, int] = None
    time_cost: Dict[str, float] = None
    page_images: List[Image.Image] = None
    per_page_results: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.tables is None:
            self.tables = []
        if self.formulas is None:
            self.formulas = []
        if self.metadata is None:
            self.metadata = {}
        if self.token_usage is None:
            self.token_usage = {}
        if self.time_cost is None:
            self.time_cost = {}
        if self.page_images is None:
            self.page_images = []
        if self.per_page_results is None:
            self.per_page_results = []


class PDFMultimodalExtractor:
    """PDF多模态信息抽取器"""

    def __init__(
            self,
            model_url: str = MODEL_URL,
            api_key: str = API_KEY,
            model_name: str = MODEL_NAME,
            pages_per_request: int = PAGES_PER_REQUEST
    ):
        self.model_url = model_url
        self.api_key = api_key
        self.model_name = model_name
        self.dpi = 100
        self.pages_per_request = pages_per_request

        # 检测API类型
        self.api_type = self._detect_api_type()
        print(f"✓ 检测到API类型: {self.api_type}")

        # 如果使用OpenAI SDK，初始化客户端
        if self.api_type == "openai_sdk":
            from openai import AsyncOpenAI
            if "/chat/completions" in model_url:
                base_url = model_url.replace("/chat/completions", "")
            else:
                base_url = model_url
            self.openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            print(f"  使用OpenAI SDK (base_url: {base_url})")
        else:
            self.openai_client = None
            print(f"  使用HTTP客户端 (url: {model_url})")

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        self.pdf_convert_time = 0
        self.api_call_time = 0
        self.total_time = 0

    def _detect_api_type(self) -> str:
        """检测API类型：openai_sdk, qwen, claude"""
        url_lower = self.model_url.lower()

        # 通义千问/DashScope
        if "dashscope" in url_lower or "aliyun" in url_lower:
            return "qwen"

        # Claude API
        if "anthropic" in url_lower or "claude" in url_lower:
            return "claude"

        # OpenAI官方或使用OpenAI SDK
        if any(x in url_lower for x in ["openai.com", "api.openai.com", "oai.azure.com"]):
            return "openai_sdk"

        # 如果模型名包含gpt，使用OpenAI SDK（兼容第三方代理）
        if "gpt" in self.model_name.lower():
            return "openai_sdk"

        # 默认使用Claude格式
        return "claude"

    def _get_request_url(self) -> str:
        """返回实际要POST的完整URL"""
        url = self.model_url
        lower = url.lower()

        # 已包含完整端点，直接返回
        if any(x in lower for x in ["/chat/completions", "/v1/completions", "completions"]):
            return url

        # DashScope兼容模式URL
        if "dashscope" in lower and "compatible-mode" in lower:
            if not url.endswith("/chat/completions"):
                return url.rstrip('/') + '/chat/completions'
            return url

        # 类似 https://host/.../v1 格式，补上 chat/completions
        parsed = urlparse(url)
        path = parsed.path or ""
        if path.rstrip('/') == '/v1':
            return url.rstrip('/') + '/chat/completions'

        return url

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """将PDF转换为图片列表"""
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        images = convert_from_bytes(pdf_content, dpi=self.dpi)
        return images

    def image_to_base64(self, image: Image.Image, max_size: int = 2000) -> str:
        """将PIL Image转换为base64字符串，并压缩图片"""
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    async def call_multimodal_api_openai_sdk(
            self,
            image_base64_list: List[str],
            page_nums: List[int],
            total_pages: int
    ) -> Dict[str, Any]:
        """使用OpenAI官方SDK调用API（适用于GPT系列）"""
        import time
        start_time = time.time()

        page_range = f"{page_nums[0]}-{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])

        prompt = self._get_extraction_prompt(page_range, total_pages, page_nums)

        # 构建消息内容 - OpenAI格式
        content_items = [{"type": "text", "text": prompt}]

        for img_base64 in image_base64_list:
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的PDF文档信息提取助手。你必须严格按照用户要求的JSON格式返回结果，不要添加任何解释性文字或markdown代码块标记。直接返回可解析的JSON对象。"
            },
            {"role": "user", "content": content_items}
        ]

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # 提取Token使用信息
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens

                print(f"  第{page_range}页 Token: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")

            content = response.choices[0].message.content

            api_time = time.time() - start_time
            print(f"  第{page_range}页 耗时: {api_time:.2f}秒")

            return self._parse_response_content(content, page_nums)

        except Exception as e:
            print(f"❌ OpenAI SDK调用失败: {type(e).__name__}: {e}")
            raise

    async def call_multimodal_api_qwen(
            self,
            image_base64_list: List[str],
            page_nums: List[int],
            total_pages: int
    ) -> Dict[str, Any]:
        """调用通义千问API（DashScope兼容模式）"""
        import time
        start_time = time.time()

        page_range = f"{page_nums[0]}-{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])

        prompt = self._get_extraction_prompt(page_range, total_pages, page_nums)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建消息内容 - 通义千问格式
        content_items = [{"type": "text", "text": prompt}]

        # 关键修复：通义千问使用 image_url 类型，而不是 image 类型
        for img_base64 in image_base64_list:
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的PDF文档信息提取助手。你必须严格按照用户要求的JSON格式返回结果，不要添加任何解释性文字或markdown代码块标记。"
                },
                {"role": "user", "content": content_items}
            ],
            "max_tokens": 4096,
            "temperature": 0.1
        }

        payload_size = len(json.dumps(payload))
        print(f"  请求体大小: {payload_size / 1024 / 1024:.2f} MB")

        request_url = self._get_request_url()
        print(f"  请求URL: {request_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        request_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    response_text = await response.text()

                    if response.status != 200:
                        display_text = response_text[:1000]
                        print(f"❌ API错误: {response.status}")
                        print(f"错误详情: {display_text}")
                        raise Exception(f"API调用失败: {response.status} - {display_text}")

                    result = await response.json()

                    # 提取Token使用信息
                    if 'usage' in result:
                        usage = result['usage']
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        completion_tokens = usage.get('completion_tokens', 0)
                        total_tokens = usage.get('total_tokens', 0)

                        self.total_prompt_tokens += prompt_tokens
                        self.total_completion_tokens += completion_tokens
                        self.total_tokens += total_tokens

                        print(
                            f"  第{page_range}页 Token: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")

                    # 提取响应内容
                    content = result['choices'][0]['message']['content']

                    api_time = time.time() - start_time
                    print(f"  第{page_range}页 耗时: {api_time:.2f}秒")

                    return self._parse_response_content(content, page_nums)

        except asyncio.TimeoutError:
            print(f"❌ 请求超时（页面 {page_range}）")
            raise
        except Exception as e:
            print(f"❌ 通义千问API调用异常: {type(e).__name__}: {e}")
            raise

    async def call_multimodal_api_claude(
            self,
            image_base64_list: List[str],
            page_nums: List[int],
            total_pages: int
    ) -> Dict[str, Any]:
        """调用Claude API"""
        import time
        start_time = time.time()

        page_range = f"{page_nums[0]}-{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])

        prompt = self._get_extraction_prompt(page_range, total_pages, page_nums)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        # 构建消息内容 - Claude格式
        claude_content = []
        for img_base64 in image_base64_list:
            claude_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_base64
                }
            })
        claude_content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model_name,
            "max_tokens": 4096,
            "temperature": 0.1,
            "system": "你是一个专业的PDF文档信息提取助手。你必须严格按照用户要求的JSON格式返回结果。",
            "messages": [{"role": "user", "content": claude_content}]
        }

        payload_size = len(json.dumps(payload))
        print(f"  请求体大小: {payload_size / 1024 / 1024:.2f} MB")
        print(f"  请求URL: {self.model_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.model_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    response_text = await response.text()

                    if response.status != 200:
                        display_text = response_text[:1000]
                        print(f"❌ API错误: {response.status}")
                        print(f"错误详情: {display_text}")
                        raise Exception(f"API调用失败: {response.status} - {display_text}")

                    result = await response.json()

                    # 提取Token使用信息
                    if 'usage' in result:
                        usage = result['usage']
                        prompt_tokens = usage.get('input_tokens', 0)
                        completion_tokens = usage.get('output_tokens', 0)
                        total_tokens = prompt_tokens + completion_tokens

                        self.total_prompt_tokens += prompt_tokens
                        self.total_completion_tokens += completion_tokens
                        self.total_tokens += total_tokens

                        print(
                            f"  第{page_range}页 Token: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")

                    # 提取响应内容
                    content = result['content'][0]['text']

                    api_time = time.time() - start_time
                    print(f"  第{page_range}页 耗时: {api_time:.2f}秒")

                    return self._parse_response_content(content, page_nums)

        except asyncio.TimeoutError:
            print(f"❌ 请求超时（页面 {page_range}）")
            raise
        except Exception as e:
            print(f"❌ Claude API调用异常: {type(e).__name__}: {e}")
            raise

    def _get_extraction_prompt(self, page_range: str, total_pages: int, page_nums: List[int]) -> str:
        """生成提取指令的prompt"""

        # 构建图片和页码的对应说明
        if len(page_nums) == 1:
            page_mapping = f"这张图片是第{page_nums[0]}页"
        else:
            mappings = [f"第{i + 1}张图片是第{page_num}页" for i, page_num in enumerate(page_nums)]
            page_mapping = "，".join(mappings)

        # 构建每页的ID前缀说明
        id_examples = []
        for page_num in page_nums:
            id_examples.append(f"第{page_num}页的元素ID格式：表格{page_num}-1、公式{page_num}-1、图片{page_num}-1")
        id_format_desc = "；".join(id_examples)

        return f"""
        【重要】请直接分析图片内容并返回JSON，不要说"我无法提取"或给出任何解释。

        **图片和页码对应关系（非常重要）：**
        {page_mapping}
        
        分析这些PDF页面（第{page_range}页，共{total_pages}页）：
        
        **核心要求：**
        - **每页markdown开头必须是标题 `{{第X页}}`**（X是该页的实际页码，用于验证）
        - **所有内容（图片、表格、公式）必须直接嵌入到markdown中原有位置**
        - 图片使用markdown语法：`![图片描述](placeholder)`，描述要详细
        - 表格使用markdown表格语法直接嵌入
        - 公式使用LaTeX语法（行内用$...$，独立用$$...$$）直接嵌入
        - 保持内容的原始顺序和位置关系
        
        1. **Markdown内容（最重要）**：
           - **每页开头必须是 `## 第X页`**（用于页面识别和验证）
           - 识别所有标题层级（# ## ###）
           - 保持段落结构和格式
           - 保留列表、引用
           - **图片必须在原位置插入**：用`![详细描述](placeholder)`表示
           - **表格必须在原位置插入**：使用markdown表格语法
           - **公式必须在原位置插入**：使用LaTeX语法
           - 忽略页眉和页脚内容
           - 如果分析多页，页间用 `---` 分隔
        
        2. **元素ID命名规则（非常重要）**：
           {id_format_desc}
           - 同一页内的元素按出现顺序编号：-1、-2、-3...
        
        3. **表格提取（用于元数据统计）**：
           - 提取所有表格的结构化数据
           - ID格式必须是：表格{{页码}}-{{序号}}
           - 标注所在页码（page字段必须准确）
        
        4. **公式提取（用于元数据统计）**：
           - 提取所有数学公式
           - ID格式必须是：公式{{页码}}-{{序号}}
           - 使用LaTeX格式
           - 标注所在页码（page字段必须准确）
        
        5. **图片描述（用于元数据统计）**：
           - 描述所有非表格、非公式的图像内容
           - ID格式必须是：图片{{页码}}-{{序号}}
           - 标注所在页码（page字段必须准确）
           - 根据图片类型提供不同详细程度的描述：
             - 数据图（柱状图、折线图、饼图等）：需详细描述数据趋势、关键数值、坐标轴含义等
             - 流程图/架构图：需描述各组成部分及其关系
             - 照片：描述主要对象、场景和内容
             - 示意图：描述所表达的概念或原理
        
        **输出格式要求（必须严格遵守）：**
        - 直接返回纯JSON对象
        - 不要用```json或```包裹
        - 不要添加任何解释文字
        - page_num字段必须准确填写页码
        
        **JSON结构：**
        {{
            "pages": [
                {{
                    "page_num": {page_nums[0]},
                    "markdown": "## 第{page_nums[0]}页\\n\\n该页完整内容...",
                    "page_title": "主标题或空字符串"
                }}
            ],
            "tables": [
                {{
                    "page": {page_nums[0]},
                    "id": "表格{page_nums[0]}-1",
                    "caption": "表格标题",
                    "content": "markdown表格",
                    "data": [["单元格"]]
                }}
            ],
            "formulas": [
                {{
                    "page": {page_nums[0]},
                    "id": "公式{page_nums[0]}-1",
                    "latex": "LaTeX公式",
                    "type": "inline或display",
                    "context": "前后文本"
                }}
            ],
            "images": [
                {{
                    "page": {page_nums[0]},
                    "id": "图片{page_nums[0]}-1",
                    "description": "根据图片类型提供相应详细程度的描述",
                    "type": "chart/graph/photo/diagram",
                    "context": "上下文"
                }}
            ]
        }}
        
        现在请分析图片并直接返回上述JSON结构："""

    def _parse_response_content(self, content: str, page_nums: List[int]) -> Dict[str, Any]:
        """解析API响应内容 - 增强版"""
        try:
            # 清理内容
            content = content.strip()

            # 移除 markdown 代码块标记
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # 🔥 新增：如果内容中包含非JSON前缀，尝试提取JSON部分
            # 查找第一个 { 和最后一个 }
            first_brace = content.find('{')
            last_brace = content.rfind('}')

            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                # 提取JSON部分
                json_content = content[first_brace:last_brace + 1]

                # 如果前面有非JSON内容，打印警告
                if first_brace > 0:
                    prefix = content[:first_brace].strip()
                    if prefix:
                        print(f"  ⚠️  检测到JSON前有额外内容（{len(prefix)}字符），已自动移除")
                        print(f"     前缀预览: {prefix[:100]}...")

                content = json_content

            parsed = json.loads(content)

            # 验证和修正page_num
            pages_data = parsed.get('pages', [])
            len_pages_data = len(pages_data)
            print(f"  📋 收到 {len_pages_data} 个页面数据，预期页码: {page_nums}")
            if len_pages_data == 0:
                pass

            for i, page_data in enumerate(pages_data):
                returned_page_num = page_data.get('page_num')
                expected_page_num = page_nums[i] if i < len(page_nums) else None

                if returned_page_num != expected_page_num:
                    print(
                        f"  ⚠️  页面{i + 1}: 大模型返回page_num={returned_page_num}, 预期={expected_page_num}, 已自动修正")
                    page_data['page_num'] = expected_page_num
                else:
                    print(f"  ✓ 页面{i + 1}: page_num={returned_page_num} 匹配正确")

                # 二次验证：检查markdown开头是否有页码标题
                markdown = page_data.get('markdown', '')
                if markdown.strip().startswith('## 第'):
                    # 尝试从markdown中提取页码
                    import re
                    match = re.match(r'##\s*第(\d+)页', markdown.strip())
                    if match:
                        markdown_page_num = int(match.group(1))
                        if markdown_page_num != expected_page_num:
                            print(f"  ⚠️  Markdown标题显示第{markdown_page_num}页，但预期是第{expected_page_num}页")

            # 验证tables、formulas、images的page字段
            for table in parsed.get('tables', []):
                if table.get('page') not in page_nums:
                    print(f"  ⚠️  表格 {table.get('id')} 的page={table.get('page')}不在预期页码中")

            for formula in parsed.get('formulas', []):
                if formula.get('page') not in page_nums:
                    print(f"  ⚠️  公式 {formula.get('id')} 的page={formula.get('page')}不在预期页码中")

            for image in parsed.get('images', []):
                if image.get('page') not in page_nums:
                    print(f"  ⚠️  图片 {image.get('id')} 的page={image.get('page')}不在预期页码中")

            # 🔥 关键修复：添加 per_page_results 字段
            return {
                'per_page_results': pages_data,  # 添加这一行
                'tables': parsed.get('tables', []),
                'formulas': parsed.get('formulas', []),
                'images': parsed.get('images', [])
            }

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON解析失败: {e}")
            print(f"原始内容前200字符: {content[:200]}...")

            # 检查是否是模型拒绝提取的回复
            refusal_keywords = ["unable to", "cannot", "can't", "sorry", "i'm not able"]
            if any(keyword in content.lower() for keyword in refusal_keywords):
                print(f"⚠️  模型拒绝提取内容，页面 {page_nums} 将标记为处理失败")
                return {
                    "per_page_results": [{  # 修改这里
                        "page_num": num,
                        "markdown": f"## 第{num}页\n**提取失败：模型拒绝处理此页面**\n{content[:500]}",
                        "page_title": f"第{num}页（提取失败）"
                    } for num in page_nums],
                    "tables": [],
                    "formulas": [],
                    "images": []
                }

            # 如果不是拒绝，尝试将原始内容作为markdown
            print(f"⚠️  将原始响应作为markdown内容保存")
            return {
                "per_page_results": [{  # 修改这里
                    "page_num": num,
                    "markdown": f"## 第{num}页\n{content}",
                    "page_title": f"第{num}页"
                } for num in page_nums],
                "tables": [],
                "formulas": [],
                "images": []
            }

    async def call_multimodal_api(
            self,
            image_base64_list: List[str],
            page_nums: List[int],
            total_pages: int
    ) -> Dict[str, Any]:
        """调用多模态API的统一入口 - 根据API类型路由"""
        if self.api_type == "openai_sdk":
            return await self.call_multimodal_api_openai_sdk(
                image_base64_list, page_nums, total_pages
            )
        elif self.api_type == "qwen":
            return await self.call_multimodal_api_qwen(
                image_base64_list, page_nums, total_pages
            )
        elif self.api_type == "claude":
            return await self.call_multimodal_api_claude(
                image_base64_list, page_nums, total_pages
            )
        else:
            raise ValueError(f"不支持的API类型: {self.api_type}")

    async def extract_from_pdf(self, pdf_path: str, original_filename: Optional[str] = None) -> ExtractionResult:
        """从PDF文件中提取完整信息"""
        import time
        overall_start = time.time()

        # 优先使用原始文件名，否则从路径提取
        filename = original_filename or Path(pdf_path).name
        print(f"filename: {filename}")

        print(f"开始处理PDF: {pdf_path}")
        print("=" * 60)

        # PDF转图片
        convert_start = time.time()
        images = self.pdf_to_images(pdf_path)
        self.pdf_convert_time = time.time() - convert_start
        total_pages = len(images)
        print(f"✓ PDF转换完成: {total_pages} 页 (耗时: {self.pdf_convert_time:.2f}秒)")

        # 批量处理页面
        per_page_results = []
        all_tables = []
        all_formulas = []

        for i in range(0, total_pages, self.pages_per_request):
            batch_images = images[i:i + self.pages_per_request]
            batch_page_nums = list(range(i + 1, min(i + 1 + self.pages_per_request, total_pages + 1)))

            image_base64_list = [self.image_to_base64(img) for img in batch_images]

            result = await self.call_multimodal_api(
                image_base64_list=image_base64_list,
                page_nums=batch_page_nums,
                total_pages=total_pages
            )

            per_page_results.extend(result['per_page_results'])
            all_tables.extend(result.get('tables', []))
            all_formulas.extend(result.get('formulas', []))

        # 组装最终markdown
        final_markdown = ""
        for page_result in per_page_results:
            final_markdown += page_result.get('markdown', '') + "\n\n"

        # 统计信息
        metadata = {
            "total_pages": total_pages,
            "total_tables": len(all_tables),
            "total_formulas": len(all_formulas),
            "model": self.model_name
        }

        token_usage = {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens
        }

        self.total_time = time.time() - overall_start
        time_cost = {
            "pdf_convert_time": round(self.pdf_convert_time, 2),
            "api_call_time": round(self.api_call_time, 2),
            "total_time": round(self.total_time, 2)
        }

        print("\n" + "=" * 60)
        print("✓ 提取完成")
        print(f"  总页数: {total_pages}")
        print(f"  表格数: {len(all_tables)}")
        print(f"  公式数: {len(all_formulas)}")
        print(
            f"  Token使用: {self.total_tokens:,} (提示: {self.total_prompt_tokens:,}, 完成: {self.total_completion_tokens:,})")
        print(
            f"  耗时: PDF转换 {self.pdf_convert_time:.2f}s + API调用 {self.api_call_time:.2f}s = 总计 {self.total_time:.2f}s")
        print("=" * 60 + "\n")

        return ExtractionResult(
            filename=filename,
            markdown_content=final_markdown,
            tables=all_tables,
            formulas=all_formulas,
            metadata=metadata,
            token_usage=token_usage,
            time_cost=time_cost,
            page_images=images,
            per_page_results=per_page_results
        )

    def save_results(self, result: ExtractionResult, output_dir: str = "output"):
        """保存提取结果到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 创建images子目录
        images_path = output_path / "images"
        images_path.mkdir(exist_ok=True)

        # 1. 保存每页图片
        print(f"\n💾 保存页面图片...")
        for idx, image in enumerate(result.page_images):
            page_num = idx + 1
            image_filename = f"page_{page_num:03d}.jpg"
            image_path = images_path / image_filename

            # 转换并保存图片
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(image_path, format='JPEG', quality=95)
            print(f"  ✓ 第{page_num}页图片: {image_filename}")

        # 2. 保存每页识别结果
        print(f"\n💾 保存每页识别结果...")
        # 为每页结果添加图片文件名
        for page_result in result.per_page_results:
            page_num = page_result['page_num']
            page_result['image_file'] = f"images/page_{page_num:03d}.jpg"

        with open(output_path / "per_page_results.json", 'w', encoding='utf-8') as f:
            json.dump(result.per_page_results, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 每页结果: per_page_results.json")

        # 3. 保存完整markdown
        with open(output_path / "full_content.md", 'w', encoding='utf-8') as f:
            f.write(result.markdown_content)
        print(f"  ✓ 完整内容: full_content.md")

        # 4. 保存所有表格
        with open(output_path / "tables.json", 'w', encoding='utf-8') as f:
            json.dump(result.tables, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 所有表格: tables.json")

        # 5. 保存所有公式
        with open(output_path / "formulas.json", 'w', encoding='utf-8') as f:
            json.dump(result.formulas, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 所有公式: formulas.json")

        # 6. 保存元数据
        complete_metadata = {
            **result.metadata,
            "token_usage": result.token_usage,
            "time_cost": result.time_cost
        }
        with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(complete_metadata, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 元数据: metadata.json")

        print(f"\n✅ 所有结果已保存到: {output_path.absolute()}")
        print(f"   - 页面图片: {len(result.page_images)} 张")
        print(f"   - 识别结果: {len(result.per_page_results)} 页")


async def main(pdf_file, output_dir="output"):
    extractor = PDFMultimodalExtractor(
        model_url=MODEL_URL,
        api_key=API_KEY,
        model_name=MODEL_NAME
    )

    result = await extractor.extract_from_pdf(pdf_file)
    extractor.save_results(result, output_dir=output_dir)

    print("\n📝 内容预览 (前500字符):")
    print(result.markdown_content[:500])
    print("...")


if __name__ == "__main__":
    pdf_file = "D:\Pycharm_project\my_multimodal_RAG\遥感检测技术文档.pdf"
    output = "D:\Pycharm_project\my_multimodal_RAG\\backend\\tools"
    asyncio.run(main(pdf_file=pdf_file, output_dir=output))