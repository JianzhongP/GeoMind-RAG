# 阿秋
# 2025/09/28 19:21
"""
简化版图像识别模块 - 第一步实现
功能：加载图片 + VLM识别 + 根据用户问题回复

支持三种场景：
1. 遥感图像处理结果解读（结构、尺寸、参数）
2. 技术流程图理解（架构、流程图语义）
3. 工业技术档案识别（研究区域、时间）
"""

import os
from dotenv import load_dotenv
load_dotenv(r"/backend/.env", override=True)

import io
import base64
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import aiohttp
import json
import requests
from urllib.parse import urlparse

@dataclass
class AnalysisResult:
    """图像分析结果"""
    image_type: str  # result_image / architecture / technical_doc
    question: str  # 用户问题
    answer: str  # VLM回答
    extracted_info: Dict[str, Any]  # 提取的结构化信息
    raw_response: str  # 原始响应
    token_usage: Dict[str, int]  # Token使用统计
    time_cost: float  # 耗时


class ImageType:
    """图像类型枚举"""
    Classification = "Classification"  # 分类图
    Detection = "Detection"  # 目标检测图
    Technology_map = "Technology_map"  # 研发架构图/流程图
    Technical_doc = "Technical_doc"  # 工业技术档案/工艺文件


class SimpleVLMAnalyzer:
    """简化版VLM图像分析器"""

    PROMPTS = {
        ImageType.Classification:
            """
            你是一位专业的遥感影像与地物分类专家。请仔细分析这张遥感**分类图**（和可选的底图影像 / 元数据）。

            **用户问题：**
            {question}
            
            **输入说明：**
                        - 图片/文件：{{map_image}}（栅格分类图）；可附带基础影像 {{base_image}}、时间戳 {{acq_date}}、分辨率 {{gsd}}、投影/坐标系 {{crs}}、图例（class->颜色映射）{{legend}}。
                        - 可选元数据：研究区边界、矢量样本、土地利用类别代码表。
            针对不同图纸类型的提示词模板
            
            **分析要求：**
            1. **类别识别与映射**：列出分类图中的所有类别及其标签（用图例或像素颜色反推），并给出每类的像素数与面积（m² / km²）。
            2. **空间分布描述**：描述每个类别的主要聚集区、典型位置（用地名或坐标），若可能列出每类的质心经纬度。
            3. **精度与不确定性**：给出可推断的预测不确定或噪声区域。
            4. **时序/变化（可选）**：若提供多时相，说明主要变化趋势（增减百分比、迁移方向、时间区间）。
            5. **解释性信息**：结合底图和元数据说明：为何该处出现特定类别（如地形、季节、混合像元等原因）。
            6. **建议与后处理**：提出改进分类/后处理建议（例如分割小斑块过滤、平滑、融合DEM或SAR数据）。
            
            **回答方式：**
            - 先用一句话直接回答用户问题（摘要结论）。
            - 然后按要点给出详细技术信息与数值。
            - 若存在关键数据（面积、精度指标、坐标），请单独列出表格/列表。
            
            **输出 JSON：**
            {{
              "answer": "一句话直接结论",
              "extracted_info": {{
                "source": "数据源",
                "acquisition_date": "绘图的时间",
                "crs": "投影坐标系",
                "resolution_m": "分辨率",
                "legend": {{"class_code": "label", "...": "..."}},
                "area_stats_km2": {{"class_label": {{"pixel_count": 12345, "area_km2": 12.345}}, "...": {{}}}},
                "spatial_summary": [
                   {{"class": "Apple orchard", "centroid_lon": 120.12, "centroid_lat": 36.12, "major_clusters": ["Yantai","Weihai"], "notes": "..."}}
                ],
                "changes": [{{"from_date":"2021-06-01","to_date":"2023-06-01","class_changes":{{"Apple": +15.2, "Water": -1.1}} }}],
                "notable_findings": ["某区果园扩张靠近坡地，可能拔秧替代耕地"],
                "recommendations": ["对小斑块进行面积阈值过滤（<0.01 km²）","融合Sentinel-1以降低混淆"]
              }}
            }}
            **注意事项：**
            - 如果标注不清晰，标注为"不可读"或给出估算值并说明
            - 优先回答用户的具体问题，不要罗列所有信息
            - 如果用户问"有几个卧室"，就重点回答卧室数量和位置
            - 如果用户问"客厅面积"，就重点回答客厅的尺寸和面积
            - 保持答案简洁、针对性强
            """,

        ImageType.Detection:
            """
            你是一位专业的遥感目标检测与变化检测专家。请详细分析这张遥感检测图或两时相对比图，并提取检测框/变化要点。

            **用户问题：**
            {question}
            
            **输入说明：**
            - 图片/文件：{{detection_image}}（单幅带检测框）或 {{before_image}} + {{after_image}}（两时相）。
            - 可选：检测模型输出（bbox list）、置信度阈值、类别字典、栅格分辨率、投影。
            
            **分析要求：**
            1. **目标列表**：列出所有检测到的目标（类别 + bbox）并给出置信度（0-1）。
            2. **地理位置**：若图像带地理引用，转换 bbox 为经纬度/投影坐标并给出质心坐标。
            3. **尺寸与面积估算**：计算每个目标的实际尺寸（宽、高、面积，单位 m 或 m²）。
            4. **变化检测**（若为两时相）：识别新增/消失/面积变化目标，给出变化百分比与位置。
            5. **干扰与误检分析**：指出可能的误检/漏检原因（云阴影、混合像元、近邻遮挡）。
            6. **应用建议**：后续验证手段（高分辨率影像、实地采样、人工校正）和阈值建议。
            
            **回答方式：**
            - 先一句话总结检测/变化结论；
            - 然后给出详细的目标清单表（或 JSON）并解释可疑/关键项。
            
            **输出 JSON：**
            {{
              "answer": "一句话结论",
              "extracted_info": {{
                "image_source": "{{detection_image}}",
                "resolution_m": {{gsd}},
                "detections": [
                  {{
                    "id": 1,
                    "class": "building",
                    "confidence": 0.94,
                    "bbox_px": [xmin, ymin, xmax, ymax],
                    "bbox_crs": {"xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...},
                    "centroid_lonlat": [120.123, 36.123],
                    "width_m": 12.3,
                    "height_m": 8.9,
                    "area_m2": 109.5,
                    "notes": "可能为仓库，靠近道路"
                  }}
                ],
                "changes": [
                  {{"type":"new","id_before": null,"id_after": 27,"class":"construction","area_change_pct": 45.3,"location":[lon,lat]}}
                ],
                "uncertainties": ["低置信度目标 id=3，可能为云阴影"],
                "recommendations": ["对置信度<0.5目标做人工复核","使用高分辨率影像进行验证"]
              }}
            }}
            
            
            **注意事项：**
            - 如果标注不清晰，标注为"不可读"或给出估算值并说明
            - 优先回答用户的具体问题，不要罗列所有信息
            - 如果用户问"有几个卧室"，就重点回答卧室数量和位置
            - 如果用户问"客厅面积"，就重点回答客厅的尺寸和面积
            - 保持答案简洁、针对性强
            """,

        ImageType.Technology_map:
            """
            你是一位专业的遥感处理的系统架构与技术流程图解析专家。请仔细阅读并分析这张**技术路线图 / 流程图 / 架构图**（示意图），并将其结构化成模块、流程步骤、输入输出、依赖关系与评估指标。
            
            **用户问题：**
            {question}
            
            **输入说明：**
            - 图片/文件：{{workflow_image}}（流程图、路线图、架构图）
            - 可选：相关文本文档或说明（{{doc_text}}）
            
            **分析要求：**
            1. **模块识别**：识别图中所有模块/组件（节点），并给出模块名称与简短功能描述。
            2. **流程步骤**：按照图中箭头或连线，列出有向流程步骤（1→2→3...），包括并行或分支结构。
            3. **输入/输出**：对每个模块标注输入数据类型与输出（例如 raw imagery → feature tiles → embeddings）。
            4. **依赖关系与瓶颈**：指出可能的性能瓶颈或单点故障（I/O、模型推理、索引）。
            5. **评估指标建议**：为每个关键模块给出可量化的评估指标（如 latency, throughput, accuracy, recall, disk usage）。
            6. **实现/替代建议**：提出实现技术选型建议与替代方案（例如使用 FAISS vs Milvus、Batch vs Stream 处理）。
            
            **回答方式：**
            - 先一句话给出整体功能概述；
            - 然后给出结构化模块清单、流程序列、瓶颈与优化建议。
            
            **输出 JSON：**
            {{
              "answer": "一句话概述",
              "extracted_info": {{
                "modules": [
                  {{"id":"ingest","name":"数据摄取","description":"接收 Sentinel-2/TIFF，执行预处理（裁剪/投影/云掩膜）","inputs":["raw_tiff"],"outputs":["preproc_tiles"]}},
                  {{"id":"embed","name":"Embedding","description":"使用 text-embedding-v4 对文本/视觉描述进行向量化","inputs":["text_blocks"],"outputs":["vectors"]}}
                ],
                "flow_sequence": [
                  {{"step":1,"module":"ingest","next":["preproc","tile_store"]}},
                  {{"step":2,"module":"feature_extraction","next":["embed","indexer"]}}
                ],
                "parallel_paths": [["ingest","manual_qc"],["embed","cv_model_inference"]],
                "potential_bottlenecks": ["Index building (RAM heavy)", "VLM inference (GPU latency)"],
                "metrics": {{"ingest":{{"throughput_tiles_per_min":100}},"embed":{{"latency_ms":200}} }},
                "implementation_suggestions": ["使用多线程I/O + 异步队列（Kafka）","索引采用 IVF-PQ 并分片以支持大规模数据"]
              }}
            }}
            """,

        ImageType.Technical_doc:
            """
            你是一位工业文档与标准规范分析专家。请仔细阅读这份技术文档 PDF（或多页扫描件），并提取结构化信息供后续检索/引用。
            
            **用户问题：**
            {question}
            
            **输入说明：**
            - PDF 文件：{{pdf_file}}（可为原生 PDF 或扫描图像 PDF）
            - 可选：希望的输出（如表格抽取、标准条款、关键数值）
            
            **分析要求：**
            1. **OCR 与文本提取**：对扫描页进行 OCR（保留段落结构、标题、表格、公式），并标注 OCR 置信度。
            2. **文档元信息**：提取标题、作者/单位、发布日期、版本号、页数、关键术语定义。
            3. **章节摘要**：为每章/节生成一句话摘要（3-5 词）与详细要点（3-6 条）。
            4. **表格/规范抽取**：识别并结构化导出表格数据（列名与每行），对关键表格给出 CSV 风格输出。
            5. **技术参数提取**：列出所有关键参数（数值、单位、公差、测试方法等）并为每个参数标注所在页码。
            6. **引用与合规性检查**：识别引用的标准（如 ISO、GB），并列出引用项。
            7. **问答准备**：基于文档生成可直接用于 RAG 检索的短段落片段（chunk），并返回每个 chunk 的向量化友好文本（例如去除表格格式后的纯文本摘要）。
            
            **回答方式：**
            - 先一句话总结文档主旨；
            - 然后按章节/表格/参数给出结构化列表与表格（CSV/JSON）。
            
            **输出 JSON：**
            {{
              "answer": "一句话主旨",
              "extracted_info": {{
                "meta": {{"title":"...", "author":"...", "version":"1.0","published_date":"YYYY-MM-DD","pages": 27}},
                "ocr_summary": [{{"page":1,"ocr_confidence":0.96,"text_snippet":"..."}}],
                "chapters": [
                  {{"chapter":"1 引言","summary":"说明范围与目的","key_points":["适用范围","术语定义"]}},
                  ...
                ],
                "tables_extracted": [
                  {{"table_id":1,"page":5,"columns":["Parameter","Value","Unit"],"rows":[["MaxLoad","1000","N"],["Tolerance","±0.5","mm"]]}}
                ],
                "technical_parameters": [
                  {{"name":"MaxLoad","value":1000,"unit":"N","page":5,"notes":"测试条件: ..."}}
                ],
                "referenced_standards": ["ISO 9001","GB/T 12345-2020"],
                "rationale_chunks": [
                   {{"id":"chunk_0001","page":3,"text":"此节描述了试验方法..."}}
                ],
                "recommendations": ["对 OCR 低置信度页做人工校对","对表格进行结构化校验"]
              }}
            }}
            """
    }

    def __init__(
        self,
        model_url: str = os.getenv('QWEN_BASE_URL'),
        api_key: str = os.getenv('QWEN_API_KEY'),
        model_name: str = os.getenv('QWEN_MULTIMODAL_MODAL_NAME'),
    ):
        """初始化分析器"""
        self.model_url = model_url
        self.api_key = api_key
        self.model_name = model_name

        # 检测API类型
        self.api_type = self._detect_api_type()
        print(f"✓ 初始化VLM分析器: {self.api_type} - {self.model_name}")

        # 分析器状态--是否在检测图片类型，默认为False
        self.is_detect_image = True

        # 如果使用OpenAI SDK，初始化客户端
        if self.api_type == "openai_sdk" or self.is_detect_image == True:
            from openai import AsyncOpenAI
            base_url = model_url.replace("/chat/completions", "") if "/chat/completions" in model_url else model_url
            self.openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.openai_client = None

    def _detect_api_type(self) -> str:
        """检测API类型"""
        url_lower = self.model_url.lower()

        if "dashscope" in url_lower or "aliyun" in url_lower:
            return "qwen"
        elif "anthropic" in url_lower or "claude" in url_lower:
            return "claude"
        elif "openai.com" in url_lower or "gpt" in self.model_name.lower():
            return "openai_sdk"
        else:
            return "openai_sdk"  # 默认使用OpenAI格式

    def _get_request_url(self) -> str:
        """获取完整的请求URL"""
        url = self.model_url
        if "/chat/completions" not in url and url.endswith("/v1"):
            return url + "/chat/completions"
        return url

    def load_image(self, image_source: Union[str, Path, Image.Image]) -> Image.Image:
        """
        加载图片（支持本地文件、URL、PIL Image对象）

        Args:
            image_source: 图片来源（本地路径、URL或PIL Image对象）

        Returns:
            PIL Image对象
        """
        # 如果已经是PIL Image对象
        if isinstance(image_source, Image.Image):
            print(f"✓ 接收到PIL Image对象: {image_source.size}")
            return image_source

        image_source = str(image_source)

        # 如果是URL
        if image_source.startswith(('http://', 'https://')):
            print(f"⬇ 正在从URL下载图片: {image_source}")
            response = requests.get(image_source, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            print(f"✓ 图片下载成功: {image.size}")
            return image

        # 否则视为本地文件路径
        image_path = Path(image_source)
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        print(f"📁 正在加载本地图片: {image_path.name}")
        image = Image.open(image_path)
        print(f"✓ 图片加载成功: {image.size}")
        return image

    def image_to_base64(self, image: Image.Image, max_size: int = 2000) -> str:
        """将PIL Image转换为base64字符串"""
        # 压缩大图片
        if image.width > max_size or image.height > max_size:
            image = image.copy()  # 避免修改原图
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"  图片已压缩到: {image.size}")

        buffer = io.BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    async def analyze_image(
        self,
        image_source: Union[str, Path, Image.Image],
        question: str,
        image_type: str = ImageType.Classification
    ) -> AnalysisResult:
        """
        分析图像并回答问题

        Args:
            image_source: 图片来源（本地路径、URL或PIL Image对象）
            question: 用户问题
            image_type: 图像类型 (cad / architecture / technical_doc)

        Returns:
            AnalysisResult对象
        """
        import time
        start_time = time.time()

        print("\n" + "="*60)
        print(f"🔍 开始图像分析")
        print(f"   类型: {image_type}")
        print(f"   问题: {question}")
        print("="*60)

        # 1. 加载图片
        image = self.load_image(image_source)

        # 2. 转换为base64
        print("🔄 正在将图片转换为base64...")
        image_base64 = self.image_to_base64(image)
        print(f"✓ 转换完成: {len(image_base64) / 1024:.1f} KB")

        # 3. 构建提示词
        if image_type not in self.PROMPTS:
            raise ValueError(f"不支持的图像类型: {image_type}，支持的类型: {list(self.PROMPTS.keys())}")

        print(image_type)
        prompt = self.PROMPTS[image_type].format(question=question)

        # 4. 调用VLM API
        print(f"🚀 正在调用VLM模型: {self.model_name}")
        response_data = await self._call_vlm_api(image_base64, prompt)

        # 5. 解析响应
        answer = response_data.get('answer', '')
        extracted_info = response_data.get('extracted_info', {})
        raw_response = response_data.get('raw_response', '')
        token_usage = response_data.get('token_usage', {})

        time_cost = time.time() - start_time

        print("\n" + "="*60)
        print("✅ 分析完成")
        print(f"   耗时: {time_cost:.2f}秒")
        print(f"   Token: {token_usage.get('total_tokens', 0)}")
        print("="*60)

        return AnalysisResult(
            image_type=image_type,
            question=question,
            answer=answer,
            extracted_info=extracted_info,
            raw_response=raw_response,
            token_usage=token_usage,
            time_cost=time_cost
        )

    async def _call_vlm_api(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """调用VLM API（根据API类型自动选择调用方式）"""
        if self.api_type == "openai_sdk":
            return await self._call_openai_api(image_base64, prompt)
        elif self.api_type == "qwen":
            return await self._call_qwen_api(image_base64, prompt)
        elif self.api_type == "claude":
            return await self._call_claude_api(image_base64, prompt)
        else:
            raise ValueError(f"不支持的API类型: {self.api_type}")

    async def _call_openai_api(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """调用OpenAI格式的API"""
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的工业图纸和技术文档分析专家。请仔细分析图像并按照要求的JSON格式返回结果。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
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
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            # Token统计
            token_usage = {}
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                print(f"  Token使用: 输入={token_usage['prompt_tokens']}, "
                      f"输出={token_usage['completion_tokens']}, "
                      f"总计={token_usage['total_tokens']}")

            # 解析JSON响应
            parsed = self._parse_json_response(content)

            return {
                'answer': parsed.get('answer', ''),
                'extracted_info': parsed.get('extracted_info', {}),
                'raw_response': content,
                'token_usage': token_usage
            }

        except Exception as e:
            print(f"❌ API调用失败: {e}")
            raise

    async def _call_qwen_api(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """调用通义千问API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的遥感影像分类或目标检测领域专家，并且是技术文档分析专家。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1
        }

        request_url = self._get_request_url()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    request_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API错误 {response.status}: {error_text[:500]}")

                    result = await response.json()
                    content = result['choices'][0]['message']['content']

                    # Token统计
                    token_usage = {}
                    if 'usage' in result:
                        usage = result['usage']
                        token_usage = {
                            "prompt_tokens": usage.get('prompt_tokens', 0),
                            "completion_tokens": usage.get('completion_tokens', 0),
                            "total_tokens": usage.get('total_tokens', 0)
                        }

                    parsed = self._parse_json_response(content)

                    return {
                        'answer': parsed.get('answer', ''),
                        'extracted_info': parsed.get('extracted_info', {}),
                        'raw_response': content,
                        'token_usage': token_usage
                    }
        except Exception as e:
            print(f"❌ 通义千问API调用失败: {e}")
            raise

    async def _call_claude_api(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """调用Claude API"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model_name,
            "max_tokens": 4096,
            "temperature": 0.1,
            "system": "你是一位专业的工业图纸和技术文档分析专家。",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.model_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API错误 {response.status}: {error_text[:500]}")

                    result = await response.json()
                    content = result['content'][0]['text']

                    # Token统计
                    token_usage = {}
                    if 'usage' in result:
                        usage = result['usage']
                        token_usage = {
                            "prompt_tokens": usage.get('input_tokens', 0),
                            "completion_tokens": usage.get('output_tokens', 0),
                            "total_tokens": usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                        }

                    parsed = self._parse_json_response(content)

                    return {
                        'answer': parsed.get('answer', ''),
                        'extracted_info': parsed.get('extracted_info', {}),
                        'raw_response': content,
                        'token_usage': token_usage
                    }
        except Exception as e:
            print(f"❌ Claude API调用失败: {e}")
            raise

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析JSON响应"""
        try:
            # 清理markdown代码块标记
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # 尝试提取JSON部分
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            if first_brace != -1 and last_brace != -1:
                content = content[first_brace:last_brace + 1]

            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败，返回原始内容: {e}")
            return {
                'answer': content,
                'extracted_info': {}
            }

    def print_result(self, result: AnalysisResult):
        """美化打印分析结果"""
        print("\n" + "="*60)
        print("📊 分析结果")
        print("="*60)
        print(f"\n【图像类型】 {result.image_type}")
        print(f"\n【用户问题】 {result.question}")
        print(f"\n【回答】\n{result.answer}")

        if result.extracted_info:
            print(f"\n【提取的结构化信息】")
            print(json.dumps(result.extracted_info, ensure_ascii=False, indent=2))

        print(f"\n【统计信息】")
        print(f"  耗时: {result.time_cost:.2f}秒")
        print(f"  Token: {result.token_usage.get('total_tokens', 0)}")
        print("="*60 + "\n")


# ============ 便捷函数 ============

async def analyze_classification(
    image_source: Union[str, Path, Image.Image], # Union 表示 image_source 可以是三种类型之一
    question: str,
    model_url: str = os.getenv("QWEN_BASE_URL"),
    api_key: str = os.getenv("QWEN_API_KEY"),
    model_name: str = os.getenv("QWEN_MULTIMODAL_MODAL_NAME"),
) -> AnalysisResult:
    """分析遥感分类图"""
    analyzer = SimpleVLMAnalyzer(model_url, api_key, model_name)
    return await analyzer.analyze_image(image_source, question, ImageType.Classification)


async def analyze_technology_map(
    image_source: Union[str, Path, Image.Image],
    question: str,
    model_url: str = os.getenv("QWEN_BASE_URL"),
    api_key: str = os.getenv("QWEN_API_KEY"),
    model_name: str = os.getenv("QWEN_MULTIMODAL_MODAL_NAME"),
) -> AnalysisResult:
    """分析技术流程图"""
    analyzer = SimpleVLMAnalyzer(model_url, api_key, model_name)
    return await analyzer.analyze_image(image_source, question, ImageType.Technology_map)


async def analyze_technical_document(
    image_source: Union[str, Path, Image.Image],
    question: str,
    model_url: str = os.getenv("QWEN_BASE_URL"),
    api_key: str = os.getenv("QWEN_API_KEY"),
    model_name: str = os.getenv("QWEN_MULTIMODAL_MODAL_NAME"),
) -> AnalysisResult:
    """分析工业技术档案/工艺文件"""
    analyzer = SimpleVLMAnalyzer(model_url, api_key, model_name)
    return await analyzer.analyze_image(image_source, question, ImageType.Technical_doc)


async def analyze_detection(
    image_source: Union[str, Path, Image.Image],
    question: str,
    model_url: str = os.getenv("QWEN_BASE_URL"),
    api_key: str = os.getenv("QWEN_API_KEY"),
    model_name: str = os.getenv("QWEN_MULTIMODAL_MODAL_NAME"),
) -> AnalysisResult:
    """分析室内平面布置图/建筑平面图"""
    analyzer = SimpleVLMAnalyzer(model_url, api_key, model_name)
    return await analyzer.analyze_image(image_source, question, ImageType.Detection)
