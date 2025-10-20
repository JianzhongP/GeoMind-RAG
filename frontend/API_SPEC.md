# 后端 API 接口规范文档

## 概述

本文档定义了多模态 RAG 检索系统前端与后端之间的 API 接口规范。后端服务应实现以下所有接口。

**基础信息**
- Base URL: `http://localhost:8000`
- 前端通过 `/api` 代理访问
- 内容类型: `application/json` (除文件上传外)
- 字符编码: UTF-8

---

## 1. 搜索接口

### 接口信息
- **路径**: `/search`
- **方法**: POST
- **描述**: 使用自然语言查询或图像进行多模态检索

### 请求体

```typescript
{
  query: string;              // 搜索查询文本
  model: "gpt-4o" | "qwen-vl" | "intern-vl";  // VLM 模型
  strategy: "vector" | "hybrid" | "two-stage"; // 检索策略
  topK?: number;              // 返回结果数量，默认 10
  filters?: {                 // 可选过滤条件
    fileTypes?: string[];     // 文件类型过滤
    dateRange?: {
      start: string;          // ISO 8601 格式
      end: string;
    };
    minSimilarity?: number;   // 最小相似度 (0-100)
    tags?: string[];
  };
}
```

### 请求示例

```json
{
  "query": "查询电机支架 CAD 的孔径与中心距",
  "model": "gpt-4o",
  "strategy": "hybrid",
  "topK": 10,
  "filters": {
    "fileTypes": ["CAD", "PDF"],
    "minSimilarity": 70
  }
}
```

### 响应体

```typescript
{
  results: SearchResult[];    // 搜索结果数组
  totalCount: number;         // 总结果数
  queryTime: number;          // 查询耗时（毫秒）
  model: string;              // 使用的模型
  strategy: string;           // 使用的策略
}

interface SearchResult {
  id: string;                 // 文档唯一标识
  fileName: string;           // 文件名
  filePath: string;           // 文件路径
  fileType: string;           // 文件类型（CAD/PDF/BOM等）
  similarity: number;         // 相似度分数 (0-100)
  page?: string;              // 页码或视图（如 "Page 12" 或 "View A"）
  date: string;               // 文档日期 (YYYY-MM-DD)
  snippet: string;            // 内容片段（100-200字）
  citationNumber: number;     // 引用编号
  thumbnailType: "cad" | "pdf" | "image";
  thumbnailUrl?: string;      // 缩略图 URL
  previewUrl?: string;        // 预览 URL
  version: string;            // 版本号
  structuredData: Array<{     // 结构化提取数据
    label: string;
    value: string;
  }>;
}
```

### 响应示例

```json
{
  "results": [
    {
      "id": "doc_123456",
      "fileName": "MotorBracket_v1.2.dwg",
      "filePath": "/工程制造/机械零件/电机支架/v1.2/",
      "fileType": "CAD",
      "similarity": 87,
      "page": "View A",
      "date": "2025-09-21",
      "snippet": "孔径 Ø8mm；中心距 42mm；材质 6061-T6；适用于标准电机安装，符合 GB/T 规范要求。",
      "citationNumber": 1,
      "thumbnailType": "cad",
      "thumbnailUrl": "/api/thumbnail/doc_123456",
      "previewUrl": "/api/preview/doc_123456",
      "version": "v1.2",
      "structuredData": [
        { "label": "孔径", "value": "Ø8mm" },
        { "label": "中心距", "value": "42mm" },
        { "label": "材质", "value": "6061-T6" },
        { "label": "公差等级", "value": "IT7" }
      ]
    }
  ],
  "totalCount": 3,
  "queryTime": 823,
  "model": "gpt-4o",
  "strategy": "hybrid"
}
```

### 错误响应

```json
{
  "error": "SearchError",
  "message": "查询处理失败",
  "statusCode": 500
}
```

---

## 2. 文件上传接口

### 接口信息
- **路径**: `/upload`
- **方法**: POST
- **描述**: 上传文档文件（图像、CAD、PDF等）
- **Content-Type**: `multipart/form-data`

### 请求参数

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| file | File | 是 | 文件对象 |
| metadata | string | 否 | JSON 字符串，包含额外元数据 |

### Metadata 格式

```typescript
{
  tags?: string[];            // 标签
  category?: string;          // 分类
  description?: string;       // 描述
}
```

### 请求示例

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/drawing.dwg" \
  -F 'metadata={"tags":["电机","支架"],"category":"机械零件"}'
```

### 响应体

```typescript
{
  success: boolean;           // 是否成功
  fileId: string;             // 文件 ID
  fileName: string;           // 文件名
  message?: string;           // 消息
}
```

### 响应示例

```json
{
  "success": true,
  "fileId": "doc_789012",
  "fileName": "drawing.dwg",
  "message": "文件上传成功"
}
```

### 支持的文件格式

- **图像**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`
- **CAD**: `.dwg`, `.dxf`, `.step`, `.stp`, `.iges`, `.igs`
- **文档**: `.pdf`, `.doc`, `.docx`
- **数据**: `.xlsx`, `.xls`, `.csv`

### 文件大小限制

- 最大文件大小: 50 MB
- 建议优化大文件后再上传

---

## 3. 追问接口

### 接口信息
- **路径**: `/question`
- **方法**: POST
- **描述**: 对特定文档进行追问，获取更详细的信息

### 请求体

```typescript
{
  documentId: string;         // 文档 ID
  question: string;           // 问题文本
  model: "gpt-4o" | "qwen-vl" | "intern-vl";  // VLM 模型
}
```

### 请求示例

```json
{
  "documentId": "doc_123456",
  "question": "中心距是否满足 v1.2 工艺规范？",
  "model": "gpt-4o"
}
```

### 响应体

```typescript
{
  answer: string;             // 回答文本
  citations: number[];        // 引用的文档编号
  confidence: number;         // 置信度 (0-1)
}
```

### 响应示例

```json
{
  "answer": "根据图纸分析，中心距为 42mm，符合 v1.2 工艺规范要求（40-45mm）。该参数在 GB/T 标准中明确定义，当前设计满足要求。",
  "citations": [1],
  "confidence": 0.95
}
```

---

## 4. 预览接口

### 接口信息
- **路径**: `/preview/{documentId}`
- **方法**: GET
- **描述**: 获取文档预览图像

### 路径参数

| 参数 | 类型 | 描述 |
|------|------|------|
| documentId | string | 文档 ID |

### 请求示例

```
GET /api/preview/doc_123456
```

### 响应

返回图像文件（PNG 或 JPEG）

- **Content-Type**: `image/png` 或 `image/jpeg`
- **分辨率**: 建议 1920x1080 或更高
- **压缩**: 适当压缩以平衡质量和加载速度

### 错误响应

- **404**: 文档不存在
- **500**: 预览生成失败

---

## 5. 缩略图接口

### 接口信息
- **路径**: `/thumbnail/{documentId}`
- **方法**: GET
- **描述**: 获取文档缩略图

### 路径参数

| 参数 | 类型 | 描述 |
|------|------|------|
| documentId | string | 文档 ID |

### 请求示例

```
GET /api/thumbnail/doc_123456
```

### 响应

返回缩略图文件（PNG 或 JPEG）

- **Content-Type**: `image/png` 或 `image/jpeg`
- **分辨率**: 200x200 像素（正方形）
- **压缩**: 高压缩以加快加载

---

## 6. 下载接口

### 接口信息
- **路径**: `/download/{documentId}`
- **方法**: GET
- **描述**: 下载原始文档文件

### 路径参数

| 参数 | 类型 | 描述 |
|------|------|------|
| documentId | string | 文档 ID |

### 请求示例

```
GET /api/download/doc_123456
```

### 响应

返回原始文件

- **Content-Type**: 根据文件类型动态设置
- **Content-Disposition**: `attachment; filename="文件名"`
- **文件完整性**: 确保文件未损坏

---

## 7. 健康检查接口

### 接口信息
- **路径**: `/health`
- **方法**: GET
- **描述**: 检查后端服务健康状态

### 响应体

```typescript
{
  status: "ok" | "error";     // 服务状态
  timestamp: string;          // ISO 8601 时间戳
  version?: string;           // API 版本（可选）
}
```

### 响应示例

```json
{
  "status": "ok",
  "timestamp": "2025-10-14T12:00:00Z",
  "version": "1.0.0"
}
```

---

## 通用错误响应格式

所有接口在出错时应返回统一的错误格式：

```typescript
{
  error: string;              // 错误类型
  message: string;            // 错误描述
  statusCode: number;         // HTTP 状态码
  details?: any;              // 详细错误信息（可选）
}
```

### 常见 HTTP 状态码

| 状态码 | 含义 | 使用场景 |
|--------|------|----------|
| 200 | 成功 | 请求成功处理 |
| 400 | 请求错误 | 参数验证失败 |
| 401 | 未授权 | 需要身份验证（如需要） |
| 404 | 未找到 | 资源不存在 |
| 413 | 载荷过大 | 上传文件过大 |
| 500 | 服务器错误 | 内部处理错误 |
| 503 | 服务不可用 | 服务暂时不可用 |

---

## CORS 配置

后端需要配置 CORS 以允许前端跨域访问：

```python
# FastAPI 示例
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 性能建议

### 1. 搜索接口
- 响应时间目标: < 2 秒
- 使用缓存减少重复查询
- 实现分页支持（通过 `topK` 和 `offset`）

### 2. 上传接口
- 支持断点续传
- 实现上传进度回调
- 异步处理文件索引

### 3. 预览生成
- 预生成常用文档的缩略图和预览
- 使用 CDN 分发静态资源
- 实现懒加载

### 4. 追问接口
- 响应时间目标: < 1 秒
- 使用流式响应（可选）
- 缓存常见问题答案

---

## 安全建议

1. **文件上传安全**
   - 验证文件类型和大小
   - 扫描恶意代码
   - 使用安全的文件存储路径

2. **输入验证**
   - 验证所有用户输入
   - 防止 SQL 注入和 XSS
   - 限制查询长度

3. **访问控制**
   - 实现身份验证（如需要）
   - 文档访问权限控制
   - API 速率限制

---

## 测试建议

### 单元测试
- 每个接口至少 5 个测试用例
- 覆盖正常和异常情况

### 集成测试
- 测试完整的搜索流程
- 测试文件上传和检索
- 测试追问功能

### 性能测试
- 并发请求测试（100+ 用户）
- 大文件上传测试
- 长时间运行稳定性测试

---

## 实现优先级

**第一阶段（MVP）**
1. ✅ 搜索接口（基础功能）
2. ✅ 健康检查接口
3. ✅ 缩略图接口（返回占位图）

**第二阶段**
4. ✅ 文件上传接口
5. ✅ 预览接口
6. ✅ 追问接口

**第三阶段（优化）**
7. ✅ 下载接口
8. ✅ 高级过滤和分页
9. ✅ 性能优化和缓存

---

## 联系方式

如有接口实现疑问，请参考：
- 前端类型定义: `frontend/types/index.ts`
- API 服务实现: `frontend/services/api.ts`
- Mock 数据参考: `frontend/services/mockApi.ts`

祝开发顺利！
