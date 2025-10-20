# 多模态文档检索 RAG（VLM）- 前端项目

这是一个基于 React + TypeScript + Vite 构建的多模态 RAG 检索系统前端应用，支持对 CAD 图纸、架构图、工程文档等进行语义检索。

## 功能特性

- 🔍 **多模态检索**: 支持文本查询和图像/CAD 文件上传检索
- 🎨 **现代化 UI**: 采用 Glass Morphism 设计风格，渐变配色方案
- 🤖 **VLM 模型支持**: 可选择 GPT-4o、Qwen-VL、InternVL 等多模态大模型
- 📊 **结构化数据提取**: 自动从文档中提取关键参数信息
- 💬 **文档追问**: 对检索结果进行深度追问，获取更多细节
- 🎯 **检索策略**: 支持向量检索、混合检索、二阶段检索等多种策略
- 📁 **文档溯源**: 完整的文档来源信息和版本追踪

## 技术栈

- **框架**: React 18 + TypeScript
- **构建工具**: Vite 6
- **样式**: Tailwind CSS 3
- **UI 组件**: Radix UI (shadcn/ui)
- **图标**: Lucide React
- **HTTP 客户端**: Axios
- **状态管理**: React Hooks

## 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# 开发模式下使用 Mock API（不需要后端）
VITE_USE_MOCK_API=true

# 如果要连接真实后端，设置为 false 并配置后端地址
# VITE_USE_MOCK_API=false
# VITE_API_BASE_URL=http://localhost:8000
```

### 3. 启动开发服务器

```bash
npm run dev
```

应用将在 http://localhost:3000 启动

### 4. 构建生产版本

```bash
npm run build
```

构建输出位于 `dist/` 目录

### 5. 预览生产构建

```bash
npm run preview
```

## 项目结构

```
frontend/
├── components/          # React 组件
│   ├── Header.tsx      # 顶部导航栏
│   ├── SearchBar.tsx   # 搜索栏组件（支持文本和文件上传）
│   ├── ResultCard.tsx  # 搜索结果卡片
│   ├── PreviewPanel.tsx # 文档预览和追问面板
│   └── EmptyState.tsx  # 空状态组件
├── ui/                 # UI 基础组件库（40+ 组件）
├── hooks/              # 自定义 React Hooks
│   ├── useSearch.ts    # 搜索功能 Hook
│   ├── useUpload.ts    # 文件上传 Hook
│   └── useFollowUpQuestion.ts # 追问功能 Hook
├── services/           # API 服务层
│   ├── api.ts          # 真实 API 客户端
│   ├── mockApi.ts      # Mock API（开发测试用）
│   └── index.ts        # 服务导出（自动切换）
├── types/              # TypeScript 类型定义
│   └── index.ts        # 核心类型接口
├── lib/                # 工具函数
│   └── utils.ts        # 通用工具函数
├── style/              # 全局样式
│   └── globals.css     # 全局 CSS 和主题变量
├── App.tsx             # 主应用组件
├── main.tsx            # 应用入口
├── index.html          # HTML 模板
└── vite.config.ts      # Vite 配置

```

## 开发模式

项目支持两种开发模式：

### Mock API 模式（默认）

适合前端独立开发，不需要启动后端服务：

```env
VITE_USE_MOCK_API=true
```

Mock API 会返回模拟数据，包括：
- 3 条预设的搜索结果
- 模拟的文件上传响应
- 模拟的追问回答

### 真实 API 模式

连接实际后端服务：

```env
VITE_USE_MOCK_API=false
```

确保后端服务运行在 `http://localhost:8000`（或配置的地址）

## 后端 API 接口规范

前端已经预留了完整的后端接口定义，后端开发者需要实现以下接口：

### 1. 搜索接口

**POST** `/api/search`

请求体：
```json
{
  "query": "查询电机支架 CAD 的孔径与中心距",
  "model": "gpt-4o",
  "strategy": "hybrid",
  "topK": 10,
  "filters": {
    "fileTypes": ["CAD", "PDF"],
    "dateRange": {
      "start": "2025-01-01",
      "end": "2025-12-31"
    },
    "minSimilarity": 0.7
  }
}
```

响应：
```json
{
  "results": [
    {
      "id": "doc123",
      "fileName": "MotorBracket_v1.2.dwg",
      "filePath": "/工程制造/机械零件/电机支架/v1.2/",
      "fileType": "CAD",
      "similarity": 87,
      "page": "View A",
      "date": "2025-09-21",
      "snippet": "孔径 Ø8mm；中心距 42mm；材质 6061-T6；适用于标准电机安装，符合 GB/T 规范要求。",
      "citationNumber": 1,
      "thumbnailType": "cad",
      "thumbnailUrl": "/api/thumbnail/doc123",
      "previewUrl": "/api/preview/doc123",
      "version": "v1.2",
      "structuredData": [
        { "label": "孔径", "value": "Ø8mm" },
        { "label": "中心距", "value": "42mm" },
        { "label": "材质", "value": "6061-T6" },
        { "label": "公差等级", "value": "IT7" }
      ]
    }
  ],
  "totalCount": 10,
  "queryTime": 823,
  "model": "gpt-4o",
  "strategy": "hybrid"
}
```

### 2. 文件上传接口

**POST** `/api/upload`

请求：`multipart/form-data`
- `file`: 文件对象
- `metadata`: JSON 字符串（可选）

响应：
```json
{
  "success": true,
  "fileId": "doc456",
  "fileName": "drawing.dwg",
  "message": "文件上传成功"
}
```

### 3. 追问接口

**POST** `/api/question`

请求体：
```json
{
  "documentId": "doc123",
  "question": "中心距是否满足 v1.2 工艺规范？",
  "model": "gpt-4o"
}
```

响应：
```json
{
  "answer": "根据图纸分析，中心距为 42mm，符合 v1.2 工艺规范要求（40-45mm）。",
  "citations": [1],
  "confidence": 0.95
}
```

### 4. 预览接口

**GET** `/api/preview/{documentId}`

返回文档预览图像（PNG/JPG）

### 5. 缩略图接口

**GET** `/api/thumbnail/{documentId}`

返回文档缩略图（PNG/JPG）

### 6. 下载接口

**GET** `/api/download/{documentId}`

返回原始文档文件（CAD/PDF/等）

### 7. 健康检查接口

**GET** `/api/health`

响应：
```json
{
  "status": "ok",
  "timestamp": "2025-10-14T12:00:00Z"
}
```

## 类型定义

所有 TypeScript 类型定义位于 `types/index.ts`，包括：

- `VLMModel`: VLM 模型类型
- `RetrievalStrategy`: 检索策略类型
- `SearchResult`: 搜索结果接口
- `SearchRequest`: 搜索请求接口
- `SearchResponse`: 搜索响应接口
- `UploadRequest`: 上传请求接口
- `UploadResponse`: 上传响应接口
- `FollowUpQuestionRequest`: 追问请求接口
- `FollowUpQuestionResponse`: 追问响应接口

## API 服务层

API 服务封装在 `services/` 目录：

```typescript
import { service } from './services';

// 搜索
const response = await service.search(request);

// 上传
const result = await service.uploadDocument(request);

// 追问
const answer = await service.askFollowUpQuestion(request);

// 获取预览 URL
const url = service.getPreviewUrl(documentId);

// 下载文档
await service.downloadDocument(documentId, fileName);
```

## 自定义 Hooks

```typescript
// 搜索
const { results, isLoading, error, search } = useSearch();
await search(query, model, strategy);

// 上传
const { isUploading, uploadDocument } = useUpload();
await uploadDocument(file, metadata);

// 追问
const { qaHistory, isAsking, askQuestion } = useFollowUpQuestion();
await askQuestion(documentId, question, model);
```

## 代理配置

Vite 开发服务器已配置代理，将 `/api` 请求转发到后端：

```typescript
// vite.config.ts
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
}
```

## 设计系统

### 颜色主题

项目使用 Cyan (#00d4ff) + Purple (#a855f7) 渐变配色：

- **Primary**: `#00d4ff` (青色)
- **Secondary**: 半透明深色卡片
- **Background**: 深色背景 + 网格纹理
- **Muted**: 中性灰色

### 组件风格

- **Glass Morphism**: 半透明卡片 + 背景模糊
- **Gradient**: 渐变按钮和装饰元素
- **Rounded**: 大圆角 (12px-16px)
- **Shadow**: 柔和阴影 + 主色发光效果

## 常见问题

### 1. 端口冲突

修改 `vite.config.ts` 中的端口：

```typescript
server: {
  port: 3001, // 改为其他端口
}
```

### 2. 后端连接失败

检查：
- 后端是否运行在 `http://localhost:8000`
- `.env` 中 `VITE_USE_MOCK_API` 是否设置为 `false`
- 代理配置是否正确

### 3. Mock 数据不显示

确保 `.env` 文件中设置：
```env
VITE_USE_MOCK_API=true
```

重启开发服务器。

## 后续开发建议

1. **连接真实后端**: 将 `VITE_USE_MOCK_API` 设置为 `false`，实现后端 API
2. **增强上传功能**: 添加拖拽上传、进度条、多文件上传
3. **优化预览**: 实现 PDF 预览器、CAD 渲染器
4. **添加过滤器**: 实现高级搜索过滤功能
5. **性能优化**: 虚拟滚动、懒加载、缓存策略
6. **国际化**: 添加多语言支持

## 许可证

MIT License

## 联系方式

- 课程: 赋范空间公开课
- 项目: 多模态文档检索 RAG（VLM）
