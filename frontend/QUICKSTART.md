# 快速启动指南

## 前提条件

确保已安装：
- Node.js >= 18.0.0
- npm >= 9.0.0

检查版本：
```bash
node --version
npm --version
```

## 安装步骤

### 1. 安装依赖（首次运行）

```bash
npm install
```

**注意**: 首次安装可能需要 3-5 分钟，请耐心等待。

如果安装速度慢，可以使用国内镜像：

```bash
# 使用淘宝镜像
npm install --registry=https://registry.npmmirror.com

# 或者使用 cnpm
npm install -g cnpm --registry=https://registry.npmmirror.com
cnpm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

应用将在 http://localhost:3000 启动

### 3. 构建生产版本

```bash
npm run build
```

## 开发模式

### Mock API 模式（默认）

项目默认使用 Mock API，无需启动后端服务即可开发：

```bash
# 启动即可，会自动使用 Mock 数据
npm run dev
```

### 连接真实后端

1. 创建 `.env` 文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件：
```env
VITE_USE_MOCK_API=false
```

3. 确保后端运行在 http://localhost:8000

4. 启动前端：
```bash
npm run dev
```

## 常见问题

### 问题 1: npm install 失败

**解决方案**:
```bash
# 清理缓存
npm cache clean --force

# 删除 node_modules
rm -rf node_modules package-lock.json

# 重新安装
npm install
```

### 问题 2: 端口被占用

**解决方案**:
修改 `vite.config.ts` 中的端口号：
```typescript
server: {
  port: 3001,  // 改为其他端口
}
```

### 问题 3: TypeScript 编译错误

**解决方案**:
```bash
# 检查 TypeScript 版本
npx tsc --version

# 如果有错误，尝试重新安装
npm install typescript@latest --save-dev
```

### 问题 4: Tailwind CSS 样式不生效

**解决方案**:
1. 确认 `style/globals.css` 已在 `main.tsx` 中导入
2. 重启开发服务器
3. 清除浏览器缓存

## 开发工具推荐

### VS Code 扩展
- **ESLint**: 代码检查
- **Prettier**: 代码格式化
- **Tailwind CSS IntelliSense**: Tailwind 智能提示
- **TypeScript Vue Plugin (Volar)**: TypeScript 支持

### Chrome 扩展
- **React Developer Tools**: React 调试
- **Redux DevTools**: 状态管理调试（如需要）

## 项目结构速览

```
frontend/
├── components/     # 业务组件
├── ui/            # UI 基础组件
├── hooks/         # 自定义 Hooks
├── services/      # API 服务
├── types/         # 类型定义
├── lib/           # 工具函数
├── style/         # 全局样式
├── App.tsx        # 主应用
└── main.tsx       # 入口文件
```

## 下一步

1. ✅ 查看 README.md 了解详细功能
2. ✅ 查看 API_SPEC.md 了解后端接口规范
3. ✅ 查看 CLAUDE.md 了解项目架构
4. ✅ 开始开发前端功能
5. ✅ 根据 API_SPEC.md 开发后端接口

## 技术支持

遇到问题？
1. 查看 README.md 中的常见问题部分
2. 检查浏览器控制台错误信息
3. 查看 `services/mockApi.ts` 了解 Mock 数据结构

祝开发顺利！🚀
