# GeoMind-RAG
### VLM-Based Multimodal RAG Remote Sensing Knowledge Question-Answering System  
这是一个针对遥感检测结果、文档、技术路线等私域知识的多模态RAG问答系统，LLM采用的是**qwen-vl-plus**，embedding model采用的是通义千问的**text-embedding-v4**。其他的自行添加即可。  

## Start:  
### 后端：  
在此之前，你需要通过以下命令安装本项目的支持包：(注意需要先 cd 到指定目录)  
`pip install requestment.txt`  

进入“./backend”文件夹，打开 main_service.py 脚本，并执行。需要注意的是，在此之前，你需要先通过 阿里的百炼平台 获取**API_KEY**和**BASE_URL**。运行后，会得到如下的提示：  
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/6f43fa2a-2325-47e5-8fbd-3fb01951864f" />  
出现 **200 OK** ，表示后端服务已启动完成。我们可以通过 FASTAPI 提供的交互文档进行测试。  
**注意**：本项目的重要参数(如API_KEY等，都放置在 **./env** 文件中，你可以通过该文件修改你的API_KEY)

### 前端：  
前端需要首先下载 **Node.js** 包，然后你可以利用 React 脚手架自行构建自己的前端，也可以利用下载好的代码，进入 frontend 目录，终端下运行如下命令：  
`npx run dev`  
然后通过得到的网址即可访问！  

## 系统前端部分：  
<img width="1910" height="923" alt="image" src="https://github.com/user-attachments/assets/612ee313-61bc-46d6-9da3-d30d55d819a3" />  
<img width="1920" height="928" alt="image" src="https://github.com/user-attachments/assets/3834be5a-365c-43a6-8d2f-44f62eebda5f" />  

## 功能预览：  
<img width="1920" height="984" alt="image" src="https://github.com/user-attachments/assets/c6dfb997-66a4-4bcc-8e15-ae58fc371ac2" />  
<img width="1920" height="988" alt="image" src="https://github.com/user-attachments/assets/16e0d314-8721-4de3-881e-e9db7e72b9ff" />  

**这里设计的预览窗口是可拉伸的。** 
