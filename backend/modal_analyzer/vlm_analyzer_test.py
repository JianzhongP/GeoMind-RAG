# 阿秋
# 2025/09/30 13:21
"""
测试脚本 - 简化版VLM图像分析器

使用示例：
python test_vlm_analyzer.py
"""

import asyncio
from pathlib import Path
from vlm_analyzer import (
    SimpleVLMAnalyzer,
    ImageType,
    analyze_classification,
    analyze_detection,
    analyze_technology_map,
    analyze_technical_document
)


async def test_classification_analysis():
    """测试分类图分析"""
    print("\n" + "🔧"*30)
    print("测试场景1: 遥感分类图分析")
    print("🔧"*30)

    # 创建分析器
    analyzer = SimpleVLMAnalyzer()

    # 示例1: 本地CAD图纸文件
    image_path = "/Snipaste_2025-09-29_17-38-49.png"
    question = "这张影像的分类共有几种地物类型？各自在哪里分布较多？"

    # 示例2: 使用便捷函数
    result = await analyze_classification(
        image_source=image_path,
        question=question
    )

    analyzer.print_result(result)

async def test_detection_analysis():
    """测试分类图分析"""
    print("\n" + "🔧"*30)
    print("测试场景1: 遥感目标检测图分析")
    print("🔧"*30)

    # 创建分析器
    analyzer = SimpleVLMAnalyzer()

    # 示例1: 本地CAD图纸文件
    image_path = "/Snipaste_2025-09-30_11-08-41(目标检测).png"
    question = "这张目标检测图主要的检测对象是什么？研究区域在哪？"

    # 示例2: 使用便捷函数
    result = await analyze_detection(
        image_source=image_path,
        question=question
    )

    analyzer.print_result(result)

async def test_technologymap_analysis():
    """测试架构图分析"""
    print("\n" + "📐"*30)
    print("测试场景2: 技术流程图分析")
    print("📐"*30)

    analyzer = SimpleVLMAnalyzer()

    # 示例1: 系统架构图
    image_path1 = "D:\Pycharm_project\my_multimodal_RAG\Snipaste_2025-09-30_11-13-35(系统架构图).png"
    question1 = "请说明这个系统的整体架构和各模块之间的调用关系"

    # 示例2: 流程图
    image_path2 = "D:\Pycharm_project\my_multimodal_RAG\Snipaste_2025-09-30_11-16-59(技术路线图).png"
    question2 = "这个技术路线中关键步骤是哪些？"

    result = await analyze_technology_map(
        image_source=image_path1,
        question=question1
    )

    analyzer.print_result(result)


async def test_technical_doc_analysis():
    """测试技术文档分析"""
    print("\n" + "📄"*30)
    print("测试场景3: 技术文档分析")
    print("📄"*30)

    analyzer = SimpleVLMAnalyzer()

    # 技术报告
    image_path = "/Snipaste_2025-09-30_11-29-44(技术文档).png"
    question = "技术报告中的关键参数和检测要求是什么？"

    result = await analyze_technical_document(
        image_source=image_path,
        question=question
    )

    analyzer.print_result(result)


async def test_with_url():
    """测试从URL加载图片"""
    print("\n" + "🌐"*30)
    print("测试场景4: 从URL加载图片")
    print("🌐"*30)

    analyzer = SimpleVLMAnalyzer()

    # 示例: 从URL加载图片
    image_url = "https://th.bing.com/th/id/R.e103e9eb9c8fa9a8b944a200f2aa942a?rik=J%2f1hsP4Px3MNjw&riu=http%3a%2f%2fwww. \
                dqxxkx.cn%2ffileup%2f1560-8999%2fFIGURE%2f2021-23-9%2fImages%2f1560-8999-23-9-1690%2fimg_7.png&ehk=OPDkAnAJY9J6NZ9ZpGALzbEUO38cBpSF12wFbt5Z76E%3d&risl=&pid=ImgRaw&r=0"
    question = "请分析这张图片的内容"

    result = await analyzer.analyze_image(
        image_source=image_url,
        question=question,
        image_type=ImageType.Classification  # 根据实际图片类型选择
    )

    analyzer.print_result(result)


async def demo_complete_workflow():
    """完整工作流演示"""
    print("\n" + "="*60)
    print("🎯 完整工作流演示")
    print("="*60)

    # 如果你有真实的图片文件，可以这样使用：

    # 1. 初始化分析器
    analyzer = SimpleVLMAnalyzer()

    # # 2. 分析分类图
    # classify_result = await analyzer.analyze_image(
    #     image_source="D:\Pycharm_project\my_multimodal_RAG\Snipaste_2025-09-31_11-08-41(目标检测).png",
    #     question="这张目标检测图主要的检测对象是什么？研究区域在哪？",
    #     image_type=ImageType.Classification
    # )
    # analyzer.print_result(classify_result)
    #
    # # 3. 分析架构图
    # arch_result = await analyzer.analyze_image(
    #     image_source="D:\Pycharm_project\my_multimodal_RAG\Snipaste_2025-09-31_11-13-35(系统架构图).png",
    #     question="模块间是如何运转的？",
    #     image_type=ImageType.Technology_map
    # )
    # analyzer.print_result(arch_result)

    # 4. 分析技术文档
    doc_result = await analyze_technical_document(
        image_source="D:\Pycharm_project\my_multimodal_RAG\Snipaste_2025-09-30_11-29-44(技术文档).png",
        question="这份文档的主要内容是什么",
        # image_type=ImageType.Technical_doc
    )
    analyzer.print_result(doc_result)

    print("\n✅ 工作流说明:")
    print("   1. 创建 SimpleVLMAnalyzer 实例")
    print("   2. 调用 analyze_image() 方法")
    print("   3. 传入图片路径/URL、用户问题、图像类型")
    print("   4. 获取 AnalysisResult 结果")
    print("   5. 使用 print_result() 打印结果")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 简化版VLM图像分析器 - 测试套件")
    print("="*60)
    print("\n支持三种场景:")
    print("  1. 遥感分类图分析")
    print("  2. 目标检测图分析")
    print("  3. 技术流程图分析")
    print("  3. 技术文档分析")
    print("\n支持的图片来源:")
    print("  ✓ 本地文件路径")
    print("  ✓ HTTP/HTTPS URL")
    print("  ✓ PIL Image对象")

    # 运行各项测试
    # await test_classification_analysis()
    # await test_architecture_analysis()
    # await test_technical_doc_analysis()
    # await test_with_url()
    await demo_complete_workflow()


if __name__ == "__main__":
    asyncio.run(main())
