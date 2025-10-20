import { FileSearch } from "lucide-react";

interface EmptyStateProps {
  type: "initial" | "no-results";
}

export function EmptyState({ type }: EmptyStateProps) {
  if (type === "initial") {
      //         可以自行添加装饰
    return (
      <div>
      </div>
    );
  }

  return (
      <div>
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <FileSearch className="h-24 w-24 text-muted-foreground/30 mb-6" />
          <h3 className="text-[16px] mb-2">未找到匹配项</h3>
          <p className="text-[13px] text-muted-foreground mb-4">
            可尝试放宽筛选或更换检索策略
          </p>
        </div>
        <div className="flex flex-col items-center justify-center py-10 px-6 text-center">
            <h3 className="text-[16px] font-semibold mb-3 text-primary">AI 智能回答</h3>
            <p className="text-[14px] text-gray-600 leading-relaxed max-w-[600px]">
              {llmAnswer || "暂无可用回答，请尝试输入更详细的问题。"}
            </p>
        </div>
       </div>
  );
}
