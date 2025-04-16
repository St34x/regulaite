import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"
import { Bot, User } from "lucide-react"
import ReactMarkdown from "react-markdown"

interface ChatMessageProps {
  message: {
    role: "user" | "assistant"
    content: string
  }
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"

  return (
    <div className={cn("flex w-full items-start gap-4 p-4", isUser ? "bg-muted/50" : "bg-background")}>
      <Avatar className="h-8 w-8">
        <AvatarFallback className={isUser ? "bg-accent" : "bg-primary"}>
          {isUser ? (
            <User className="h-4 w-4 text-accent-foreground" />
          ) : (
            <Bot className="h-4 w-4 text-primary-foreground" />
          )}
        </AvatarFallback>
      </Avatar>
      <div className="flex-1 space-y-2">
        <div className="prose-sm prose dark:prose-invert max-w-none">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
