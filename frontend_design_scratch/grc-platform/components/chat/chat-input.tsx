"use client"

import type React from "react"

import { useState } from "react"
import { Send, Paperclip } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"

interface ChatInputProps {
  onSend: (message: string) => void
  isLoading?: boolean
}

export function ChatInput({ onSend, isLoading = false }: ChatInputProps) {
  const [input, setInput] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSend(input)
      setInput("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="flex items-end gap-2 rounded-lg border bg-background p-2">
        <Button type="button" variant="ghost" size="icon" className="h-8 w-8 shrink-0 rounded-full">
          <Paperclip className="h-4 w-4" />
          <span className="sr-only">Attach file</span>
        </Button>
        <Textarea
          placeholder="Ask a question about governance, risk, or compliance..."
          className={cn("min-h-10 resize-none border-0 p-2 shadow-none focus-visible:ring-0")}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <Button
          type="submit"
          size="icon"
          disabled={input.trim() === "" || isLoading}
          className={cn("h-8 w-8 shrink-0 rounded-full", input.trim() === "" && "opacity-50")}
        >
          <Send className="h-4 w-4" />
          <span className="sr-only">Send</span>
        </Button>
      </div>
    </form>
  )
}
