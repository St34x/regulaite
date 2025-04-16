"use client"

import { useState } from "react"
import { MessageSquare, Plus, Search, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

interface ChatSession {
  id: string
  title: string
  date: string
  preview: string
}

interface ChatHistoryProps {
  sessions: ChatSession[]
  activeSessionId: string
  onSelectSession: (sessionId: string) => void
  onNewSession: () => void
  onDeleteSession: (sessionId: string) => void
}

export function ChatHistory({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
}: ChatHistoryProps) {
  const [searchQuery, setSearchQuery] = useState("")

  const filteredSessions = sessions.filter((session) => session.title.toLowerCase().includes(searchQuery.toLowerCase()))

  return (
    <div className="flex h-full w-full flex-col border-r">
      <div className="flex items-center justify-between p-4">
        <h2 className="text-lg font-semibold">Chat History</h2>
        <Button onClick={onNewSession} size="icon" variant="ghost" className="h-8 w-8">
          <Plus className="h-4 w-4" />
          <span className="sr-only">New Chat</span>
        </Button>
      </div>
      <div className="px-4 pb-2">
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search conversations"
            className="pl-8"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>
      <div className="flex-1 overflow-auto p-2">
        <Button
          variant="ghost"
          className="mb-2 flex w-full items-center justify-start gap-2 rounded-md p-2 text-left"
          onClick={onNewSession}
        >
          <Plus className="h-4 w-4" />
          <span>New Chat</span>
        </Button>
        {filteredSessions.length > 0 ? (
          <div className="space-y-2">
            {filteredSessions.map((session) => (
              <div
                key={session.id}
                className={cn(
                  "group flex cursor-pointer flex-col rounded-md p-3 text-sm transition-colors hover:bg-muted",
                  activeSessionId === session.id && "bg-muted",
                )}
                onClick={() => onSelectSession(session.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-muted-foreground" />
                    <span className="font-medium">{session.title}</span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 opacity-0 transition-opacity group-hover:opacity-100"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteSession(session.id)
                    }}
                  >
                    <Trash2 className="h-3 w-3" />
                    <span className="sr-only">Delete</span>
                  </Button>
                </div>
                <div className="mt-1 line-clamp-2 text-xs text-muted-foreground">{session.preview}</div>
                <div className="mt-1 text-xs text-muted-foreground">{session.date}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center p-4 text-center">
            <MessageSquare className="h-8 w-8 text-muted-foreground" />
            <h3 className="mt-2 text-sm font-medium">No conversations found</h3>
            <p className="mt-1 text-xs text-muted-foreground">Start a new chat or try a different search.</p>
          </div>
        )}
      </div>
    </div>
  )
}
