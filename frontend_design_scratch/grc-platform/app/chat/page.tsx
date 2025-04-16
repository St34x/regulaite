"use client"

import { useState, useEffect } from "react"
import { Shield, PanelLeft, X } from "lucide-react"
import { ChatInput } from "@/components/chat/chat-input"
import { ChatMessage } from "@/components/chat/chat-message"
import { ChatHistory } from "@/components/chat/chat-history"
import { Button } from "@/components/ui/button"
import { useMediaQuery } from "@/hooks/use-mobile"

// Sample chat sessions
const sampleSessions = [
  {
    id: "1",
    title: "SOC 2 Compliance",
    date: "Today, 2:30 PM",
    preview: "What are the key requirements for SOC 2 compliance?",
    messages: [
      {
        role: "assistant" as const,
        content:
          "Hello! I'm your GRC AI Assistant. I can help you with governance, risk, and compliance questions. How can I assist you today?",
      },
      {
        role: "user" as const,
        content: "What are the key requirements for SOC 2 compliance?",
      },
      {
        role: "assistant" as const,
        content:
          "SOC 2 is a framework for managing customer data. The key requirements include security, availability, processing integrity, confidentiality, and privacy controls. Would you like me to elaborate on any specific aspect?",
      },
    ],
  },
  {
    id: "2",
    title: "Risk Assessment",
    date: "Yesterday, 10:15 AM",
    preview: "How do I conduct a risk assessment for my organization?",
    messages: [
      {
        role: "assistant" as const,
        content:
          "Hello! I'm your GRC AI Assistant. I can help you with governance, risk, and compliance questions. How can I assist you today?",
      },
      {
        role: "user" as const,
        content: "How do I conduct a risk assessment for my organization?",
      },
      {
        role: "assistant" as const,
        content:
          "A risk assessment involves identifying assets, threats, vulnerabilities, and existing controls, then evaluating likelihood and impact to determine risk levels. Would you like me to outline a step-by-step process?",
      },
    ],
  },
  {
    id: "3",
    title: "GDPR Requirements",
    date: "Apr 15, 2023",
    preview: "What are the GDPR requirements for data processing?",
    messages: [
      {
        role: "assistant" as const,
        content:
          "Hello! I'm your GRC AI Assistant. I can help you with governance, risk, and compliance questions. How can I assist you today?",
      },
      {
        role: "user" as const,
        content: "What are the GDPR requirements for data processing?",
      },
      {
        role: "assistant" as const,
        content:
          "GDPR requires lawful basis for processing, data minimization, purpose limitation, accuracy, storage limitation, integrity, confidentiality, and accountability. Organizations must also respect data subject rights and report breaches.",
      },
    ],
  },
]

// Initial message for new chats
const initialMessage = {
  role: "assistant" as const,
  content:
    "Hello! I'm your GRC AI Assistant. I can help you with governance, risk, and compliance questions. How can I assist you today?",
}

// Sample suggested questions
const suggestedQuestions = [
  "What are the key requirements for SOC 2 compliance?",
  "How do I conduct a risk assessment for my organization?",
  "Explain the main components of a governance framework",
  "What are the GDPR requirements for data processing?",
]

export default function ChatPage() {
  const [sessions, setSessions] = useState(sampleSessions)
  const [activeSessionId, setActiveSessionId] = useState("1")
  const [messages, setMessages] = useState(sessions.find((s) => s.id === "1")?.messages || [initialMessage])
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const isMobile = useMediaQuery("(max-width: 768px)")

  // Close sidebar on mobile by default
  useEffect(() => {
    if (isMobile) {
      setIsSidebarOpen(false)
    } else {
      setIsSidebarOpen(true)
    }
  }, [isMobile])

  const handleSendMessage = async (content: string) => {
    // Add user message
    const userMessage = { role: "user" as const, content }
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)

    // Update session with new message
    updateSessionMessages(activeSessionId, updatedMessages)

    // Set loading state
    setIsLoading(true)

    // Simulate AI response (in a real app, this would be an API call)
    setTimeout(() => {
      // Sample responses based on keywords
      let responseContent = "I'll need to research that further. Can you provide more details about your question?"

      if (content.toLowerCase().includes("soc 2")) {
        responseContent =
          "SOC 2 is a framework for managing customer data. The key requirements include security, availability, processing integrity, confidentiality, and privacy controls. Would you like me to elaborate on any specific aspect?"
      } else if (content.toLowerCase().includes("risk assessment")) {
        responseContent =
          "A risk assessment involves identifying assets, threats, vulnerabilities, and existing controls, then evaluating likelihood and impact to determine risk levels. Would you like me to outline a step-by-step process?"
      } else if (content.toLowerCase().includes("governance")) {
        responseContent =
          "A governance framework typically includes leadership structures, policies, procedures, standards, and accountability mechanisms. It should align with your organization's objectives and regulatory requirements."
      } else if (content.toLowerCase().includes("gdpr")) {
        responseContent =
          "GDPR requires lawful basis for processing, data minimization, purpose limitation, accuracy, storage limitation, integrity, confidentiality, and accountability. Organizations must also respect data subject rights and report breaches."
      }

      const assistantMessage = { role: "assistant" as const, content: responseContent }
      const finalMessages = [...updatedMessages, assistantMessage]
      setMessages(finalMessages)
      updateSessionMessages(activeSessionId, finalMessages)
      setIsLoading(false)
    }, 1500)
  }

  const handleSelectSession = (sessionId: string) => {
    setActiveSessionId(sessionId)
    const session = sessions.find((s) => s.id === sessionId)
    if (session) {
      setMessages(session.messages)
    }
    if (isMobile) {
      setIsSidebarOpen(false)
    }
  }

  const handleNewSession = () => {
    const newSessionId = String(Date.now())
    const newSession = {
      id: newSessionId,
      title: "New Conversation",
      date: "Just now",
      preview: "Start a new conversation",
      messages: [initialMessage],
    }

    setSessions([newSession, ...sessions])
    setActiveSessionId(newSessionId)
    setMessages([initialMessage])
    if (isMobile) {
      setIsSidebarOpen(false)
    }
  }

  const handleDeleteSession = (sessionId: string) => {
    const updatedSessions = sessions.filter((session) => session.id !== sessionId)
    setSessions(updatedSessions)

    if (sessionId === activeSessionId && updatedSessions.length > 0) {
      handleSelectSession(updatedSessions[0].id)
    } else if (updatedSessions.length === 0) {
      handleNewSession()
    }
  }

  const updateSessionMessages = (sessionId: string, updatedMessages: typeof messages) => {
    setSessions((prevSessions) =>
      prevSessions.map((session) => {
        if (session.id === sessionId) {
          // Update the title based on the first user message if it's still the default
          let title = session.title
          if (title === "New Conversation" && updatedMessages.length > 1) {
            const firstUserMessage = updatedMessages.find((m) => m.role === "user")
            if (firstUserMessage) {
              title = firstUserMessage.content.slice(0, 30) + (firstUserMessage.content.length > 30 ? "..." : "")
            }
          }

          // Update the preview with the latest message
          const latestMessage = updatedMessages[updatedMessages.length - 1]
          const preview = latestMessage
            ? latestMessage.content.slice(0, 60) + (latestMessage.content.length > 60 ? "..." : "")
            : ""

          return {
            ...session,
            title,
            preview,
            messages: updatedMessages,
            date: "Just now",
          }
        }
        return session
      }),
    )
  }

  const handleSuggestedQuestion = (question: string) => {
    handleSendMessage(question)
  }

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col">
      <div className="flex h-full">
        {/* Sidebar */}
        <div
          className={`${
            isSidebarOpen ? "translate-x-0" : "-translate-x-full"
          } absolute inset-y-0 z-20 w-64 transform transition-transform duration-200 ease-in-out md:relative md:translate-x-0`}
        >
          <ChatHistory
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={handleSelectSession}
            onNewSession={handleNewSession}
            onDeleteSession={handleDeleteSession}
          />
        </div>

        {/* Main Chat Area */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Chat Header */}
          <div className="flex items-center justify-between border-b p-4">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                className="md:hidden"
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              >
                {isSidebarOpen ? <X className="h-5 w-5" /> : <PanelLeft className="h-5 w-5" />}
              </Button>
              <h1 className="text-xl font-semibold">
                {sessions.find((s) => s.id === activeSessionId)?.title || "Chat"}
              </h1>
            </div>
            <Button variant="outline" size="sm" onClick={handleNewSession}>
              New Chat
            </Button>
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-2 p-8 text-center">
                <Shield className="h-12 w-12 text-accent opacity-50" />
                <h3 className="text-xl font-semibold">GRC AI Assistant</h3>
                <p className="text-sm text-muted-foreground">Ask me anything about governance, risk, and compliance.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <ChatMessage key={index} message={message} />
                ))}
                {isLoading && (
                  <div className="flex w-full items-center gap-4 rounded-lg bg-muted/20 p-4">
                    <div className="h-8 w-8 animate-pulse rounded-full bg-muted"></div>
                    <div className="space-y-2">
                      <div className="h-4 w-24 animate-pulse rounded bg-muted"></div>
                      <div className="h-4 w-64 animate-pulse rounded bg-muted"></div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Suggested Questions (only show for new chats) */}
            {messages.length === 1 && (
              <div className="mt-8 grid gap-2 sm:grid-cols-2">
                <h3 className="col-span-full text-sm font-medium">Suggested Questions</h3>
                {suggestedQuestions.map((question, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    className="justify-start text-left text-sm"
                    onClick={() => handleSuggestedQuestion(question)}
                  >
                    {question}
                  </Button>
                ))}
              </div>
            )}
          </div>

          {/* Chat Input */}
          <div className="border-t p-4">
            <ChatInput onSend={handleSendMessage} isLoading={isLoading} />
          </div>
        </div>
      </div>
    </div>
  )
}
