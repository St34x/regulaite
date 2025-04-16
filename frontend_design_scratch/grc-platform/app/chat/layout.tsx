import type React from "react"
import { Header } from "@/components/header"

export default function ChatLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">{children}</main>
      {/* Footer intentionally removed for chat page */}
    </div>
  )
}
