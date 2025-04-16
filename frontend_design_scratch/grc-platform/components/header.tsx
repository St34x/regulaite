"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Shield, Menu, X } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/theme-toggle"
import { cn } from "@/lib/utils"

export function Header() {
  const pathname = usePathname()
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen)

  const navigation = [
    { name: "Dashboard", href: "/" },
    { name: "AI Assistant", href: "/chat" },
    { name: "Compliance", href: "/compliance" },
    { name: "Risk Management", href: "/risk" },
    { name: "Governance", href: "/governance" },
  ]

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-2">
          <Link href="/" className="flex items-center gap-2">
            <Shield className="h-6 w-6 text-accent" />
            <span className="hidden font-bold sm:inline-block">GRC AI Platform</span>
          </Link>
        </div>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-6">
          {navigation.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "text-sm font-medium transition-colors hover:text-accent",
                pathname === item.href ? "text-accent" : "text-muted-foreground",
              )}
            >
              {item.name}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <ThemeToggle />
          <Button asChild variant="default" size="sm" className="hidden md:flex">
            <Link href="/login">Sign In</Link>
          </Button>
          <Button variant="ghost" size="icon" className="md:hidden" onClick={toggleMenu}>
            {isMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
      </div>

      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="md:hidden container py-4 border-t">
          <nav className="flex flex-col space-y-4">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "text-sm font-medium transition-colors hover:text-accent",
                  pathname === item.href ? "text-accent" : "text-muted-foreground",
                )}
                onClick={() => setIsMenuOpen(false)}
              >
                {item.name}
              </Link>
            ))}
            <Button asChild variant="default" size="sm" className="w-full">
              <Link href="/login">Sign In</Link>
            </Button>
          </nav>
        </div>
      )}
    </header>
  )
}
