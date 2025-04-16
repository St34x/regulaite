import Link from "next/link"
import { Shield } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t bg-background">
      <div className="container py-8 md:py-12">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          <div className="flex flex-col gap-2">
            <Link href="/" className="flex items-center gap-2">
              <Shield className="h-6 w-6 text-accent" />
              <span className="font-bold">GRC AI Platform</span>
            </Link>
            <p className="text-sm text-muted-foreground">
              Simplifying governance, risk, and compliance with AI-powered insights.
            </p>
          </div>
          <div>
            <h3 className="text-sm font-medium">Platform</h3>
            <ul className="mt-4 space-y-2 text-sm">
              <li>
                <Link href="/compliance" className="text-muted-foreground hover:text-accent">
                  Compliance
                </Link>
              </li>
              <li>
                <Link href="/risk" className="text-muted-foreground hover:text-accent">
                  Risk Management
                </Link>
              </li>
              <li>
                <Link href="/governance" className="text-muted-foreground hover:text-accent">
                  Governance
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-medium">Resources</h3>
            <ul className="mt-4 space-y-2 text-sm">
              <li>
                <Link href="/documentation" className="text-muted-foreground hover:text-accent">
                  Documentation
                </Link>
              </li>
              <li>
                <Link href="/guides" className="text-muted-foreground hover:text-accent">
                  Guides
                </Link>
              </li>
              <li>
                <Link href="/support" className="text-muted-foreground hover:text-accent">
                  Support
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-medium">Legal</h3>
            <ul className="mt-4 space-y-2 text-sm">
              <li>
                <Link href="/privacy" className="text-muted-foreground hover:text-accent">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/terms" className="text-muted-foreground hover:text-accent">
                  Terms of Service
                </Link>
              </li>
              <li>
                <Link href="/security" className="text-muted-foreground hover:text-accent">
                  Security
                </Link>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-8 border-t pt-8 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} GRC AI Platform. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}
