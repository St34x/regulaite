import Link from "next/link"
import { Shield } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function NotFound() {
  return (
    <div className="container flex h-[calc(100vh-9rem)] flex-col items-center justify-center">
      <Shield className="h-16 w-16 text-accent mb-4" />
      <h1 className="text-4xl font-bold">404</h1>
      <p className="mt-2 text-lg text-muted-foreground">The page you're looking for doesn't exist.</p>
      <Button asChild className="mt-8">
        <Link href="/">Return to Dashboard</Link>
      </Button>
    </div>
  )
}
