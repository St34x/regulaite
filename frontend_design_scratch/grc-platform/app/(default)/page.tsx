import { CardContent } from "@/components/ui/card"
import { CardTitle } from "@/components/ui/card"
import { CardHeader } from "@/components/ui/card"
import { Card } from "@/components/ui/card"
import { AlertTriangle, CheckCircle2, Shield, ShieldAlert, ShieldCheck } from "lucide-react"
import { MetricCard } from "@/components/dashboard/metric-card"
import { ComplianceChart } from "@/components/dashboard/compliance-chart"
import { RiskChart } from "@/components/dashboard/risk-chart"
import { RecentAlerts } from "@/components/dashboard/recent-alerts"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export default function Home() {
  return (
    <div className="container py-8">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">Your GRC overview and key metrics at a glance.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button asChild>
            <Link href="/chat">
              <Shield className="mr-2 h-4 w-4" />
              Ask AI Assistant
            </Link>
          </Button>
        </div>
      </div>

      <div className="mt-8 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Compliance Score"
          value="85%"
          description="Overall compliance across all frameworks"
          icon={<ShieldCheck />}
          trend={{ value: 5, isPositive: true }}
        />
        <MetricCard
          title="Active Risks"
          value="24"
          description="Identified risks requiring attention"
          icon={<ShieldAlert />}
          trend={{ value: 3, isPositive: false }}
        />
        <MetricCard
          title="Controls Implemented"
          value="156/180"
          description="Security controls in place"
          icon={<CheckCircle2 />}
          trend={{ value: 8, isPositive: true }}
        />
        <MetricCard
          title="Pending Tasks"
          value="12"
          description="Tasks requiring action"
          icon={<AlertTriangle />}
          trend={{ value: 2, isPositive: true }}
        />
      </div>

      <div className="mt-8 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <ComplianceChart />
        <RiskChart />
      </div>

      <div className="mt-8 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <RecentAlerts />
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Upcoming Deadlines</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="rounded-md border p-4">
                <div className="flex items-center justify-between">
                  <div className="font-medium">SOC 2 Audit</div>
                  <div className="text-sm text-destructive">5 days left</div>
                </div>
                <div className="mt-1 text-sm text-muted-foreground">Annual SOC 2 Type II audit preparation</div>
              </div>
              <div className="rounded-md border p-4">
                <div className="flex items-center justify-between">
                  <div className="font-medium">GDPR Review</div>
                  <div className="text-sm text-amber-500">12 days left</div>
                </div>
                <div className="mt-1 text-sm text-muted-foreground">Quarterly GDPR compliance review</div>
              </div>
              <div className="rounded-md border p-4">
                <div className="flex items-center justify-between">
                  <div className="font-medium">Risk Assessment</div>
                  <div className="text-sm text-accent">20 days left</div>
                </div>
                <div className="mt-1 text-sm text-muted-foreground">Quarterly risk assessment update</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
