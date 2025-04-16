import { AlertTriangle, CheckCircle2, Clock } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

const alerts = [
  {
    id: 1,
    title: "PCI DSS Compliance Alert",
    description: "3 controls need review before the deadline",
    status: "warning",
    time: "2 hours ago",
  },
  {
    id: 2,
    title: "GDPR Data Processing",
    description: "New data processing activity detected",
    status: "info",
    time: "5 hours ago",
  },
  {
    id: 3,
    title: "SOC 2 Audit Preparation",
    description: "All controls reviewed successfully",
    status: "success",
    time: "1 day ago",
  },
  {
    id: 4,
    title: "Risk Assessment Update",
    description: "Quarterly risk assessment due in 5 days",
    status: "warning",
    time: "2 days ago",
  },
]

export function RecentAlerts() {
  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle>Recent Alerts</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {alerts.map((alert) => (
            <div key={alert.id} className="flex items-start space-x-4 rounded-md border p-4">
              <div
                className={cn(
                  "mt-0.5 rounded-full p-1",
                  alert.status === "warning" && "bg-amber-100 text-amber-600 dark:bg-amber-900/20 dark:text-amber-500",
                  alert.status === "success" && "bg-green-100 text-green-600 dark:bg-green-900/20 dark:text-green-500",
                  alert.status === "info" && "bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-blue-500",
                )}
              >
                {alert.status === "warning" && <AlertTriangle className="h-4 w-4" />}
                {alert.status === "success" && <CheckCircle2 className="h-4 w-4" />}
                {alert.status === "info" && <Clock className="h-4 w-4" />}
              </div>
              <div className="flex-1 space-y-1">
                <p className="font-medium leading-none">{alert.title}</p>
                <p className="text-sm text-muted-foreground">{alert.description}</p>
              </div>
              <div className="text-xs text-muted-foreground">{alert.time}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
