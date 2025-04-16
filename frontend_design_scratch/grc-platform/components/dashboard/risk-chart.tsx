"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Cell, Legend, Pie, PieChart, ResponsiveContainer } from "recharts"

const data = [
  { name: "High", value: 15, color: "hsl(var(--destructive))" },
  { name: "Medium", value: 30, color: "hsl(38, 92%, 50%)" },
  { name: "Low", value: 55, color: "hsl(var(--accent))" },
]

export function RiskChart() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Risk Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            high: {
              label: "High",
              color: "hsl(var(--destructive))",
            },
            medium: {
              label: "Medium",
              color: "hsl(38, 92%, 50%)",
            },
            low: {
              label: "Low",
              color: "hsl(var(--accent))",
            },
          }}
          className="h-[300px]"
        >
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <ChartTooltip content={<ChartTooltipContent />} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
