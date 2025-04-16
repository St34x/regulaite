"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  {
    name: "Jan",
    compliant: 65,
    nonCompliant: 35,
  },
  {
    name: "Feb",
    compliant: 70,
    nonCompliant: 30,
  },
  {
    name: "Mar",
    compliant: 75,
    nonCompliant: 25,
  },
  {
    name: "Apr",
    compliant: 80,
    nonCompliant: 20,
  },
  {
    name: "May",
    compliant: 85,
    nonCompliant: 15,
  },
  {
    name: "Jun",
    compliant: 90,
    nonCompliant: 10,
  },
]

export function ComplianceChart() {
  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle>Compliance Trend</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            compliant: {
              label: "Compliant",
              color: "hsl(var(--accent))",
            },
            nonCompliant: {
              label: "Non-Compliant",
              color: "hsl(var(--destructive))",
            },
          }}
          className="h-[300px]"
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Legend />
              <Bar dataKey="compliant" stackId="a" fill="var(--color-compliant)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="nonCompliant" stackId="a" fill="var(--color-nonCompliant)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
