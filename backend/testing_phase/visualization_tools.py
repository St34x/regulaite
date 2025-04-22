"""
Visualization Tools for Parsing Quality Evaluation
=================================================

This module provides tools for visualizing and analyzing the results
of the parsing quality evaluation tests. It creates various charts,
graphs, and statistical analyses to help understand how parsing quality
affects agent response quality.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class EvaluationVisualizer:
    """Creates visualizations for parsing quality evaluation results"""
    
    def __init__(self, 
                results_file: str = None, 
                results_data: List[Dict[str, Any]] = None,
                output_dir: str = "./visualization_results"):
        """
        Initialize the visualizer with either a results file or data.
        
        Args:
            results_file: Path to JSON results file
            results_data: List of result dictionaries
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if results_file and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.results = json.load(f)
        elif results_data:
            self.results = results_data
        else:
            self.results = []
            
        self.df = pd.DataFrame(self.results) if self.results else pd.DataFrame()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up visualization style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        self.colors = sns.color_palette("viridis", n_colors=4)
        
    def create_all_visualizations(self) -> List[str]:
        """
        Create all available visualizations and return file paths.
        
        Returns:
            List of file paths to created visualizations
        """
        if self.df.empty:
            print("No data available for visualization")
            return []
            
        visualizations = []
        
        # Create individual visualizations
        visualizations.append(self.create_heatmap())
        visualizations.append(self.create_parsing_comparison())
        visualizations.append(self.create_agent_comparison())
        visualizations.append(self.create_category_comparison())
        visualizations.append(self.create_response_time_analysis())
        visualizations.append(self.create_score_distribution())
        visualizations.append(self.create_best_worst_analysis())
        
        # Create summary dashboard
        visualizations.append(self.create_summary_dashboard())
        
        return [v for v in visualizations if v]  # Filter out None values
        
    def create_heatmap(self) -> Optional[str]:
        """
        Create a heatmap showing average scores by agent and parsing method.
        
        Returns:
            Path to saved visualization
        """
        if 'agent_type' not in self.df.columns or 'parsing_method' not in self.df.columns:
            return None
            
        try:
            plt.figure(figsize=(12, 10))
            
            # Create pivot table
            heatmap_data = self.df.pivot_table(
                values="evaluation_score", 
                index="agent_type", 
                columns="parsing_method", 
                aggfunc="mean"
            )
            
            # Create heatmap
            ax = sns.heatmap(
                heatmap_data, 
                annot=True, 
                cmap="YlGnBu", 
                fmt=".2f",
                linewidths=.5,
                cbar_kws={"label": "Average Score (0-10)"}
            )
            
            plt.title("Agent Performance by Parsing Method", fontsize=16)
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"heatmap_agent_parsing_{self.timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating heatmap: {str(e)}")
            return None
    
    def create_parsing_comparison(self) -> Optional[str]:
        """
        Create a bar chart comparing performance across parsing methods.
        
        Returns:
            Path to saved visualization
        """
        if 'parsing_method' not in self.df.columns:
            return None
            
        try:
            plt.figure(figsize=(12, 7))
            
            # Calculate mean scores by parsing method
            parsing_scores = self.df.groupby("parsing_method")["evaluation_score"].mean().reset_index()
            parsing_scores = parsing_scores.sort_values("evaluation_score", ascending=False)
            
            # Create bar chart
            ax = sns.barplot(
                x="parsing_method", 
                y="evaluation_score", 
                data=parsing_scores, 
                palette="Blues_d",
                hue="parsing_method",
                legend=False
            )
            
            # Add data labels
            for i, row in enumerate(parsing_scores.itertuples()):
                ax.text(
                    i, 
                    row.evaluation_score + 0.1, 
                    f"{row.evaluation_score:.2f}", 
                    ha="center",
                    fontweight="bold"
                )
            
            plt.title("Average Score by Parsing Method", fontsize=16)
            plt.ylabel("Average Score (0-10)")
            plt.xlabel("Parsing Method")
            plt.ylim(0, 10.5)
            
            # Add standard deviation as error bars
            std_by_method = self.df.groupby("parsing_method")["evaluation_score"].std().reset_index()
            std_by_method = std_by_method.set_index("parsing_method")
            for i, method in enumerate(parsing_scores["parsing_method"]):
                std = std_by_method.loc[method, "evaluation_score"]
                plt.errorbar(i, parsing_scores.iloc[i]["evaluation_score"], yerr=std, fmt="none", color="black", capsize=5)
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"parsing_comparison_{self.timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating parsing comparison: {str(e)}")
            return None
    
    def create_agent_comparison(self) -> Optional[str]:
        """
        Create a bar chart comparing performance across agent types.
        
        Returns:
            Path to saved visualization
        """
        if 'agent_type' not in self.df.columns:
            return None
            
        try:
            plt.figure(figsize=(14, 7))
            
            # Calculate mean scores by agent type
            agent_scores = self.df.groupby("agent_type")["evaluation_score"].mean().reset_index()
            agent_scores = agent_scores.sort_values("evaluation_score", ascending=False)
            
            # Create bar chart
            ax = sns.barplot(
                x="agent_type", 
                y="evaluation_score", 
                data=agent_scores, 
                palette="Greens_d",
                hue="agent_type",
                legend=False
            )
            
            # Add data labels
            for i, row in enumerate(agent_scores.itertuples()):
                ax.text(
                    i, 
                    row.evaluation_score + 0.1, 
                    f"{row.evaluation_score:.2f}", 
                    ha="center",
                    fontweight="bold"
                )
            
            plt.title("Average Score by Agent Type", fontsize=16)
            plt.ylabel("Average Score (0-10)")
            plt.xlabel("Agent Type")
            plt.ylim(0, 10.5)
            plt.xticks(rotation=45, ha="right")
            
            # Add standard deviation as error bars
            std_by_agent = self.df.groupby("agent_type")["evaluation_score"].std().reset_index()
            std_by_agent = std_by_agent.set_index("agent_type")
            for i, agent in enumerate(agent_scores["agent_type"]):
                std = std_by_agent.loc[agent, "evaluation_score"]
                plt.errorbar(i, agent_scores.iloc[i]["evaluation_score"], yerr=std, fmt="none", color="black", capsize=5)
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"agent_comparison_{self.timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating agent comparison: {str(e)}")
            return None
    
    def create_category_comparison(self) -> Optional[str]:
        """
        Create a grouped bar chart comparing performance across query categories and parsing methods.
        
        Returns:
            Path to saved visualization
        """
        if 'query_category' not in self.df.columns or 'parsing_method' not in self.df.columns:
            return None
            
        try:
            plt.figure(figsize=(14, 8))
            
            # Calculate mean scores by category and parsing method
            category_scores = self.df.groupby(["query_category", "parsing_method"])["evaluation_score"].mean().reset_index()
            
            # Create grouped bar chart
            ax = sns.catplot(
                x="query_category", 
                y="evaluation_score", 
                hue="parsing_method", 
                data=category_scores, 
                kind="bar",
                palette="muted",
                height=6,
                aspect=1.5
            )
            
            ax.set_ylabels("Average Score (0-10)")
            ax.set_xlabels("Query Category")
            plt.title("Performance by Category and Parsing Method", fontsize=16)
            plt.ylim(0, 10)
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"category_comparison_{self.timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating category comparison: {str(e)}")
            return None
    
    def create_response_time_analysis(self) -> Optional[str]:
        """
        Create visualizations comparing response time across agents and parsing methods.
        
        Returns:
            Path to saved visualization
        """
        if 'response_time' not in self.df.columns:
            return None
            
        try:
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            
            # Response time by agent type
            agent_times = self.df.groupby("agent_type")["response_time"].mean().reset_index()
            agent_times = agent_times.sort_values("response_time")
            
            sns.barplot(
                x="response_time", 
                y="agent_type", 
                data=agent_times, 
                palette="Oranges_d",
                ax=axes[0]
            )
            
            axes[0].set_title("Average Response Time by Agent Type", fontsize=14)
            axes[0].set_xlabel("Response Time (seconds)")
            axes[0].set_ylabel("Agent Type")
            
            # Response time by parsing method
            parsing_times = self.df.groupby("parsing_method")["response_time"].mean().reset_index()
            parsing_times = parsing_times.sort_values("response_time")
            
            sns.barplot(
                x="response_time", 
                y="parsing_method", 
                data=parsing_times, 
                palette="Purples_d",
                ax=axes[1]
            )
            
            axes[1].set_title("Average Response Time by Parsing Method", fontsize=14)
            axes[1].set_xlabel("Response Time (seconds)")
            axes[1].set_ylabel("Parsing Method")
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"response_time_analysis_{self.timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating response time analysis: {str(e)}")
            return None
    
    def create_score_distribution(self) -> Optional[str]:
        """
        Create a boxplot showing the distribution of scores by parsing method.
        
        Returns:
            Path to saved visualization
        """
        if 'evaluation_score' not in self.df.columns or 'parsing_method' not in self.df.columns:
            return None
            
        try:
            plt.figure(figsize=(12, 7))
            
            # Create boxplot
            ax = sns.boxplot(
                x="parsing_method", 
                y="evaluation_score", 
                data=self.df,
                palette="Set3"
            )
            
            # Add a swarmplot to show individual points
            ax = sns.swarmplot(
                x="parsing_method", 
                y="evaluation_score", 
                data=self.df,
                color="0.25", 
                alpha=0.5
            )
            
            plt.title("Distribution of Evaluation Scores by Parsing Method", fontsize=16)
            plt.ylabel("Score (0-10)")
            plt.xlabel("Parsing Method")
            plt.ylim(0, 10.5)
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"score_distribution_{self.timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating score distribution: {str(e)}")
            return None
    
    def create_best_worst_analysis(self) -> Optional[str]:
        """
        Create a visualization showing the best and worst agent-parsing combinations.
        
        Returns:
            Path to saved visualization
        """
        if 'agent_type' not in self.df.columns or 'parsing_method' not in self.df.columns:
            return None
            
        try:
            # Get top and bottom combinations
            combo_scores = self.df.groupby(["agent_type", "parsing_method"])["evaluation_score"].mean().reset_index()
            top_combos = combo_scores.nlargest(5, "evaluation_score")
            bottom_combos = combo_scores.nsmallest(5, "evaluation_score")
            
            # Combine for visualization
            combined = pd.concat([top_combos, bottom_combos])
            combined["combination"] = combined["agent_type"] + " + " + combined["parsing_method"]
            combined["rank_group"] = ["Top 5" if i < 5 else "Bottom 5" for i in range(len(combined))]
            
            plt.figure(figsize=(14, 8))
            
            # Create bar chart
            ax = sns.barplot(
                x="evaluation_score", 
                y="combination", 
                hue="rank_group",
                data=combined, 
                palette={"Top 5": "darkgreen", "Bottom 5": "darkred"}
            )
            
            # Add data labels
            for i, row in enumerate(combined.itertuples()):
                ax.text(
                    row.evaluation_score + 0.1, 
                    i, 
                    f"{row.evaluation_score:.2f}", 
                    va="center",
                    fontweight="bold"
                )
            
            plt.title("Best and Worst Agent-Parsing Method Combinations", fontsize=16)
            plt.xlabel("Average Score (0-10)")
            plt.ylabel("")
            plt.xlim(0, 10.5)
            plt.legend(title="Ranking")
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"best_worst_analysis_{self.timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating best/worst analysis: {str(e)}")
            return None
    
    def create_summary_dashboard(self) -> Optional[str]:
        """
        Create a comprehensive dashboard with key metrics.
        
        Returns:
            Path to saved visualization
        """
        try:
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(20, 20))
            
            # Define grid layout
            gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
            
            # Add title
            fig.suptitle("Parsing Quality Evaluation Results", fontsize=24, y=0.98)
            
            # 1. Overall results by parsing method (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            parsing_scores = self.df.groupby("parsing_method")["evaluation_score"].mean().reset_index()
            parsing_scores = parsing_scores.sort_values("evaluation_score", ascending=False)
            sns.barplot(x="parsing_method", y="evaluation_score", data=parsing_scores, ax=ax1, palette="Blues_d")
            ax1.set_title("Average Score by Parsing Method", fontsize=14)
            ax1.set_ylim(0, 10)
            ax1.set_xlabel("")
            ax1.set_ylabel("Score (0-10)")
            
            # 2. Overall results by agent type (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            agent_scores = self.df.groupby("agent_type")["evaluation_score"].mean().reset_index()
            agent_scores = agent_scores.sort_values("evaluation_score", ascending=False)
            sns.barplot(x="agent_type", y="evaluation_score", data=agent_scores, ax=ax2, palette="Greens_d")
            ax2.set_title("Average Score by Agent Type", fontsize=14)
            ax2.set_ylim(0, 10)
            ax2.set_xlabel("")
            ax2.set_ylabel("Score (0-10)")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
            
            # 3. Heat map of agent-parsing combinations (middle left span)
            ax3 = fig.add_subplot(gs[1, :])
            heatmap_data = self.df.pivot_table(
                values="evaluation_score", 
                index="agent_type", 
                columns="parsing_method", 
                aggfunc="mean"
            )
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax3, cbar_kws={"label": "Score (0-10)"})
            ax3.set_title("Agent Performance by Parsing Method", fontsize=14)
            
            # 4. Score distribution (lower left)
            ax4 = fig.add_subplot(gs[2, 0])
            sns.boxplot(x="parsing_method", y="evaluation_score", data=self.df, ax=ax4, palette="Set3")
            ax4.set_title("Score Distribution by Parsing Method", fontsize=14)
            ax4.set_ylim(0, 10)
            ax4.set_xlabel("")
            ax4.set_ylabel("Score (0-10)")
            
            # 5. Category performance (lower right)
            ax5 = fig.add_subplot(gs[2, 1])
            category_scores = self.df.groupby("query_category")["evaluation_score"].mean().reset_index()
            category_scores = category_scores.sort_values("evaluation_score", ascending=False)
            sns.barplot(x="query_category", y="evaluation_score", data=category_scores, ax=ax5, palette="Oranges_d")
            ax5.set_title("Average Score by Query Category", fontsize=14)
            ax5.set_ylim(0, 10)
            ax5.set_xlabel("")
            ax5.set_ylabel("Score (0-10)")
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha="right")
            
            # 6. Response times (bottom)
            ax6 = fig.add_subplot(gs[3, :])
            
            # Combine agent and parsing method response times
            agent_times = self.df.groupby("agent_type")["response_time"].mean().reset_index()
            agent_times["group"] = "Agent"
            agent_times = agent_times.rename(columns={"agent_type": "entity"})
            
            parsing_times = self.df.groupby("parsing_method")["response_time"].mean().reset_index()
            parsing_times["group"] = "Parsing Method"
            parsing_times = parsing_times.rename(columns={"parsing_method": "entity"})
            
            combined_times = pd.concat([agent_times, parsing_times])
            
            # Create grouped bar chart
            sns.barplot(x="entity", y="response_time", hue="group", data=combined_times, ax=ax6, palette="Set1")
            ax6.set_title("Average Response Time (seconds)", fontsize=14)
            ax6.set_xlabel("")
            ax6.set_ylabel("Time (seconds)")
            ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha="right")
            
            # Add key statistics as text
            stats_text = (
                f"Total Tests: {len(self.df)}\n"
                f"Best Parsing Method: {parsing_scores.iloc[0]['parsing_method']} "
                f"({parsing_scores.iloc[0]['evaluation_score']:.2f})\n"
                f"Best Agent: {agent_scores.iloc[0]['agent_type']} "
                f"({agent_scores.iloc[0]['evaluation_score']:.2f})\n"
            )
            
            # Add text box with key stats
            plt.figtext(0.5, 0.02, stats_text, ha="center", fontsize=12, 
                       bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"summary_dashboard_{self.timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating summary dashboard: {str(e)}")
            return None
    
    def generate_statistical_report(self) -> Dict[str, Any]:
        """
        Generate a statistical report of the evaluation results.
        
        Returns:
            Dictionary with statistical measures
        """
        if self.df.empty:
            return {"error": "No data available for statistical analysis"}
            
        try:
            report = {
                "total_tests": len(self.df),
                "timestamp": datetime.now().isoformat(),
                "overall_statistics": {
                    "mean_score": float(self.df["evaluation_score"].mean()),
                    "median_score": float(self.df["evaluation_score"].median()),
                    "std_dev": float(self.df["evaluation_score"].std()),
                    "min_score": float(self.df["evaluation_score"].min()),
                    "max_score": float(self.df["evaluation_score"].max()),
                },
                "by_parsing_method": {},
                "by_agent_type": {},
                "by_query_category": {},
                "top_combinations": [],
                "correlation_analysis": {}
            }
            
            # Statistics by parsing method
            for method in self.df["parsing_method"].unique():
                subset = self.df[self.df["parsing_method"] == method]
                report["by_parsing_method"][method] = {
                    "mean_score": float(subset["evaluation_score"].mean()),
                    "median_score": float(subset["evaluation_score"].median()),
                    "std_dev": float(subset["evaluation_score"].std()),
                    "sample_size": int(len(subset)),
                    "confidence_interval": [
                        float(subset["evaluation_score"].mean() - 1.96 * subset["evaluation_score"].std() / np.sqrt(len(subset))),
                        float(subset["evaluation_score"].mean() + 1.96 * subset["evaluation_score"].std() / np.sqrt(len(subset)))
                    ]
                }
            
            # Statistics by agent type
            for agent in self.df["agent_type"].unique():
                subset = self.df[self.df["agent_type"] == agent]
                report["by_agent_type"][agent] = {
                    "mean_score": float(subset["evaluation_score"].mean()),
                    "median_score": float(subset["evaluation_score"].median()),
                    "std_dev": float(subset["evaluation_score"].std()),
                    "sample_size": int(len(subset)),
                    "confidence_interval": [
                        float(subset["evaluation_score"].mean() - 1.96 * subset["evaluation_score"].std() / np.sqrt(len(subset))),
                        float(subset["evaluation_score"].mean() + 1.96 * subset["evaluation_score"].std() / np.sqrt(len(subset)))
                    ]
                }
            
            # Statistics by query category
            for category in self.df["query_category"].unique():
                subset = self.df[self.df["query_category"] == category]
                report["by_query_category"][category] = {
                    "mean_score": float(subset["evaluation_score"].mean()),
                    "median_score": float(subset["evaluation_score"].median()),
                    "std_dev": float(subset["evaluation_score"].std()),
                    "sample_size": int(len(subset)),
                    "confidence_interval": [
                        float(subset["evaluation_score"].mean() - 1.96 * subset["evaluation_score"].std() / np.sqrt(len(subset))),
                        float(subset["evaluation_score"].mean() + 1.96 * subset["evaluation_score"].std() / np.sqrt(len(subset)))
                    ]
                }
            
            # Top agent-parsing combinations
            combos = self.df.groupby(["agent_type", "parsing_method"])["evaluation_score"].mean().reset_index()
            top_10 = combos.nlargest(10, "evaluation_score")
            for _, row in top_10.iterrows():
                report["top_combinations"].append({
                    "agent_type": row["agent_type"],
                    "parsing_method": row["parsing_method"],
                    "mean_score": float(row["evaluation_score"])
                })
            
            # Correlation between response time and score
            if "response_time" in self.df.columns:
                correlation = self.df["response_time"].corr(self.df["evaluation_score"])
                report["correlation_analysis"]["response_time_vs_score"] = float(correlation)
            
            # Save report to file
            report_file = os.path.join(self.output_dir, f"statistical_report_{self.timestamp}.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
        except Exception as e:
            print(f"Error generating statistical report: {str(e)}")
            return {"error": str(e)}

def main():
    """Main function to demonstrate usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize parsing quality evaluation results")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--output", default="./visualization_results", help="Output directory for visualizations")
    args = parser.parse_args()
    
    visualizer = EvaluationVisualizer(results_file=args.results, output_dir=args.output)
    visualization_files = visualizer.create_all_visualizations()
    
    for file in visualization_files:
        print(f"Created visualization: {file}")
    
    stats_report = visualizer.generate_statistical_report()
    print(f"Statistical report generated with {len(stats_report.keys())} sections")

if __name__ == "__main__":
    main() 