# Parsing Quality Evaluation Framework for RegulAIte

This framework evaluates how document parsing quality affects agent response quality in the RegulAIte system. It tests different agents with documents parsed using various methods and analyzes the impact on response quality.

## Overview

The framework tests responses from the following agents:
- Regulatory Agent (regulatory compliance)
- Research Agent (research and analysis)
- Tree Reasoning Agent (structured reasoning)
- Dynamic Tree Agent (dynamic decision trees)
- Vulnerability Assessment Agent
- Compliance Mapping Agent
- Threat Modeling Agent

Using documents parsed with:
- Doctly API
- LlamaParse API
- Unstructured API
- Unstructured local container

## Components

The framework consists of:

1. **Parsing Quality Evaluation** (`parsing_quality_evaluation.py`): The main script that runs the evaluation tests.
2. **Test Cases** (`test_cases.py`): Predefined test queries and expected content elements.
3. **Visualization Tools** (`visualization_tools.py`): Tools for analyzing and visualizing results.

## Installation Requirements

```bash
pip install pandas matplotlib seaborn numpy openai
```

## Usage

### Running the Evaluation

To run the evaluation with all agents, parsing methods, and query categories:

```bash
python parsing_quality_evaluation.py
```

To test specific agents, parsing methods, or categories:

```bash
python parsing_quality_evaluation.py --agents regulatory compliance_mapping --parsing-methods doctly llamaparse --categories regulatory risk_assessment
```

### Command Line Arguments

- `--agents`: Agent types to test (default: all)
- `--parsing-methods`: Parsing methods to test (default: all)
- `--categories`: Query categories to test (default: all)
- `--output-dir`: Directory to save evaluation results (default: ./evaluation_results)
- `--agent-model`: Model to use for agents (default: gpt-4)
- `--evaluator-model`: Model to use for evaluation (default: gpt-4)
- `--verbose`: Enable verbose logging

### Visualizing Results

After running the evaluation, you can visualize the results:

```bash
python visualization_tools.py --results evaluation_results/results_YYYYMMDD_HHMMSS.json --output visualization_results
```

## How It Works

1. **Setup**: The framework initializes the RAG system, agents, and evaluation metrics.
2. **Testing**: For each combination of agent, parsing method, and query:
   - Configures the RAG system to use the specified parsing method
   - Creates and runs the agent with the test query
   - Records response time and content
3. **Evaluation**: Uses an LLM-based evaluator to assess responses on:
   - Accuracy (0-3 points)
   - Relevance (0-2 points)
   - Completeness (0-2 points)
   - Clarity (0-1 point)
   - Actionability (0-2 points)
4. **Analysis**: Generates statistical reports and visualizations to compare:
   - Performance by parsing method
   - Performance by agent type
   - Performance by query category
   - Best/worst combinations

## Output Files

The framework generates:

1. **Results JSON**: Complete test results with queries, responses, and scores
2. **Results CSV**: CSV format of the test results for import into other tools
3. **Summary JSON**: Summary statistics of the evaluation
4. **Visualizations**: Charts showing performance comparisons

## Customizing the Evaluation

### Adding Test Cases

Edit `test_cases.py` to add new test queries and expected response elements:

```python
TEST_CASES = {
    "category_name": [
        {
            "query": "Your test query",
            "expected_elements": [
                "Expected element 1",
                "Expected element 2"
            ],
            "negative_elements": [
                "Problematic element 1",
                "Problematic element 2"
            ]
        }
    ]
}
```

### Adjusting Evaluation Criteria

Edit the `EVALUATION_CRITERIA` dictionary in `test_cases.py` to change the weights for different evaluation aspects.

## Integration with RegulAIte

The framework integrates with the existing RegulAIte system by:

1. Using the same RAG system for retrieving context
2. Creating agents using the same factory methods
3. Filtering by parsing method for response comparison

## Technical Implementation Details

- **EvaluationAgent**: An AI agent that evaluates response quality based on predefined criteria
- **TestRunner**: Handles test execution, agent creation, and result collection
- **EvaluationMetrics**: Collects and stores test results
- **EvaluationVisualizer**: Creates visualizations and statistical reports

## Example Workflow

1. Set up the test environment and required dependencies
2. Run the evaluation with default settings
3. Review the summary report to identify trends
4. Generate visualizations to communicate findings
5. Drill down into specific agent/parsing method combinations
6. Make data-driven decisions about which parsing methods to use

## Troubleshooting

- If agents fail to retrieve context, verify that documents are properly indexed in Qdrant
- If evaluation fails, check that the OpenAI API key is properly configured
- For visualization errors, ensure all required Python packages are installed

## License

This framework is part of the RegulAIte system and follows the same licensing terms.

## Contributing

To contribute to the testing framework:
1. Add new test cases to improve coverage
2. Enhance visualization capabilities
3. Implement additional evaluation metrics
4. Create new agent types for testing 