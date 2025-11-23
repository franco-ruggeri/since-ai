"""
Example usage of the Model Orchestrator.
Demonstrates how to integrate with Featherless AI.
"""

import os
from typing import Optional
from orchestrator import ModelOrchestrator
from model_registry import TaskType


def example_basic_usage():
    """Basic example of model selection"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Model Selection")
    print("=" * 60)

    orchestrator = ModelOrchestrator(default_priority="balanced")

    # Example prompts
    prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Explain quantum computing in simple terms",
        "Analyze this dataset and find trends in sales data",
        "Quick question: what is the capital of France?",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        result = orchestrator.select_model(prompt)
        print(f"Selected Model: {result.model.name} ({result.model.model_id})")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Estimated Cost: ${result.estimated_cost:.6f}")
        print(f"Estimated Latency: {result.estimated_latency}")
        print("-" * 60)


def example_priority_optimization():
    """Example showing different priority optimizations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Priority-Based Selection")
    print("=" * 60)

    orchestrator = ModelOrchestrator()
    prompt = "Write a comprehensive analysis of climate change impacts"

    priorities = ["speed", "cost", "quality", "balanced"]

    for priority in priorities:
        result = orchestrator.select_model(prompt, priority=priority)
        print(f"\nPriority: {priority.upper()}")
        print(f"Selected: {result.model.name}")
        print(f"Cost/1k tokens: ${result.model.cost_per_1k_tokens:.4f}")
        print(f"Speed Tier: {result.model.speed_tier}/5")
        print(f"Estimated Latency: {result.estimated_latency}")


def example_with_constraints():
    """Example with cost and quality constraints"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Selection with Constraints")
    print("=" * 60)

    orchestrator = ModelOrchestrator()
    prompt = "Generate a complex data analysis report with visualizations"

    # With cost constraint
    result = orchestrator.select_model(
        prompt,
        max_cost_per_1k=0.0005,
        required_capabilities=["json_mode"]
    )
    print(f"\nWith max cost $0.0005/1k tokens:")
    print(f"Selected: {result.model.name}")
    print(f"Cost: ${result.model.cost_per_1k_tokens:.4f}/1k")

    # With quality constraint
    result = orchestrator.select_model(
        prompt,
        min_quality_score=0.85
    )
    print(f"\nWith min quality score 0.85:")
    print(f"Selected: {result.model.name}")
    print(f"Reasoning Score: {result.model.reasoning_score:.2f}")
    print(f"Instruction Following: {result.model.instruction_following_score:.2f}")


def example_task_recommendations():
    """Example of getting task-specific recommendations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Task-Specific Recommendations")
    print("=" * 60)

    orchestrator = ModelOrchestrator()

    tasks = [TaskType.CODE, TaskType.MATH, TaskType.DATA_ANALYSIS]

    for task in tasks:
        print(f"\nBest models for {task.value.upper()}:")
        recommendations = orchestrator.get_model_recommendations(task, max_results=3)
        for i, model in enumerate(recommendations, 1):
            print(f"{i}. {model.name} (${model.cost_per_1k_tokens:.4f}/1k)")


def example_with_featherless_api():
    """Example integration with Featherless AI API"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Integration with Featherless AI")
    print("=" * 60)

    # This would be used in your actual application
    def call_featherless_api(model_id: str, prompt: str, api_key: str):
        """
        Pseudo-code for calling Featherless AI API.
        Replace with actual API call using langchain-featherless-ai
        """
        print(f"\nCalling Featherless AI with model: {model_id}")
        print(f"Prompt: {prompt[:50]}...")

        # Example using langchain
        # from langchain_featherless_ai import ChatFeatherlessAI
        #
        # llm = ChatFeatherlessAI(
        #     model=model_id,
        #     api_key=api_key,
        #     temperature=0.7
        # )
        # response = llm.invoke(prompt)
        # return response

        return "API response would be here"

    orchestrator = ModelOrchestrator()
    prompt = "Explain machine learning to a 10-year old"

    # Select best model
    result = orchestrator.select_model(prompt, priority="quality")

    print(f"Selected Model: {result.model.name}")
    print(f"Model ID for API: {result.model.model_id}")

    # Use the selected model with API
    api_key = os.getenv("FEATHERLESS_API_KEY", "your-api-key")
    response = call_featherless_api(result.model.model_id, prompt, api_key)

    print("\nAlternative models if needed:")
    for alt in result.alternative_models[:2]:
        print(f"  - {alt.name} ({alt.model_id})")


def example_statistics():
    """Example showing orchestrator statistics"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Orchestrator Statistics")
    print("=" * 60)

    orchestrator = ModelOrchestrator()

    # Make several selections
    test_prompts = [
        "Quick math: 2+2",
        "Write Python code for sorting",
        "Detailed analysis of market trends",
        "Simple question about history",
        "Complex reasoning task about ethics"
    ]

    for prompt in test_prompts:
        orchestrator.select_model(prompt)

    # Get statistics
    stats = orchestrator.get_statistics()
    print("\nOrchestrator Statistics:")
    print(f"Total Selections: {stats['total_selections']}")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")
    print(f"Total Estimated Cost: ${stats['total_estimated_cost']:.6f}")
    print("\nModel Usage:")
    for model, count in sorted(stats['model_usage'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {count} times")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_priority_optimization()
    example_with_constraints()
    example_task_recommendations()
    example_with_featherless_api()
    example_statistics()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
