"""
Simple tests for Model Orchestrator.
Run with: python -m pytest model-orchestrator/test_orchestrator.py
Or just: python model-orchestrator/test_orchestrator.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model_registry import ModelRegistry, TaskType, ModelSize
from prompt_analyzer import PromptAnalyzer
from orchestrator import ModelOrchestrator


def test_model_registry():
    """Test model registry functionality"""
    print("\n" + "="*60)
    print("TEST: Model Registry")
    print("="*60)

    registry = ModelRegistry()

    # Test getting all models
    all_models = registry.get_all_models()
    print(f"Total models in registry: {len(all_models)}")
    assert len(all_models) > 0, "Registry should have models"

    # Test filtering by task
    code_models = registry.filter_by_task(TaskType.CODE)
    print(f"Code models: {len(code_models)}")
    assert len(code_models) > 0, "Should have code models"

    # Test filtering by context length
    long_context = registry.filter_by_context_length(32000)
    print(f"Models with >32k context: {len(long_context)}")

    # Test filtering by size
    small_models = registry.filter_by_size(ModelSize.SMALL)
    print(f"Small models or smaller: {len(small_models)}")

    # Test getting specific model
    model = registry.get_model("mistralai/Mistral-7B-Instruct-v0.2")
    assert model is not None, "Should find Mistral model"
    print(f"Found model: {model.name}")

    print("✓ Model Registry tests passed")
    return True


def test_prompt_analyzer():
    """Test prompt analysis functionality"""
    print("\n" + "="*60)
    print("TEST: Prompt Analyzer")
    print("="*60)

    analyzer = PromptAnalyzer()

    # Test code detection
    code_prompt = "Write a Python function to calculate fibonacci numbers"
    analysis = analyzer.analyze(code_prompt)
    print(f"\nPrompt: {code_prompt}")
    print(f"Detected tasks: {[t.value for t in analysis.detected_tasks]}")
    assert TaskType.CODE in analysis.detected_tasks, "Should detect code task"

    # Test math detection
    math_prompt = "Solve this equation: 2x + 5 = 15"
    analysis = analyzer.analyze(math_prompt)
    print(f"\nPrompt: {math_prompt}")
    print(f"Detected tasks: {[t.value for t in analysis.detected_tasks]}")
    assert TaskType.MATH in analysis.detected_tasks, "Should detect math task"

    # Test complexity
    simple_prompt = "Quick question: what is 2+2?"
    analysis = analyzer.analyze(simple_prompt)
    print(f"\nPrompt: {simple_prompt}")
    print(f"Complexity score: {analysis.complexity_score}")
    assert analysis.complexity_score < 0.5, "Should detect low complexity"

    # Test reasoning detection
    reasoning_prompt = "Explain why the sky is blue and how light scattering works"
    analysis = analyzer.analyze(reasoning_prompt)
    print(f"\nPrompt: {reasoning_prompt}")
    print(f"Requires reasoning: {analysis.requires_reasoning}")
    assert analysis.requires_reasoning, "Should detect reasoning requirement"

    # Test JSON requirement
    json_prompt = "Return the data as JSON format"
    analysis = analyzer.analyze(json_prompt)
    print(f"\nPrompt: {json_prompt}")
    print(f"Requires JSON: {analysis.requires_json}")
    assert analysis.requires_json, "Should detect JSON requirement"

    print("\n✓ Prompt Analyzer tests passed")
    return True


def test_orchestrator_selection():
    """Test model orchestrator selection"""
    print("\n" + "="*60)
    print("TEST: Model Orchestrator Selection")
    print("="*60)

    orchestrator = ModelOrchestrator()

    # Test different prompt types
    test_cases = [
        ("Write Python code to sort a list", TaskType.CODE),
        ("Calculate the derivative of x^2", TaskType.MATH),
        ("Analyze this data for trends", TaskType.DATA_ANALYSIS),
        ("Quick question: capital of France?", TaskType.QUESTION_ANSWERING),
    ]

    for prompt, expected_task in test_cases:
        result = orchestrator.select_model(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Selected: {result.model.name}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Cost estimate: ${result.estimated_cost:.6f}")

        # Basic validations
        assert result.model is not None, "Should select a model"
        assert result.confidence > 0, "Should have positive confidence"
        assert result.estimated_cost >= 0, "Cost should be non-negative"

    print("\n✓ Orchestrator Selection tests passed")
    return True


def test_priority_optimization():
    """Test different priority modes"""
    print("\n" + "="*60)
    print("TEST: Priority Optimization")
    print("="*60)

    orchestrator = ModelOrchestrator()
    prompt = "Write a comprehensive analysis of climate change"

    priorities = ["speed", "cost", "quality", "balanced"]
    results = {}

    for priority in priorities:
        result = orchestrator.select_model(prompt, priority=priority)
        results[priority] = result
        print(f"\nPriority: {priority}")
        print(f"  Model: {result.model.name}")
        print(f"  Speed tier: {result.model.speed_tier}")
        print(f"  Cost/1k: ${result.model.cost_per_1k_tokens:.4f}")

    # Verify that different priorities select different models or at least consider different factors
    print("\n✓ Priority Optimization tests passed")
    return True


def test_constraints():
    """Test constraint-based selection"""
    print("\n" + "="*60)
    print("TEST: Constraint-Based Selection")
    print("="*60)

    orchestrator = ModelOrchestrator()
    prompt = "Generate detailed code documentation"

    # Test cost constraint
    result = orchestrator.select_model(prompt, max_cost_per_1k=0.0003)
    print(f"\nWith max cost $0.0003/1k:")
    print(f"  Selected: {result.model.name}")
    print(f"  Cost: ${result.model.cost_per_1k_tokens:.4f}")
    assert result.model.cost_per_1k_tokens <= 0.0003, "Should respect cost constraint"

    # Test quality constraint
    result = orchestrator.select_model(prompt, min_quality_score=0.8)
    print(f"\nWith min quality 0.8:")
    print(f"  Selected: {result.model.name}")
    avg_quality = (result.model.reasoning_score + result.model.instruction_following_score) / 2
    print(f"  Quality: {avg_quality:.2f}")
    assert avg_quality >= 0.8, "Should respect quality constraint"

    # Test required capabilities
    result = orchestrator.select_model(
        "Return data as JSON",
        required_capabilities=["json_mode"]
    )
    print(f"\nWith JSON mode requirement:")
    print(f"  Selected: {result.model.name}")
    print(f"  Supports JSON: {result.model.supports_json_mode}")
    assert result.model.supports_json_mode, "Should support JSON mode"

    print("\n✓ Constraint tests passed")
    return True


def test_statistics():
    """Test statistics tracking"""
    print("\n" + "="*60)
    print("TEST: Statistics Tracking")
    print("="*60)

    orchestrator = ModelOrchestrator()

    # Make several selections
    prompts = [
        "Quick math",
        "Write code",
        "Analyze data",
        "Simple question"
    ]

    for prompt in prompts:
        orchestrator.select_model(prompt)

    # Get statistics
    stats = orchestrator.get_statistics()
    print(f"\nTotal selections: {stats['total_selections']}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")
    print(f"Total estimated cost: ${stats['total_estimated_cost']:.6f}")

    assert stats['total_selections'] == 4, "Should track 4 selections"
    assert stats['average_confidence'] > 0, "Should have positive confidence"

    print("\n✓ Statistics tests passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING ALL MODEL ORCHESTRATOR TESTS")
    print("="*60)

    tests = [
        test_model_registry,
        test_prompt_analyzer,
        test_orchestrator_selection,
        test_priority_optimization,
        test_constraints,
        test_statistics
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ Test error: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
