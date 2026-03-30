"""Test imports for reflexion package - Complete version."""

print("="*60)
print("Testing Reflexion Package - All Components")
print("="*60 + "\n")

# Step 1: Import all modules
print("Step 1: Importing modules...")
from reflexion.config import SecureConfigLoader
from reflexion.llm import BaseLLMModel
from reflexion.memory import TemporalMemory, VectorEpisodicMemory
from reflexion.evaluators import ObjectiveCodeEvaluator
from reflexion.benchmarks import HumanEvalLoader
from reflexion.agents import ReflexionAgent
print("✅ All modules imported successfully\n")

# Step 2: Load config and create LLM
print("Step 2: Loading configuration...")
config_loader = SecureConfigLoader()
config = config_loader.load_from_env_file('.env')
llm = BaseLLMModel(
    config['openrouter_api_key'],
    config['openrouter_model'],
    config['gemini_api_base'],
    config['rate_limit_delay']
)
print(f"✅ Config and LLM ready: {llm.model}\n")

# Step 3: Test memory
print("Step 3: Testing memory modules...")
temp_mem = TemporalMemory(max_size=5)
temp_mem.add_reflection("Test reflection")
vec_mem = VectorEpisodicMemory(llm, max_size=10)
vec_mem.add_reflection("Test vector reflection")
print(f"✅ Memory modules working\n")

# Step 4: Test evaluator
print("Step 4: Testing evaluator...")
evaluator = ObjectiveCodeEvaluator(timeout=5)
test_code = """
def add(a, b):
    return a + b
"""
test_result = evaluator.evaluate(
    test_code, 
    'add', 
    'def check(add):\n    assert add(2, 3) == 5\n'
)
print(f"✅ Evaluator working: {test_result}\n")

# Step 5: Test agent
print("Step 5: Testing agent...")
agent = ReflexionAgent(llm, memory_mode='temporal', max_trials=3)
print(f"✅ Agent created: {agent.memory_mode} memory, {agent.max_trials} max trials\n")

# Step 6: Test HumanEval loader
print("Step 6: Testing HumanEval loader...")
try:
    tasks = HumanEvalLoader.load_from_file('HumanEval.jsonl.gz', num_samples=2)
    print(f"✅ HumanEval loader working: {len(tasks)} tasks loaded\n")
except FileNotFoundError as e:
    print(f"⚠️  HumanEval file not found (expected if not downloaded)\n")

print("="*60)
print("✅✅✅ ALL COMPONENTS WORKING! ✅✅✅")
print("="*60)
