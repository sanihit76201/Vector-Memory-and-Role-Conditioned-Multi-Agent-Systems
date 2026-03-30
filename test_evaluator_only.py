# test_evaluator_only.py
print("Testing evaluator import...\n")

try:
    from reflexion.evaluators.code import ObjectiveCodeEvaluator
    print("✅ Direct import from code.py works")
    
    evaluator = ObjectiveCodeEvaluator(timeout=5)
    print(f"✅ Evaluator instance created: timeout={evaluator.timeout}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nNow testing package import...")

try:
    from reflexion.evaluators import ObjectiveCodeEvaluator
    print("✅ Package import works")
except Exception as e:
    print(f"❌ Package import failed: {e}")
    import traceback
    traceback.print_exc()
