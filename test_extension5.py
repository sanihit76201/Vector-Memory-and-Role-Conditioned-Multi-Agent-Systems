# test_extension5.py
print("Testing Extension 5 components...\n")

try:
    from reflexion.reflection import ReflectionOptimizer
    print("✅ ReflectionOptimizer imported")
    
    from reflexion.agents import OptimizedReflexionAgent
    print("✅ OptimizedReflexionAgent imported")
    
    from reflexion.config import SecureConfigLoader
    from reflexion.llm import BaseLLMModel
    
    config = SecureConfigLoader().load_from_env_file('.env')
    llm = BaseLLMModel(
        config['openrouter_api_key'],
        config['openrouter_model'],
        config['gemini_api_base'],
        config['rate_limit_delay']
    )
    print("✅ LLM initialized")
    
    agent = OptimizedReflexionAgent(llm, memory_mode='temporal', optimize_reflections=True)
    print(f"✅ OptimizedReflexionAgent created")
    print(f"   Memory mode: {agent.memory_mode}")
    print(f"   Optimizations: {agent.optimize_reflections}")
    
    optimizer = ReflectionOptimizer(llm, min_score=0.6)
    score = optimizer.score_reflection(
        "Trial 1: The function failed because it didn't handle empty lists. Fix: Add check for empty input.",
        "Write a function to process lists"
    )
    print(f"✅ Reflection scoring works: {score:.2f}")
    
    print("\n" + "="*60)
    print("✅✅✅ EXTENSION 5 READY! ✅✅✅")
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
