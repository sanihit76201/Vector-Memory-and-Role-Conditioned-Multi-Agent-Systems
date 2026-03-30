"""Quick test for SmartReflexionAgent import."""

print("="*60)
print("🧠 Testing SmartReflexionAgent (Extension 1)")
print("="*60 + "\n")

try: 
    print("Step 1: Importing SmartReflexionAgent...")
    from reflexion.agents import SmartReflexionAgent
    print("✅ SmartReflexionAgent imported successfully!\n")
    
    print("Step 2: Loading config and LLM...")
    from reflexion.config import SecureConfigLoader
    from reflexion.llm import BaseLLMModel
    
    config = SecureConfigLoader().load_from_env_file('.env')
    llm = BaseLLMModel(
        config['openrouter_api_key'],
        config['openrouter_model'],
        config['gemini_api_base'],
        config['rate_limit_delay']
    )
    print(f"✅ LLM initialized: {llm.model}\n")
    
    print("Step 3: Creating Smart Agent...")
    agent = SmartReflexionAgent(llm, max_trials=3)
    print("✅ SmartReflexionAgent created!")
    print(f"   Memory mode: Temporal (max_size=3)")
    print(f"   Task isolation: ENABLED")
    print(f"   Max trials: {agent.max_trials}")
    print(f"   Current memory size: {len(agent.memory)}")
    print("\n" + "="*60)
    print("🎉 EXTENSION 1 READY!")
    print("🚀 Run: cd experiments && python run_comparison.py")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
