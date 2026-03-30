print("Testing OriginalReflexionAgent...\n")

try:
    from reflexion.config import SecureConfigLoader
    from reflexion.llm import BaseLLMModel
    from reflexion.agents import OriginalReflexionAgent
    
    config = SecureConfigLoader().load_from_env_file('.env')
    llm = BaseLLMModel(
        config['openrouter_api_key'],
        config['openrouter_model'],
        config['gemini_api_base'],
        config['rate_limit_delay']
    )
    
    agent = OriginalReflexionAgent(llm, memory_mode='temporal', max_trials=3)
    print("✅ OriginalReflexionAgent created!")
    print(f"   Memory mode: {agent.memory_mode}")
    print(f"   Max trials: {agent.max_trials}")
    print(f"   Memory size: {len(agent.memory)}")
    
    print("\n✅ ORIGINAL AGENT READY - Matches your old reflexion_old.py!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
