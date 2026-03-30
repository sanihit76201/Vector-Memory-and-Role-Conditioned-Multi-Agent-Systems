from reflexion.agents.rl_reflexion import RLReflexionAgent
from reflexion.llm import BaseLLMModel
from reflexion.config import SecureConfigLoader

config = SecureConfigLoader().load_from_env_file('.env')
llm = BaseLLMModel(config['openrouter_api_key'], config['openrouter_model'])

agent = RLReflexionAgent(llm, env_name='CartPole-v1')
state, _ = agent.env.reset()

for episode in range(10):
    result = agent.solve_task({'id': f'episode_{episode}'})
    print(f"Episode {episode}: Reward={result['reward']}, Success={result['success']}")
