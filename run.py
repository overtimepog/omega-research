from evolve_agent import EvolveAgent    
import asyncio
import logging

# logging.basicConfig(
#     level=logging.DEBUG, 
#     # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
#     # filename='app.log',  
# )

evolve_agent = EvolveAgent(
    initial_program_path="results/initial_program.py",
    evaluation_file="results/evaluator.py",
    initial_proposal_path="results/initial_program.py",
    config_path="configs/default_config.yaml"
)

async def main():
    best_program = await evolve_agent.run(iterations=50) 
    print(best_program)

asyncio.run(main())
# print(evolve_agent)

