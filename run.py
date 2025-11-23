from evolve_agent import EvolveAgent    
import asyncio
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# logging.basicConfig(
#     level=logging.DEBUG, 
#     # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
#     # filename='app.log',  
# )

evolve_agent = EvolveAgent(
    initial_program_path="benchmark/MSTD/initial_program.py",
    evaluation_file="benchmark/MSTD/evaluator.py",
    initial_proposal_path="benchmark/MSTD/initial_proposal.txt",
    config_path="configs/openrouter_config.yaml",
)

async def main():
    best_program = await evolve_agent.run(iterations=50) 
    print(best_program)

asyncio.run(main())
# print(evolve_agent)

