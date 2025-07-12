#!/usr/bin/env python3
"""
Alpha Research Startup Script
"""

import asyncio
import os
import json
from typing import List, Dict
from pathlib import Path

from evolve_agent import EvolveAgent
from evolve_agent.config import load_config, RewardModelConfig
from evolve_agent.reward_model import RewardModel
from evolve_agent.llm.ensemble import LLMEnsemble


class AlphaResearchStarter:

    def __init__(self, config_path: str = "configs/default_config.yaml"):
        self.config = load_config(config_path)
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize reward model
        self.reward_model = RewardModel(self.config.rewardmodel)
        
        # Initialize LLM ensemble for code generation
        print((self.config.llm.models))
        self.llm_ensemble = LLMEnsemble(self.config.llm.models)
    
    async def generate_proposal_from_idea(self, simple_idea: str) -> str:

        proposal_prompt = f"""
Please write a paragraph of proposal for the idea focusing on the following points: 

1.clarity 
2.novelty 
3.exact method technically
4.technical rigor 
5.potential impact

{simple_idea}
"""

        try:
            response = await self.llm_ensemble.generate_with_context(
                system_message="You are an expert research proposal writer with extensive experience in computer science and algorithm development.",
                messages=[{"role": "user", "content": proposal_prompt}]
            )
            return response.strip()
        except Exception as e:
            print(f"Error generating proposal: {e}")
            return f"Research Proposal: {simple_idea}\n\nThis research aims to explore and develop solutions for {simple_idea}."
    
    async def score_proposal(self, proposal: str) -> float:

        data = [{
            "title": "",
            "abstract": proposal,
            "gt_score": 0.0  
        }]
        
        try:    
            print(data)
            results = await self.reward_model.score_research_proposal(data)
            if results and len(results) > 0:
                score = results[0].get('score', -1.0)
                print(f"Proposal score: {score}/10")
                return score
            else:
                print("No score returned from reward model")
                return -1.0
        except Exception as e:
            print(f"Error scoring proposal: {e}")
            return -1.0
    
    async def generate_initial_program(self, proposal: str) -> str:

        code_prompt = f"""Based on the following research proposal, implement a Python program that demonstrates the core concepts and algorithms described.

Requirements:
1. Write clean, well-documented Python code
2. Include algorithm implementation
4. Use appropriate data structures and algorithms
5. Make the code executable and testable

Proposal:
{proposal}

"""

        try:
            response = await self.llm_ensemble.generate_with_context(
                system_message="You are an expert Python programmer.",
                messages=[{"role": "user", "content": code_prompt}]
            )
            
            code_lines = []
            in_code_block = False
            for line in response.split('\n'):
                if line.strip().startswith('```python'):
                    in_code_block = True
                    continue
                elif line.strip() == '```' and in_code_block:
                    in_code_block = False
                    continue
                elif in_code_block:
                    code_lines.append(line)
                elif not in_code_block and line.strip() and not line.startswith('#'):

                    if any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from ', 'if __name__']):
                        code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines)
            else:
                return response  
                
        except Exception as e:
            print(f"Error generating program: {e}")

            return '''def main():
    """Main function implementing the research proposal"""
    print("Research proposal implementation")
    return True

if __name__ == "__main__":
    result = main()
    print(f"Execution result: {result}")
'''
    
    def save_files(self, proposal: str, program: str, score: float):
        
        proposal_path = os.path.join(self.results_dir, "initial_proposal.txt")
        with open(proposal_path, 'w', encoding='utf-8') as f:
            f.write(proposal)
        print(f"Saved proposal to: {proposal_path}")
        

        program_path = os.path.join(self.results_dir, "initial_program.py")
        with open(program_path, 'w', encoding='utf-8') as f:
            f.write(program)
        print(f"Saved program to: {program_path}")
        

        metadata = {
            "proposal_score": score,
            "timestamp": __import__('time').time(),
            "files": {
                "proposal": proposal_path,
                "program": program_path
            }
        }
        metadata_path = os.path.join(self.results_dir, "generation_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")
        
        return proposal_path, program_path
    
    async def start_evolution(self, proposal_path: str, program_path: str):
        
        if not os.path.exists(proposal_path) or not os.path.exists(program_path):
            raise FileNotFoundError("Proposal or program file not found")
        
        evolve_agent = EvolveAgent(
            initial_program_path=program_path,
            initial_proposal_path=proposal_path,
            evaluation_file=os.path.join(self.results_dir, "evaluator.py"),
            config=self.config
        )
        
        print("Starting evolution process...")
        
        best_program = await evolve_agent.run(iterations=50)  
        
        if best_program:
            print(f"Evolution completed successfully!")
            print(f"Best program metrics: {best_program.metrics}")
            return best_program
        else:
            print("Evolution failed - no valid programs found")
            return None
    
    async def run_complete_workflow(self, simple_idea: str):
        
        print(f"🚀 Starting Alpha Research with idea: {simple_idea}")
        print("=" * 60)
        
        # Step 1: Generate proposal
        print("📝 Step 1: Generating research proposal...")
        proposal = await self.generate_proposal_from_idea(simple_idea)
        print(f"Generated proposal ({len(proposal)} characters)")
        
        # Step 2: Score proposal with reward model
        print("\n🔍 Step 2: Scoring proposal with reward model...")
        score = await self.score_proposal(proposal)
        
        if score < self.config.rewardmodel.proposal_score_threshold:
            print(f"❌ Proposal score {score:.2f} below threshold {self.config.rewardmodel.proposal_score_threshold}")
            print("Consider refining the idea and trying again.")
            return None
        
        print(f"✅ Proposal score: {score:.2f}/10")
        
        print("\n💻 Step 3: Generating initial program...")
        program = await self.generate_initial_program(proposal)
        print(f"Generated program ({len(program)} characters)")
        
        # Step 4: Save files
        print("\n💾 Step 4: Saving files...")
        proposal_path, program_path = self.save_files(proposal, program, score)
        
        # Step 5: Start evolution
        print("\n🧬 Step 5: Starting evolution...")
        try:
            best_program = await self.start_evolution(proposal_path, program_path)
            
            if best_program:
                print("\n🎉 Alpha Research completed successfully!")
                print(f"Best program ID: {best_program.id}")
                print(f"Final metrics: {best_program.metrics}")
            else:
                print("\n⚠️ Evolution completed but no improvement found")
                
            return best_program
            
        except Exception as e:
            print(f"\n❌ Evolution failed: {e}")
            print("Initial files have been saved and can be used manually.")
            return None


async def main():
    """Main function"""
    
    # Brief research ideas
    simple_ideas = [
        "挂谷猜想",
    ]
    
    # Select an idea or let user input
    selected_idea = simple_ideas[0]  # Can modify selection
    
    print(f"Selected research idea: {selected_idea}")
    
    # Create starter
    starter = AlphaResearchStarter()
    
    # Run complete workflow
    result = await starter.run_complete_workflow(selected_idea)
    
    if result:
        print("\n🎯 Workflow completed successfully!")
    else:
        print("\n💡 Try adjusting the idea or configuration and run again.")


if __name__ == "__main__":
    asyncio.run(main())