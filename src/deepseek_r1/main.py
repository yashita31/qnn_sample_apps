"""
author: Derrick Johnson
date: 05/12/2025
todo: 
    replace print with logging
    add comments
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import numpy as np

from model_loader import ModelLoader
from deepseek_model_inference import DeepSeekModelInference

# from deepseek_model_inference import ModelInference

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser(description="DeepSeek R1 App: Run main.py with or without arguments")

    parser.add_argument("--query", 
                        type=str, 
                        default="What is the key to a happy life, give me some steps I can follow.",
                        help="Initial Query")
    parser.add_argument("--persona", 
                        type=str, 
                        default="",
                        help="Personas Available: THERAPIST, CYBER_SECURITY, CHEF, CARE_TAKER")
    parser.add_argument("--system", 
                        type=str, 
                        default="windows",
                        help="Operating System")
    parser.add_argument("--model", 
                        type=str, 
                        default="deepseek_7b",
                        help="Models: deepseek_7b, deepseek_14b")
    parser.add_argument("--processor", 
                        type=str, 
                        default="npu",
                        help="Processors Available: Hexagon(NPU), CPU")
    parser.add_argument("--model_type", 
                        type=str, 
                        default="default",
                        help="All DeepSeek Models are Quantized (Do Not Change)")
    parser.add_argument("--max_tokens", 
                        type=int, 
                        default=100,
                        help="Max Tokens to Generate")
    parser.add_argument("--temperature", 
                        type=float, 
                        default=0.6,
                        help="Temperature Scaling")
    parser.add_argument("--top_k", 
                        type=int, 
                        default=5,
                        help="Top K")
    parser.add_argument("--repetition_penalty", 
                        type=float, 
                        default=1.1,
                        help="Repetition Penalty")
    parser.add_argument("--verbose",
                        type=int,
                        default=0,
                        help="Verbose levels: 0:None, 1:Basic, 2:Detailed")

    args = parser.parse_args()

    iLoad = ModelLoader(model=args.model, processor=args.processor,
                        model_type=args.model_type)
    
    model_subdirectory = iLoad.model_subdirectory_path

    graphs = iLoad.graphs
    model_sessions = {graph_name: iLoad.load_model(graph) for graph_name,graph in graphs.items() if graph.endswith(".onnx")}
    tokenizer = next((file for file in graphs.values() if file.endswith("tokenizer.json")), None)

    iInfer = DeepSeekModelInference(model_sessions=model_sessions,
                                    tokenizer= tokenizer,
                                    model_subdirectory=model_subdirectory,
                                    verbose=args.verbose)
    
    iInfer.run_inference(query=args.query,
                         top_k=args.top_k,
                         temperature=args.temperature,
                         persona=args.persona,
                         max_tokens=args.max_tokens,
                         repetition_penalty=args.repetition_penalty)
    

if __name__=="__main__":
    main()