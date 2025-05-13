import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging

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

    parser.add_argument("--query", type=str, default="what is the key to a happy life")
    parser.add_argument("--persona", type=str, default="")
    parser.add_argument("--system", type=str, default="windows")
    parser.add_argument("--model", type=str, default="deepseek_7b")
    parser.add_argument("--processor", type=str, default="npu")
    parser.add_argument("--model_type", type=str, default="default")

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
                                    verbose=2)
    
    iInfer.embedding_session(query=args.query, persona=args.persona)

if __name__=="__main__":
    main()