"""
author: Derrick Johnson
date: 05/12/2025
todo: 
    replace print with logging
"""

import onnxruntime as ort
import numpy as np
import logging

from enum import IntEnum, Enum
from tokenizers import Tokenizer
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

class VerbosityLevel(IntEnum):
    NONE = 0
    BASIC = 1
    DETAILED = 2

class InferencePersona(Enum):
    THERAPIST = "therapist"
    CYBER_SECURITY ="cyber security specialist"
    CHEF ="chef"

@dataclass
class ModelParameters:
    batch_size: int=1
    num_heads: int=28
    attn_head_size: int=128
    num_layers: int=28
    max_seq_len: int=64
    temp: float=0.7
    num_key_value_heads: int=4
    seq_len: Optional[int] = None
    hidden_size: Optional[int] = None


logger = logging.getLogger(__name__)

class DeepSeekModelInference():

    def __init__(self, model_sessions: Dict[str,ort.InferenceSession], 
                 tokenizer: str,
                 model_subdirectory: Path,
                 verbose: VerbosityLevel = VerbosityLevel.NONE):
        self.session_mapper = model_sessions
        self.root_dir = Path.cwd()
        self.model_subdirectory = model_subdirectory
        self.tokenizer_path = model_subdirectory/tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.model_params = ModelParameters()
        self.verbose = verbose

        self.verbosity_init(self.verbose)

    def query(self, query: str, persona: Optional[str]=None) -> str:
        user = "<｜User｜>\n"
        assistant = "\n<｜Assistant｜><think>\n"
        query_build = user

        if persona:
            try:
                persona_enum = InferencePersona[persona.upper()]
                persona_context = self._build_persona(persona_enum)
                query_build += persona_context
            except KeyError:
                available_personas = ", ".join([role.name for role in InferencePersona])
                print(f".....Available Personas: {available_personas}")
                       
        query_build += query.strip()
        query_build += assistant

        return query_build

    def tokenize(self, prompt: str) -> np.array:
        return np.array([self.tokenizer.encode(prompt).ids], dtype=np.int64)
    
    def embedding_session(self, query: str, persona: str):
        prompt = self.query(query, persona)
        token_ids = self.tokenize(prompt)
        expected_outputs = self.session_mapper["EMBEDDING"].get_outputs()[0]
        self.model_params.seq_len = expected_outputs.shape[1]
        self.model_params.hidden_size = expected_outputs.shape[2]

        embedding_output = self.session_mapper["EMBEDDING"].run(None, {"input_ids": token_ids})[0]
        
        self.verbosity_embedding(token_id=token_ids,
                                 embed_output=embedding_output.shape, 
                                 verbose=self.verbose)

    # CONTINUE WITH FROM SETTING UP EMPTY KV CACHE    
            
    def _build_persona(self, role: InferencePersona) -> str:
            return f"Imagine you are a {role.value}, "
        
    def verbosity_embedding(self, token_id: List[np.array],
                            embed_output,
                            verbose: int=VerbosityLevel.NONE):
        token_size = token_id.shape[1]
        print(f"\n.....EMBEDDING_SESSION")
        match verbose:
            case 1:
                print(f".....Token Count: {token_size}")
            case 2:
                print(f".....Token Count: {token_size}")
                print(f".....Prompt:\n{self.tokenizer.decode(token_id.flatten())}", end="")
                print(f".....Token ID: {token_id.flatten()}")
                print(f".....Embedding Output: {embed_output}")
                
    def verbosity_init(self, verbose: int=VerbosityLevel.NONE):
        model_name = self.model_subdirectory.name
        tokenizer_path = str(self.tokenizer_path)
        print(f"\n.....INIT")
        match verbose:
            case 1:
                keys = ", ".join(list(self.session_mapper.keys()))
                print(f".....Model: {model_name}") #logger.info()
                print(f".....Graphs: {keys}")
                print(f".....Tokenizer Path: {tokenizer_path}")

            case 2:
                print(f".....Model: {model_name}")
                for graph_name, graph_session in self.session_mapper.items():
                    session_inputs = graph_session.get_inputs()
                    session_outputs = graph_session.get_outputs()
                    input_head = session_inputs[0]
                    output_head = session_outputs[0]
                    print(f".....Graph Name: {graph_name}")
                    print(f".....Expected Input Name: {input_head.name}")
                    print(f".....Expected Input Shape: {input_head.shape}")
                    print(f".....Expected Input Type: {input_head.type}")
                    print("")
                    print(f".....Expected Output Name: {output_head.name}")
                    print(f".....Expected Output Shape: {output_head.shape}")
                    print(f".....Expected Output Type: {output_head.type}")
                    print("."*50)
                print(f".....Tokenizer Path: {tokenizer_path}")

            case _:
                pass
        

if __name__=="__main__":
    dummy_dict = {"EMBEDDING":"EMBEDDING_DUMMY",
                  "CONTEXT":"CONTEXT_DUMMY",
                  "CONTEXT_ITER":"CONTEXT_ITER_DUMMY",
                  "HEAD":"HEAD_DUMMY"}
    tokenizer = "tokenizer.json"
    persona = "chef"
    query="provide me a step by step recipe for chicken, be sure to include cook times, ingredients, and preparation"
    model_subdirectory = Path(r"C:\Users\DFS\Desktop\qualcomm_official\Pose-Detection-with-HRPoseNet\models\qnn-deepseek-r1-distill-qwen-7b")
    iInfer = DeepSeekModelInference(dummy_dict, tokenizer, model_subdirectory)
    #query = iInfer.query(query="provide me a step by step recipe for chicken, be sure to include cook times, ingredients, and preparation", persona=persona)
    iInfer.embedding_session(query=query,persona=persona)