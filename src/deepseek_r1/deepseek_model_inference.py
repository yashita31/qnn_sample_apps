"""
author: Derrick Johnson
date: 05/12/2025
todo: 
    replace print with logging
    add comments
"""

import onnxruntime as ort
import numpy as np
import logging
import time

from enum import IntEnum, Enum
from tokenizers import Tokenizer
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

class VerbosityLevel(IntEnum):
    NONE = 0
    BASIC = 1
    DETAILED = 2

class InferencePersona(Enum):
    THERAPIST = "therapist"
    CYBER_SECURITY ="cyber security specialist"
    CHEF ="chef"
    CARE_TAKER = "care taker"
    DOCTOR = "medical professional"

@dataclass
class ModelParameters:
    batch_size: int=1
    num_heads: int=12
    attn_head_size: int=128
    num_layers: int=28
    max_seq_len: int=64
    temp: float=0.6 # Unnecessary, passing directly to generate function from args
    num_key_value_heads: int=2
    seq_len: Optional[int] = None
    hidden_size: Optional[int] = None


logger = logging.getLogger(__name__)

class DeepSeekModelInference():

    def __init__(self, model_sessions: Dict[str,ort.InferenceSession], 
                 tokenizer: str,
                 model_subdirectory: Path,
                 model_meta: dict,
                 verbose: VerbosityLevel = VerbosityLevel.NONE):
        self.session_mapper = model_sessions
        self.root_dir = Path.cwd()
        self.model_subdirectory = model_subdirectory
        self.tokenizer_path = model_subdirectory/tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.model_params = ModelParameters(**model_meta)
        self.verbose = verbose
        self.softmax = lambda x, temperature=1: np.exp((x-np.max(x))/temperature)/np.sum(np.exp((x-np.max(x))/temperature), axis=-1)

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
                       
        query_build += query
        query_build += assistant
        return query_build

    def tokenize(self, prompt: str) -> np.array:
        return np.array([self.tokenizer.encode(prompt).ids], dtype=np.int64)
    
    def embedding_session(self, query: str, persona: Optional[str]=None, iter: bool=True) -> np.array:
        if not iter:
            prompt = self.query(query, persona)
            token_ids = self.tokenize(prompt)
            verbose = self.verbose
        else:
            # Turn Verbose off within Autoregressive loop
            verbose = 0
            token_ids = query
        expected_outputs = self.session_mapper["EMBEDDING"].get_outputs()[0]
        self.model_params.seq_len = expected_outputs.shape[1]
        self.model_params.hidden_size = expected_outputs.shape[2]

        embedding_output = self.session_mapper["EMBEDDING"].run(None, {"input_ids": token_ids})[0]
        
        self.verbosity_embedding(token_id=token_ids,
                                 embed_output=embedding_output.shape, 
                                 verbose=verbose)
        return embedding_output
    
    def context_session(self, embedding_session_outputs: np.array) -> np.array:
        init_prompts = self._cache_init(embedding_session_outputs)
        ctx_outputs = self.session_mapper["CONTEXT"].run(None, init_prompts)
        self.kv_cache = self.kv_cache_update(ctx_outputs=ctx_outputs)
        hidden_states = ctx_outputs[0]
        self.verbosity_context(init_prompt_inputs=init_prompts,
                               ctx_outputs=ctx_outputs,
                               verbose=self.verbose)
        return hidden_states

    def head_session(self, ctx_hidden_states: np.array) -> np.array:

        logits = self.session_mapper["HEAD"].run(None, {"output_hidden_states": ctx_hidden_states})[0]

        # self.verbosity_head()
        
        return logits
    
    def context_itr_session(self, embedding_session_output: np.array,
                            previous_sequence_length: int=64,
                            io_binding=True):
        seq_lengths = {
            "past_seq_len": np.array([[previous_sequence_length]], dtype=np.int32),
            "total_seq_len": np.array([previous_sequence_length+1], dtype=np.int32)
            }
        
        iter_inputs = {
            "input_hidden_states": embedding_session_output,
            **self.kv_cache,
            **seq_lengths
        }
        if io_binding:
            if not hasattr(self,"iBindingManager"):
                    raise ValueError("IO binding cannot proceed: 'iBindingManager' has not been initialized")
            for name, value in iter_inputs.items():
                self.iBindingManager.bind_input(
                    name=name,
                    buffer=value
                )
           
            self.iBindingManager.bind_output(
                name = self.iBindingManager.layer_names[0],
                buffer = self.output_hidden_states_buffer # need to add checks for all used buffers, do it at the top and abstract out
            )
            
            kv_shape_update = (1, self.model_params.num_key_value_heads, previous_sequence_length+1, self.model_params.attn_head_size)
            
            self.iBindingManager.buffer_reallocation_kv(
                updated_buffer_shape=kv_shape_update,
                num_layers=self.model_params.num_layers,
                present_key_buffer=self.present_key_buffer,
                present_value_buffer=self.present_value_buffer,
            )
            # start = time.time()
            self.session_mapper.get("CONTEXT_ITER").run_with_iobinding(self.iBindingManager.io_binding)
            # duration = time.time() - start
            # print("Time per iteration (IOBinding):", duration)
            hidden_states = self.output_hidden_states_buffer
            self.kv_cache = {
                **self.present_key_buffer,
                **self.present_value_buffer
            }
            

        else:
            # start = time.time()
            iter_outputs = self.session_mapper["CONTEXT_ITER"].run(None, iter_inputs)
            # duration = time.time() - start
            # print("Time per iteration (No IOBinding):", duration)
            self.kv_cache = self.kv_cache_update(ctx_outputs=iter_outputs) 
            hidden_states = iter_outputs[0]
        # self.verbosity_context_iter()

        return hidden_states 
        

    def next_token_prediction(self, logits: list, generated_ids: list,
                              temperature: float=1, top_k: Optional[int]=None,
                              repetition_penalty: Optional[float]=None):
        last_logit = logits[0,-1]
        if repetition_penalty:
            last_logit = self.apply_repetition_penalty(logits=last_logit, generated_ids=generated_ids, penalty=repetition_penalty)

        probas = self.softmax(last_logit, temperature=temperature)
        indices = probas.copy()
 
        if top_k:
            indices, probas = self._top_k_probas(probas=probas, k=top_k)
            next_token_id = int(np.random.choice(indices, p=probas))
        else:
            next_token_id = int(np.random.choice(len(indices), p=probas))
        
        return next_token_id
    
    def run_inference(self, query: str, 
                      top_k: int, 
                      temperature: float,
                      persona: Optional[str]=None, 
                      max_tokens: int=100,
                      repetition_penalty: float=1.1,
                      io_binding: bool=True
                      ) -> None:
        
        # Iter set to false because this grabs the initial embeddings
        embedding_output = self.embedding_session(query=query, persona=persona, iter=False)
        context_output = self.context_session(embedding_session_outputs=embedding_output)
        logits = self.head_session(ctx_hidden_states=context_output)
        next_token_id = self.next_token_prediction(logits=logits, generated_ids=logits, temperature=temperature)

        generated_ids = [next_token_id]
        prev_sequence_length = self.model_params.max_seq_len

        if io_binding:
            self.iBindingManager = IOBindingManager(inference_session=self.session_mapper["CONTEXT_ITER"])
            _, _, hidden_dimensions = context_output.shape
            _, num_heads, _, head_dimensions = self.kv_cache.get("past_keys_0").shape

            # Initial Buffer allocation
            hidden_state_dimensions = (1,1,hidden_dimensions)
            kv_cache_dimensions = (1, num_heads, prev_sequence_length, head_dimensions)
            self.output_hidden_states_buffer = self.iBindingManager.buffer_preallocation_hidden_states(buffer_shape=hidden_state_dimensions)

            self.present_key_buffer = self.iBindingManager.buffer_preallocation_kv(buffer_shape=kv_cache_dimensions,
                                                                         num_layers=ModelParameters.num_layers,
                                                                         keys_or_values="keys")
            self.present_value_buffer = self.iBindingManager.buffer_preallocation_kv(buffer_shape=kv_cache_dimensions,
                                                                           num_layers=ModelParameters.num_layers,
                                                                           keys_or_values="values")
        print("\nInitial Query:\n", query)
        print("Generated:")
        # for output in self.session_mapper.get("CONTEXT_ITER").get_outputs():
        #     print(f"{output.name}: shape={output.shape}, type={output.type}")

        for _ in range(max_tokens):
            
            input_ids = np.array([[next_token_id]], dtype=np.int64)
            print(self.tokenizer.decode([next_token_id], skip_special_tokens=True), end="", flush=True)
            
            embedding_output = self.embedding_session(query=input_ids)
            iter_outputs = self.context_itr_session(embedding_session_output=embedding_output,
                                                    previous_sequence_length=prev_sequence_length,
                                                    io_binding=io_binding)
            logits = self.head_session(ctx_hidden_states=iter_outputs)
            next_token_id = self.next_token_prediction(logits=logits, generated_ids=generated_ids,
                                                       temperature=temperature, top_k=top_k,
                                                       repetition_penalty=repetition_penalty)
            generated_ids.append(next_token_id)
            prev_sequence_length += 1

            if next_token_id == self.tokenizer.token_to_id("< | end_of_sentence | >"):
                break


    def kv_cache_update(self, ctx_outputs):
        present_kv = {f"past_keys_{layer}": ctx_outputs[1 + layer * 2] for layer in range(self.model_params.num_layers)}
        present_kv.update({f"past_values_{layer}": ctx_outputs[1 + layer * 2 + 1] for layer in range(self.model_params.num_layers)})
        return present_kv  
          
    def _build_persona(self, role: InferencePersona) -> str:
            return f"From the perspective of a {role.value}.\n"#f"You are a {role.value}.\n"

    def _cache_init(self, embedding_output: np.array) -> Dict[str,np.array]:
        empty_kv = defaultdict()
        output_dimensionality = embedding_output.shape[1]
        past_shape = (self.model_params.batch_size,
                          self.model_params.num_key_value_heads,
                          self.model_params.max_seq_len,
                          self.model_params.attn_head_size)
        
        for layer in range(self.model_params.num_layers):
            empty_kv[f"past_keys_{layer}"] = np.zeros(past_shape, dtype=np.float32)
            empty_kv[f"past_values_{layer}"] = np.zeros(past_shape, dtype=np.float32)
        
        seq_lengths = {
            "past_seq_len": np.array(output_dimensionality-1, dtype=np.int32).reshape(1,1),
            "total_seq_len": np.array([past_shape[2]], dtype=np.int32)
        }
        padded_embedding_outputs = self.input_padding(embedding_output=embedding_output)

        init_prompt_inputs = {
            **empty_kv,
            **seq_lengths,
            "input_hidden_states": padded_embedding_outputs
        }

        return init_prompt_inputs
    
    def _softmax_numpy(self, x: np.array, temparature: float=1.0) -> np.array:
        x = x- np.max(x)
        x = x/temparature
        return np.exp(x)/np.sum(np.exp(x), axis=-1)
    
    def _top_k_probas(self, probas: np.array, k: int=5) -> np.array:
        probas = probas.copy()
        probas /= np.sum(probas)
        top_indices_sorted = np.argsort(-probas)[:k]
        top_k_probas = probas[top_indices_sorted]
        top_k_probas /= np.sum(top_k_probas)
        return top_indices_sorted, top_k_probas
    
    def apply_repetition_penalty(self, logits: np.array, generated_ids: list, penalty: float=1.1):
        for token_id in set(generated_ids):
            logits[token_id] /= penalty
        return logits


    def input_padding(self, embedding_output, padding_id: int=151643) -> np.array:
        batch_size, seq_len, embed_dim = embedding_output.shape
        padded_embedding = np.full((batch_size, self.model_params.max_seq_len, embed_dim),
                                   padding_id,
                                   dtype=embedding_output.dtype)
        padded_embedding[:, :seq_len, :] = embedding_output
        return padded_embedding
    

    def verbosity_head():
        pass

    def verbosity_context(self, init_prompt_inputs: dict,
                          ctx_outputs: np.array,
                          verbose: int=VerbosityLevel.NONE):
        match verbose:
            case 1 | 2:
                num_layers_verified = (len([layer for layer in init_prompt_inputs.keys() if "past" in layer])-1)//2
                cache_key = next((layer for layer in init_prompt_inputs.keys() if "past_keys" in layer))
                cache_shape = init_prompt_inputs[cache_key].shape
                print("\n.....INITIALIZATION")
                print(f".....Calculated Layers from KV Cache: {num_layers_verified}")
                print(f".....KV Cache Shape: {cache_shape}")
            case _:
                pass

    def verbosity_embedding(self, token_id: List[np.array],
                            embed_output,
                            verbose: int=VerbosityLevel.NONE):
        token_size = token_id.shape[1]
        
        match verbose:
            case 1:
                print(f"\n.....EMBEDDING_SESSION")
                print(f".....Token Count: {token_size}")

            case 2:
                print(f"\n.....EMBEDDING_SESSION")
                print(f".....Token Count: {token_size}")
                print(f".....Prompt:\n{self.tokenizer.decode(token_id.flatten())}", end="")
                print(f".....Token ID: {token_id.flatten()}")
                print(f".....Embedding Output: {embed_output}")

            case _:
                pass 
                
    def verbosity_init(self, verbose: int=VerbosityLevel.NONE):
        model_name = self.model_subdirectory.name
        tokenizer_path = str(self.tokenizer_path)
        
        match verbose:
            case 1:
                print(f"\n.....INIT")
                keys = ", ".join(list(self.session_mapper.keys()))
                print(f".....Model: {model_name}") #logger.info()
                print(f".....Graphs: {keys}")
                print(f".....Tokenizer Path: {tokenizer_path}")

            case 2:
                print(f"\n.....INIT")
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

class IOBindingManager():
    def __init__(self, inference_session: ort.InferenceSession):
        self.session = inference_session
        self.outputs = self.session.get_outputs()
        self.layer_names = [output.name for output in self.outputs]
        self.io_binding = self.session.io_binding()

    def buffer_preallocation_hidden_states(self,
                                           buffer_shape: tuple,
                                           dtype: np.dtype=np.float32,
                                        ) -> np.array:
        return np.empty(buffer_shape, dtype=dtype)

    def buffer_preallocation_kv(self, 
                                buffer_shape: Tuple,
                                num_layers: int,
                                keys_or_values: str,
                                dtype: np.dtype=np.float32
                                ) -> Dict[str,str]:
        return {f"past_{keys_or_values}_{layer}": np.empty(buffer_shape, dtype=dtype) \
                for layer in range(num_layers)}   
    
    def buffer_reallocation_kv(self, 
                               updated_buffer_shape: Tuple,
                               num_layers: int,
                               present_key_buffer: Dict[str,str],
                               present_value_buffer: Dict[str,str],
                               dtype: np.dtype=np.float32
                               ) -> None:
        for layer in range(num_layers):
            key_name = self.layer_names[1 + layer * 2]
            value_name = self.layer_names[1 + layer * 2 + 1]

            key_buffer = np.empty(updated_buffer_shape, dtype=dtype)
            value_buffer = np.empty(updated_buffer_shape, dtype=dtype)

            present_key_buffer[f"past_keys_{layer}"] = key_buffer
            present_value_buffer[f"past_values_{layer}"] = value_buffer
            
            self.bind_output(name=key_name,
                             buffer=key_buffer,
                              device_type="cpu",
                              device_id=0,
                              )
            self.bind_output(name=value_name,
                             buffer=value_buffer,
                              device_type="cpu",
                              device_id=0,
                              )
        # print("Key Shapes:", key_buffer.shape)
        # print("value Shapes:", value_buffer.shape)
    def bind_output(self,
                 name: str,
                 buffer: np.array,
                 device_type: str="cpu",
                 device_id: int=0
                 ) -> None:
        
        self.io_binding.bind_output(
            name=name,
            device_type=device_type,
            device_id=device_id,
            element_type=buffer.dtype,
            shape=buffer.shape,
            buffer_ptr=buffer.ctypes.data)
        
    def bind_input(self,
                    name: str,
                    buffer: np.array,
                    device_id: int=0,
                    device_type: str="cpu",
                    ) -> None:
        
        self.io_binding.bind_input(name,
                                   device_type=device_type,
                                   element_type=buffer.dtype,
                                   shape=buffer.shape,
                                   buffer_ptr=buffer.ctypes.data,
                                   device_id=device_id
                                   ) 



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