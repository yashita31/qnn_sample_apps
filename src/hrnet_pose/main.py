import cv2 as cv
import onnxruntime as ort
import onnx
import os
import torch
import json
import re

from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

class ModelLoader:
    def __init__(self, system: str, processor: str):
        self.system = system.upper()
        self.processor = processor.upper()
        self.config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    def _get_config_path(self, filename: str) -> str:
        """
        Constructs the absolute path to a configuration file based on its filename.

        Args:
            filename (str): The name of the configuration file.

        Returns:
            str: The absolute path to the configuration file.
        """

        return os.path.abspath(os.path.join(self.config_dir,f"{filename}"))

    def _get_model_path(self, model_name: str, ) -> ort.InferenceSession:
        """
        Retrieves the absolute path of an ONNX model based on the model name provided, using the path specified in a JSON configuration file.

        Args:
            model_name (str): The name of the model to load. This should correspond to a key in the JSON configuration file.

        Returns:
            str: The absolute path to the ONNX model file.

        Raises:
            ValueError: If the specified model name is not found in the JSON configuration file.
        """

        config_path = self._get_config_path(filename="models.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        model_path = os.path.abspath(config["MODELS"].get(model_name.upper()))

        if not model_path:
            raise ValueError(f"Model path for {model_name} not found in config")

        return model_path

    def _get_dll_path(self, system: str, processor: str) -> str:
        """
        Retrieves the absolute path of a specified DLL file based on the system and processor type.
        The path is determined from a JSON configuration file.

        Args:
            dll_name (str): The name of the DLL to load. 

        Returns:
            str: The absolute path to the DLL file.

        Raises:
            KeyError: If the specified system or processor type is not found in the JSON configuration file.
        """
        config_path = self._get_config_path(filename= "dll.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        dll_path = os.path.abspath(config[system.upper()].get(processor.upper()))

        if not dll_path:
            raise ValueError(f"Processor: {processor} or System: {system} not found within config")

        return dll_path

    def load_model(self, model_name: str) -> ort.InferenceSession:
        """
        Loads an ONNX model with specified system and processor settings, enabling profiling.

        Args:
            model_name (str): The name of the model to load.
            system (str): The operating system type for which the DLL is configured (e.g., "Windows", "Linux").
            processor (str): The processor type required for the DLL (e.g., "x86", "arm").

        Returns:
            ort.InferenceSession: An ONNX Runtime InferenceSession configured with the specified model, 
                                execution provider, and profiling options.
        """

        model_path = self._get_model_path(model_name=model_name)
        dll_path = self._get_dll_path(system=system, processor=processor)
        profile_path = re.sub(r"\.onnx$",".profile",model_path)

        session = ort.InferenceSession(model_name=model_path,
                                    providers=["QNNExecutionProvider"],
                                    provider_options=[{"backend_path":dll_path,
                                                        "profiling_level":"detailed",
                                                        "profiling_file_path": profile_path
                                                        }])
        return session

if __name__=="__main__":
    
    model_name = "hrnet_pose"
    system = "windows"
    processor = "npu"
    model_loader = ModelLoader(system=system, processor=processor)
    session = model_loader.load_model(model_name)
