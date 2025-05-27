import onnxruntime as ort
import os
import json
import platform

from pathlib import Path

import sys
# print(ort.get_all_providers())

class ModelLoader:
    def __init__(self, model: str, processor: str, model_type: str) -> None:
        """
        Initializes an instance with the specified model and processor.

        This constructor sets up the model name, processor type, and configuration directory path.
        The configuration directory is set to the directory two levels up from the current file location.

        Args:
            model (str): The name of the model to be used.
            processor (str): The type of processor (e.g., "cpu", "npu").
            model_type (str): 
                                Refers to Default or Quantized for most models
                                Model size for LLMs (LLMs are quantized by default)

        Attributes:
            model (str): Stores the name of the specified model.
            processor (str): Stores the processor type.
            config_dir (str): The absolute path to the configuration directory, located two levels up 
                            from the current file directory.
        """
        self.processor = processor.upper()
        self.model = model
        self.model_type = model_type
        self.root_dir = Path.cwd()
        self.onnx_root = Path(ort.__file__).parent.resolve()

        self.platform_manager()

        self.model_config = self._get_config(filename="models.json")
        self.executioner_config = self._get_config(filename="executioner.json")
        self.dll_config =  self._get_config(filename= "dll.json")       

        self.model_subdirectory_path = self.get_model_path_subdirectory(model_name=self.model)
        
        

    def _get_config(self, filename: str) -> json:
        """
        Constructs the absolute path to a configuration file based on its filename.

        Args:
            filename (str): The name of the configuration file.

        Returns:
            str: The absolute path to the configuration file.
        """
        config_path = self.root_dir/filename

        with open(config_path, "r") as f:
            config = json.load(f)

        return config

    def get_model_path_subdirectory(self, model_name: str) -> ort.InferenceSession:
        """
        Retrieves the absolute path of an ONNX model based on the model name provided, using the path specified within models.json configuration file.

        Args:
            model_name (str): The name of the model to load. This should correspond to a key in the models.json configuration file.

        Returns:
            str: The absolute path to the ONNX model file.

        Raises:
            ValueError: If the specified model name is not found in the models.json configuration file.
        """

        
        model_subdirectory = self.model_config["MODELS"][model_name.upper()].get("PATH_SUBDIRECTORY")
        # models = config["MODELS"][model_name.upper()][self.model_type.upper()]
        model_subdirectory_path = self.root_dir/"models"/model_subdirectory #config["MODELS"][model_name.upper()][self.model_type.upper()]

        if not model_subdirectory_path:
            raise ValueError(f"Model path for {model_name} not found in config")

        return model_subdirectory_path

    def _get_dll_path(self, runtime: str="qnn") -> str:
        """
        Retrieves the absolute path of a specified DLL file based on the system and processor type.
        The path is determined from the dll.json configuration file.

        Args:
            dll_name (str): The name of the DLL to load. 

        Returns:
            str: The absolute path to the DLL file.

        Raises:
            KeyError: If the specified system or processor type is not found in the dll.json configuration file.
        """
        # system = system if system != None else self.system

        dll_path = self.onnx_root/self.executioner_config[self.processor].get("PATH")
    # WHY IS THIS RETURNING ERROR, WHY DO I EVEN CARE IF THE PATH DOES EXIST
    #     if not dll_path.is_file():
    #         raise ValueError(f"Processor: {processor} or System: {system.upper()} not found within config")
        

        return dll_path
    
    def _get_executioner(self) -> str:
        """
        Retrieves the ExecutionProvider configuration for a specified processor.

        This function loads a configuration file (executioner.json) containing available processors 
        and their corresponding ExecutionProvider. It checks if the specified processor is available 
        in the configuration file. If the processor is not listed, it raises an error indicating 
        the available options. If the processor is found, it returns the relevant ExecutionProvider 
        for the selected processor.

        Returns:
            str: The ExecutionProvider for the specified processor.

        Raises:
            ValueError: If the specified processor does not have a supported ExecutionProvider.
        """
        # config_file_name = "executioner.json"
        # config = self._get_config(filename= config_file_name)
        if self.processor not in self.executioner_config.keys():
            raise ValueError(f"Selected processor ({self.processor}) not available. Please select {' | '.join(list(self.executioner_config.keys()))}")
        return self.executioner_config[self.processor]


    def load_model(self, onnx_graph: ort, htp_performance_mode: str="burst", 
                   soc_model: str="60", profiling_level: str="off",
                   profiling_file_path: str=None, 
                   htp_graph_finalization_optimization_mode: str="3") -> ort.InferenceSession:
        """
        Loads an ONNX model with specified system and processor settings.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            ort.InferenceSession: An ONNX Runtime InferenceSession configured with the specified model, 
                                execution provider.
        """
        session_options = ort.SessionOptions()
        
        model_path = self.model_subdirectory_path/onnx_graph
        executioner = self._get_executioner()
        dll_path = self._get_dll_path()

        qnn_provider_options = {
            "backend_path": dll_path, 
            "htp_performance": htp_performance_mode,
            "soc_model": soc_model,
            "profiling_level": profiling_level,
            "profiling_file_path": str(self.model_subdirectory_path) if profiling_file_path == None else str(profiling_file_path),
            "htp_graph_finalization_optimization_mode": htp_graph_finalization_optimization_mode,
            "enable_htp_shared_memory_allocator": 1,
        }

        session = ort.InferenceSession(model_path, 
                                       providers=[(executioner.get("EP"),qnn_provider_options)],
                                       sess_options=session_options
                                       )

        return session
    
    # @property
    # def executioners(self) -> str:
    #     """
    #     Retrieves and lists available executioners from a configuration file.

    #     This property reads from the `executioner.json` configuration file to obtain
    #     key-value pairs of executioners and their settings, formats them into a list,
    #     and returns a formatted string listing each executioner and its associated value.

    #     Returns:
    #     -------
    #     str
    #         A formatted string where each line represents an executioner in the format
    #         "key: value", with one executioner per line.

    #     Notes:
    #     ------
    #     The configuration file `executioner.json` should be located in the expected
    #     directory, and the `_get_config` method should correctly handle reading the
    #     JSON file and returning its contents as a dictionary.
    #     """
    #     config_file_name = "executioner.json"
    #     config = self._get_config(filename= config_file_name)
    #     available_executioners = [f"{key}: {value}" for key,value in config.items()]
    #     return "\n".join(available_executioners)
    
    @property
    def graphs(self) -> str:

        return self.model_config["MODELS"][self.model.upper()][self.model_type.upper()]
    
    def platform_manager(self):
        self.arch = platform.machine().lower()
        self.system = platform.system().upper()

        # Inference on Hexagon only available in ARM64 environment
        # Change processor to CPU if environment is incorrect
        if "arm64" not in self.arch and "aarch64" not in self.arch:
            self.processor = "CPU" if "ARM64" not in self.arch else self.processor

        self


if __name__=="__main__":
    
    model_name = "hrnet_pose"
    system = "windows"
    processor = "npu"
    model_type = "quantized"
    iLoad = ModelLoader(model=model_name, processor=processor, model_type=model_type)
    onnx_graph = "hrnet_quantized.onnx"#
    print(iLoad.graphs)
    session = iLoad.load_model(onnx_graph)
    print(session.get_providers())
   
    print(session.get_inputs()[0])
    # print(iLoad.executioners)
    print(session.get_providers())
    
