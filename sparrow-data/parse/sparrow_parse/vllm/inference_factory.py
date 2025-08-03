from sparrow_parse.vllm.huggingface_inference import HuggingFaceInference
from sparrow_parse.vllm.local_gpu_inference import LocalGPUInference
# from sparrow_parse.vllm.mlx_inference import MLXInference
from sparrow_parse.vllm.openai_inference import OpenAIInference


class InferenceFactory:
    def __init__(self, config, fields=None, schema=None):
        self.config = config
        self.fields = fields
        self.schema = schema

    def get_inference_instance(self):
        if self.config["method"] == "huggingface":
            return HuggingFaceInference(hf_space=self.config["hf_space"], hf_token=self.config["hf_token"])
        elif self.config["method"] == "local_gpu":
            model = self._load_local_model()  # Replace with actual model loading logic
            return LocalGPUInference(model=model, device=self.config.get("device", "cuda"))
        # elif self.config["method"] == "mlx":
            # return MLXInference(model_name=self.config["model_name"])
        elif self.config["method"] == "openai":
            return OpenAIInference(
                api_key=self.config.get("api_key"),
                model_name=self.config.get("model_name", "gpt-4o"),
                max_tokens=self.config.get("max_tokens", 4000),
                temperature=self.config.get("temperature", 0.0),
                fields=self.fields,
                schema=self.schema
            )
        else:
            raise ValueError(f"Unknown method: {self.config['method']}")

    def _load_local_model(self):
        # Example: Load a PyTorch model (replace with actual loading code)
        # model = torch.load('model.pth')
        # return model
        raise NotImplementedError("Model loading logic not implemented")
