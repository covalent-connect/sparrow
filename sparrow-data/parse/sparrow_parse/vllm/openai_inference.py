import openai
import base64
import json
import os
from sparrow_parse.vllm.inference_base import ModelInference
from rich import print


class OpenAIInference(ModelInference):
    """
    A class for performing inference using OpenAI's API.
    Handles image preprocessing, response formatting, and model interaction.
    """

    def __init__(self, api_key=None, model_name="gpt-4o", max_tokens=4000, temperature=0.0, fields=None):
        """
        Initialize the inference class with OpenAI configuration.

        :param api_key: OpenAI API key (if None, will use environment variable OPENAI_API_KEY)
        :param model_name: Name of the OpenAI model to use
        :param max_tokens: Maximum tokens for response generation
        :param temperature: Temperature for response generation
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fields = fields
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
        print(f"OpenAIInference initialized for model: {model_name}")

    def process_response(self, output_text):
        """
        Process and clean the model's raw output to format as JSON.
        """
        try:
            # Check if we have markdown code block markers
            if "```" in output_text:
                # Handle markdown-formatted output
                json_start = output_text.find("```json")
                if json_start != -1:
                    # Extract content between ```json and ```
                    content = output_text[json_start + 7:]
                    json_end = content.rfind("```")
                    if json_end != -1:
                        content = content[:json_end].strip()
                        formatted_json = json.loads(content)
                        return json.dumps(formatted_json, indent=2)

            # Handle raw JSON (no markdown formatting)
            # First try to find JSON array or object patterns
            for pattern in [r'\[\s*\{.*\}\s*\]', r'\{.*\}']:
                import re
                matches = re.search(pattern, output_text, re.DOTALL)
                if matches:
                    potential_json = matches.group(0)
                    try:
                        formatted_json = json.loads(potential_json)
                        return json.dumps(formatted_json, indent=2)
                    except:
                        pass

            # Last resort: try to parse the whole text as JSON
            formatted_json = json.loads(output_text.strip())
            return json.dumps(formatted_json, indent=2)

        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            return output_text

    def encode_image_to_base64(self, image_path):
        """
        Encode image to base64 string for OpenAI API.
        
        :param image_path: Path to the image file
        :return: Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def inference(self, input_data, apply_annotation=False, mode=None):
        """
        Perform inference on input data using OpenAI's API.

        :param input_data: A list of dictionaries containing image file paths and text inputs.
        :param apply_annotation: Optional flag to apply annotations to the output (not supported for OpenAI).
        :param mode: Optional mode for inference ("static" for simple JSON output).
        :return: List of processed model responses.
        """
        # Handle static mode
        if mode == "static":
            return [self.get_simple_json()]

        # Determine if we're doing text-only or image-based inference
        is_text_only = input_data[0].get("file_path") is None
        
        if is_text_only:
            # Text-only inference
            messages = [{"role": "user", "content": input_data[0]["text_input"]}]
            response = self._generate_text_response(messages)
            results = [response]
        else:
            # Image-based inference
            file_paths = self._extract_file_paths(input_data)
            results = self._process_images(file_paths, input_data)
        
        return results

    def _generate_text_response(self, messages):
        """
        Generate a text response for text-only inputs.
        
        :param messages: List of message dictionaries
        :return: Generated response
        """
        try:
            client = openai.OpenAI(api_key=self.api_key) if hasattr(self, "api_key") and self.api_key else openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            result = response.choices[0].message.content
            print("Text inference completed successfully")
            return result
        except Exception as e:
            print(f"Error in text inference: {e}")
            return f"Error: {str(e)}"

    def _process_images(self, file_paths, input_data):
        """
        Process images and generate responses for each.
        
        :param file_paths: List of image file paths
        :param input_data: Original input data
        :return: List of processed responses
        """
        results = []
        
        for file_path in file_paths:
            try:
                # Validate file exists
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} does not exist")
                    results.append(json.dumps({"error": f"File {file_path} not found"}))
                    continue

                # Encode image to base64
                base64_image = self.encode_image_to_base64(file_path)
                
                # Prepare messages for image analysis
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": input_data[0]["text_input"]
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]

                # Generate response
                client = openai.OpenAI(api_key=self.api_key) if hasattr(self, "api_key") and self.api_key else openai.OpenAI()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "summary",
                            "schema": {
                                "type": "object",
                                "properties": {field: {"type": "string"} for field in self.fields} if hasattr(self, "fields") and self.fields else {},
                                "required": list(self.fields) if hasattr(self, "fields") and self.fields else [],
                                "additionalProperties": False
                            }
                        }
                    }
                )
                
                result = response.choices[0].message.content
                processed_response = self.process_response(result)
                results.append(processed_response)
                
                print(f"Image inference completed successfully for: {file_path}")
                
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
                results.append(json.dumps({"error": f"Failed to process {file_path}: {str(e)}"}))

        return results

    @staticmethod
    def _extract_file_paths(input_data):
        """
        Extract and resolve absolute file paths from input data.

        :param input_data: List of dictionaries containing image file paths.
        :return: List of absolute file paths.
        """
        file_paths = []
        for data in input_data:
            file_path = data.get("file_path")
            if file_path:
                if isinstance(file_path, list):
                    file_paths.extend([os.path.abspath(path) for path in file_path])
                else:
                    file_paths.append(os.path.abspath(file_path))
        return file_paths
