

from settings import sparrow_settings as django_settings
from settings import *

init_django()

from betterbrain.sql.models import *

# from django.conf import settings


import json
from sparrow_parse.vllm.inference_factory import InferenceFactory
from sparrow_parse.vllm.openai_inference import OpenAIInference
from sparrow_parse.extractors.vllm_extractor import VLLMExtractor
from sparrow_parse.vllm.utils.combine_answers import combine_json_results

import os
import pprint

def get_structured_info_from_file(
    file_id: int
):
    file = File.objects.get(id=file_id)

    # if file.type != FileType.PDF:
    #     raise ValueError(f"File {file_id} is not a PDF")

    # Initialize extractor
    extractor = VLLMExtractor()

    # Configure OpenAI backend
    config = {
        "method": "openai",
        "model_name": "gpt-4o",
        "max_tokens": 4000,
        "temperature": 0.0,
        "api_key": django_settings.OPENAI_API_KEY
    }

    # Create inference instance
    factory = InferenceFactory(config)
    model_inference_instance = factory.get_inference_instance()

    return {
        'clause1': 'clause1',
        'clause2': 'clause2',
        'clause3': 'clause3',
        'clause4': 'clause4',
        'clause5': 'clause5',
        'clause6': 'clause6',
        'clause7': 'clause7',
    }

    # # multi-page test
    # print("\n=== PDF PROCESSING ===")
    # pdf_input_data = [{
    #     "file_path": f"{django_settings.ROOT_DIR}/submodules/sparrow/sparrow-data/parse/sparrow_parse/images/20240101_Services Agreement_ExcelSports.pdf",  # Replace with your PDF path
    #     "text_input": "Extract major clause summaries from this agreement. Return the data in JSON format with fields: intellectual_property, retainer_fee, confidentiality, non_disparagement, termination, indemnification, assignment, warranties, liability, arbitration, choice_of_law, and miscellaneous. If information is not present, return null for that field."
    # }]

    # factory = InferenceFactory(config, fields=["intellectual_property", "retainer_fee", "confidentiality", "non_disparagement", "termination", "indemnification", "assignment", "warranties", "liability", "arbitration", "choice_of_law", "miscellaneous"])
    # model_inference_instance = factory.get_inference_instance()

    # pdf_results, pdf_num_pages = extractor.run_inference(
    #     model_inference_instance, 
    #     pdf_input_data,
    #     debug=True
    # )

    # final_result = combine_json_results(pdf_results)
    # pprint.pprint(final_result)
