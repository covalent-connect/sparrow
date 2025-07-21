
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

# Example 1: Image Analysis (existing)
# print("=== IMAGE ANALYSIS ===")
# input_data = [{
#     "file_path": "sparrow_parse/images/bonds_table.png",
#     "text_input": "Extract the bond information from this table. Return the data in JSON format with fields: instrument_name, valuation, and maturity_date."
# }]

# results, num_pages = extractor.run_inference(
#     model_inference_instance, 
#     input_data,
#     debug=True
# )

# print(f"Extracted data: {results[0]}")
# print(f"Number of pages: {num_pages}")

# new test
# print("\n=== PDF PROCESSING ===")
# pdf_input_data = [{
#     "file_path": "sparrow_parse/images/abhishek_resume.pdf",  # Replace with your PDF path
#     "text_input": "Extract experiences from this resume. Return the data in JSON format with fields: company_name, title, start_date, end_date, description."
# }]

# pdf_results, pdf_num_pages = extractor.run_inference(
#     model_inference_instance, 
#     pdf_input_data,
#     debug=True
# )

# print(f"PDF extracted data:")
# pprint.pprint(pdf_results)
# for pdf_result in pdf_results:
#     pdf_result = json.loads(pdf_result)
#     pprint.pprint(pdf_result)
# print(f"Number of PDF pages: {pdf_num_pages}")

# multi-page test
print("\n=== PDF PROCESSING ===")
pdf_input_data = [{
    "file_path": f"{django_settings.ROOT_DIR}/submodules/sparrow/sparrow-data/parse/sparrow_parse/images/20240101_Services Agreement_ExcelSports.pdf",  # Replace with your PDF path
    "text_input": "Extract major clause summaries from this agreement. Return the data in JSON format with fields: intellectual_property, retainer_fee, confidentiality, non_disparagement, termination, indemnification, assignment, warranties, liability, arbitration, choice_of_law, and miscellaneous. If information is not present, return null for that field."
}]

factory = InferenceFactory(config, fields=["intellectual_property", "retainer_fee", "confidentiality", "non_disparagement", "termination", "indemnification", "assignment", "warranties", "liability", "arbitration", "choice_of_law", "miscellaneous"])
model_inference_instance = factory.get_inference_instance()

pdf_results, pdf_num_pages = extractor.run_inference(
    model_inference_instance, 
    pdf_input_data,
    debug=True
)

final_result = combine_json_results(pdf_results)
pprint.pprint(final_result)

# print(f"PDF extracted data:")
# pprint.pprint(pdf_results)
# ctr = 1
# for pdf_result in pdf_results:
#     print(f"Page {ctr}:")
#     pdf_result = json.loads(pdf_result)
#     pprint.pprint(pdf_result)
#     ctr += 1
# print(f"Number of PDF pages: {pdf_num_pages}")



# Example 2: PDF Processing (new)
# print("\n=== PDF PROCESSING ===")
# pdf_input_data = [{
#     "file_path": "sparrow_parse/images/oracle_10k_2024_q1_small.pdf",  # Replace with your PDF path
#     "text_input": "Extract financial data from this document. Return key metrics in JSON format with fields: revenue, net_income, total_assets."
# }]

# pdf_results, pdf_num_pages = extractor.run_inference(
#     model_inference_instance, 
#     pdf_input_data,
#     debug=True
# )

# print(f"PDF extracted data: {pdf_results[0]}")
# print(f"Number of PDF pages: {pdf_num_pages}")

# Example 3: PDF with Table Extraction Only
# print("\n=== PDF TABLE EXTRACTION ===")
# pdf_tables_input = [{
#     "file_path": "sparrow_parse/images/oracle_10k_2024_q1_small.pdf",  # Replace with your PDF path
#     "text_input": "Extract all tables from this document. Return table data in JSON format."
# }]

# pdf_tables_results, pdf_tables_num_pages = extractor.run_inference(
#     model_inference_instance, 
#     pdf_tables_input,
#     tables_only=True,  # This is the key parameter for table extraction
#     debug=True
# )

# print(f"PDF tables extracted: {pdf_tables_results[0]}")
# print(f"Number of PDF pages with tables: {pdf_tables_num_pages}")

# Example 4: PDF with Generic Query
# print("\n=== PDF GENERIC QUERY ===")
# pdf_generic_input = [{
#     "file_path": "sparrow_parse/images/oracle_10k_2024_q1_small.pdf",  # Replace with your PDF path
#     "text_input": "Analyze this document and extract key information."  # This will be overridden
# }]

# pdf_generic_results, pdf_generic_num_pages = extractor.run_inference(
#     model_inference_instance, 
#     pdf_generic_input,
#     generic_query=True,  # This overrides the text_input with a generic prompt
#     debug=True
# )

# print(f"PDF generic analysis: {pdf_generic_results[0]}")
# print(f"Number of PDF pages: {pdf_generic_num_pages}")

# Example 5: Text-only inference (existing)
# print("\n=== TEXT-ONLY INFERENCE ===")
# text_input_data = [{
#     "file_path": None,
#     "text_input": "Explain the concept of machine learning in simple terms."
# }]

# text_results, _ = extractor.run_inference(
#     model_inference_instance, 
#     text_input_data,
#     debug=True
# )

# print(f"Text response: {text_results[0]}") 

# python -m submodules.sparrow.sparrow-data.parse.sparrow_parse.testing.test_openai_inference