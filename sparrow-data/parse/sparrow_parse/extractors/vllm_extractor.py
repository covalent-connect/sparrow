import json
from betterbrain.sql.models.files.file import FileType
from sparrow_parse.vllm.inference_factory import InferenceFactory
from sparrow_parse.helpers.pdf_optimizer import PDFOptimizer
from sparrow_parse.helpers.image_optimizer import ImageOptimizer
from sparrow_parse.processors.table_structure_processor import TableDetector
from sparrow_parse.vllm.utils.data_type import DataType


from rich import print
import os
import tempfile
import shutil
import io
from typing import Union, 
from typing import List, Dict, Tuple

from utils.embedding.indexing.utils.is_image import is_file_type_image


class VLLMExtractor(object):
    def __init__(self):
        pass

    def run_inference(self, model_inference_instance, input_data, tables_only=False,
                      generic_query=False, crop_size=None, apply_annotation=False, debug_dir=None, debug=False, mode=None) -> Tuple[List[Dict], int]:
        """
        Main entry point for processing input data using a model inference instance.
        Handles generic queries, PDFs, and table extraction.
        Supports file_path as a filename, bytes, or BytesIO.
        """
        if generic_query:
            input_data[0]["text_input"] = "retrieve document data. return response in JSON format"
            apply_annotation = False

        if debug:
            print("Input data:", input_data)

        # Handle both missing file_path and file_path=None as text-only inference
        # is_text_only = "file_path" not in input_data[0] or input_data[0]["file_path"] is None

        is_text_only = input_data[0]["is_text_only"]

        if is_text_only:
            # Ensure file_path exists and is None for consistency
            input_data[0]["file_path"] = None
            results = model_inference_instance.inference(input_data)
            return results, 0

        # Support file_path as filename, bytes, or BytesIO
        file_type = input_data[0]["file_type"]
        data_type = input_data[0]["data_type"]

        print(f"File type: {file_type}")
        print(f"Data type: {data_type}")

        tmp_path_on_disk = None
        if data_type == DataType.FILE_PATH:
            # load the file and turn it into bytes object
            file_path = input_data[0]["content"]
        elif data_type == DataType.FILE_BYTES:
            if isinstance(file_type, str):
                suffix = f'.{file_type.lower()}'
            else:
                suffix = f'.{file_type.value.lower()}'
            tmp_path, tmp_path_on_disk = self._ensure_file_on_disk(
                input_data[0]["content"], 
                suffix=suffix
            )
            file_path = tmp_path_on_disk
        elif data_type == DataType.TEXT_INPUT:
            pass
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        input_data[0]["file_path"] = file_path

        try:
            if is_file_type_image(file_type):
                print(f"Is image")
                return self._process_non_pdf(
                    model_inference_instance, 
                    input_data, 
                    tables_only, 
                    crop_size, 
                    apply_annotation, 
                    debug, 
                    debug_dir, 
                    file_path=file_path
                )
            elif file_type == FileType.PDF:
                print(f"Is PDF")
                return self._process_pdf(
                    model_inference_instance, 
                    input_data, 
                    tables_only, 
                    crop_size, 
                    apply_annotation, 
                    debug, 
                    debug_dir, 
                    mode, 
                    file_path=file_path
                )
            # TODO: Handle docx, doc, txt, any other file types we need to
        finally:
            if tmp_path_on_disk is not None:
                try:
                    os.remove(tmp_path_on_disk)
                except Exception:
                    pass


        # if data_type == DataType.FILE_PATH:
        #     file_path_obj = input_data[0]["file_path"]
        # elif data_type == DataType.FILE_BYTES:
        #     file_path_obj = input_data[0]["content"]
        # else:
        #     raise ValueError(f"Invalid data type: {data_type}")
        # file_path, temp_file = self._ensure_file_on_disk(file_path_obj, suffix=self._guess_file_suffix(file_path_obj))
        # try:
        #     if self.is_pdf(file_path):
        #         return self._process_pdf(model_inference_instance, input_data, tables_only, crop_size, apply_annotation, debug, debug_dir, mode, file_path=file_path)
        #     else:
        #         return self._process_non_pdf(model_inference_instance, input_data, tables_only, crop_size, apply_annotation, debug, debug_dir, file_path=file_path)
        # finally:
        #     if temp_file is not None:
        #         try:
        #             os.remove(temp_file)
        #         except Exception:
        #             pass

    def _ensure_file_on_disk(self, file_path_obj: Union[str, bytes, io.BytesIO], suffix=None):
        """
        Ensures that the input is a file path on disk.
        If file_path_obj is a string (filename), returns it directly.
        If it's bytes or BytesIO, writes to a temp file and returns the path.
        Returns (file_path, temp_file_path or None)
        """
        if isinstance(file_path_obj, str):
            return file_path_obj, None
        elif isinstance(file_path_obj, bytes):
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix or "")
            with os.fdopen(temp_fd, "wb") as f:
                f.write(file_path_obj)
            return temp_path, temp_path
        elif isinstance(file_path_obj, io.BytesIO):
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix or "")
            with os.fdopen(temp_fd, "wb") as f:
                f.write(file_path_obj.getvalue())
            return temp_path, temp_path
        else:
            raise ValueError("file_path must be a filename, bytes, or BytesIO object")

    def _guess_file_suffix(self, file_path_obj):
        """
        Guess the file suffix for temp file creation.
        """
        if isinstance(file_path_obj, str):
            _, ext = os.path.splitext(file_path_obj)
            return ext
        # If bytes or BytesIO, default to .pdf (could be improved)
        return ".pdf"

    def _process_pdf(self, model_inference_instance, input_data, tables_only, crop_size, apply_annotation, debug, debug_dir, mode, file_path=None):
        """
        Handles processing and inference for PDF files, including page splitting and optional table extraction.
        Accepts file_path as a filename.
        """
        pdf_optimizer = PDFOptimizer()
        num_pages, output_files, temp_dir = pdf_optimizer.split_pdf_to_pages(
            file_path or input_data[0]["file_path"],
            debug_dir, convert_to_images=True
        )

        # Update input_data[0]["file_path"] to the file_path used
        input_data[0]["file_path"] = file_path or input_data[0]["file_path"]

        results = self._process_pages(model_inference_instance, output_files, input_data, tables_only, crop_size, apply_annotation, debug, debug_dir)

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        return results, num_pages

    def _process_non_pdf(self, model_inference_instance, input_data, tables_only, crop_size, apply_annotation, debug, debug_dir, file_path=None):
        """
        Handles processing and inference for non-PDF files, with optional table extraction.
        Accepts file_path as a filename.
        """
        file_path = file_path or input_data[0]["file_path"]

        if tables_only:
            return self._extract_tables(model_inference_instance, file_path, input_data, apply_annotation, debug, debug_dir), 1
        else:
            temp_dir = tempfile.mkdtemp()

            if crop_size:
                if debug:
                    print(f"Cropping image borders by {crop_size} pixels.")
                image_optimizer = ImageOptimizer()
                cropped_file_path = image_optimizer.crop_image_borders(file_path, temp_dir, debug_dir, crop_size)
                input_data[0]["file_path"] = cropped_file_path
                file_path = cropped_file_path

            input_data[0]["file_path"] = [file_path]
            results = model_inference_instance.inference(input_data, apply_annotation)

            shutil.rmtree(temp_dir, ignore_errors=True)

            return results, 1

    def _process_pages(self, model_inference_instance, output_files, input_data, tables_only, crop_size, apply_annotation, debug, debug_dir):
        """
        Processes individual pages (PDF split) and handles table extraction or inference.

        Args:
            model_inference_instance: The model inference object.
            output_files: List of file paths for the split PDF pages.
            input_data: Input data for inference.
            tables_only: Whether to only process tables.
            crop_size: Size for cropping image borders.
            apply_annotation: Flag to apply annotations to the output.
            debug: Debug flag for logging.
            debug_dir: Directory for saving debug information.

        Returns:
            List of results from the processing or inference.
        """
        results_array = []

        if tables_only:
            if debug:
                print(f"Processing {len(output_files)} pages for table extraction.")
            # Process each page individually for table extraction
            for i, file_path in enumerate(output_files):
                tables_result = self._extract_tables(model_inference_instance, file_path, input_data, apply_annotation, debug, debug_dir, page_index=i)
                # Since _extract_tables returns a list with one JSON string, unpack it
                results_array.extend(tables_result)  # Unpack the single JSON string
        else:
            if debug:
                print(f"Processing {len(output_files)} pages for inference at once.")

            temp_dir = tempfile.mkdtemp()
            cropped_files = []

            if crop_size:
                if debug:
                    print(f"Cropping image borders by {crop_size} pixels from {len(output_files)} images.")

                image_optimizer = ImageOptimizer()

                # Process each file in the output_files array
                for file_path in output_files:
                    cropped_file_path = image_optimizer.crop_image_borders(
                        file_path,
                        temp_dir,
                        debug_dir,
                        crop_size
                    )
                    cropped_files.append(cropped_file_path)

                # Use the cropped files for inference
                input_data[0]["file_path"] = cropped_files
            else:
                # If no cropping needed, use original files directly
                input_data[0]["file_path"] = output_files

            # Process all files at once
            results = model_inference_instance.inference(input_data, apply_annotation)
            results_array.extend(results)

            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        return results_array

    def _extract_tables(self, model_inference_instance, file_path, input_data, apply_annotation, debug, debug_dir, page_index=None):
        """
        Detects and processes tables from an input file.
        """
        table_detector = TableDetector()
        cropped_tables = table_detector.detect_tables(file_path, local=False, debug_dir=debug_dir, debug=debug)
        results_array = []

        # Check if no tables were found
        if cropped_tables is None:
            if debug:
                print(f"No tables detected in {file_path}")
            # Return a structured no-tables-found response instead of failing
            return [json.dumps({"message": "No tables detected in the document", "status": "empty"})]

        temp_dir = tempfile.mkdtemp()

        for i, table in enumerate(cropped_tables):
            table_index = f"page_{page_index + 1}_table_{i + 1}" if page_index is not None else f"table_{i + 1}"
            print(f"Processing {table_index} for document {file_path}")

            output_filename = os.path.join(temp_dir, f"{table_index}.jpg")
            table.save(output_filename, "JPEG")

            input_data[0]["file_path"] = [output_filename]
            result = self._run_model_inference(model_inference_instance, input_data, apply_annotation)
            results_array.append(result)

        shutil.rmtree(temp_dir, ignore_errors=True)

        # Merge results_array elements into a single JSON structure
        merged_results = {"page_tables": results_array}

        # Format the merged results as a JSON string with indentation
        formatted_results = json.dumps(merged_results, indent=4)

        # Return the formatted JSON string wrapped in a list
        return [formatted_results]

    @staticmethod
    def _run_model_inference(model_inference_instance, input_data, apply_annotation):
        """
        Runs model inference and handles JSON decoding.
        """
        result = model_inference_instance.inference(input_data, apply_annotation)[0]
        try:
            return json.loads(result) if isinstance(result, str) else result
        except json.JSONDecodeError:
            return {"message": "Invalid JSON format in LLM output", "valid": "false"}

    @staticmethod
    def is_pdf(file_path):
        """Checks if a file is a PDF based on its extension."""
        if not isinstance(file_path, str):
            # If file_path is not a string, we can't check extension, assume PDF for bytes/BytesIO
            return True
        return file_path.lower().endswith('.pdf')


if __name__ == "__main__":
    # run locally: python -m sparrow_parse.extractors.vllm_extractor

    extractor = VLLMExtractor()

    # # export HF_TOKEN="hf_"
    # config = {
    #     "method": "mlx",  # Could be 'huggingface', 'mlx' or 'local_gpu'
    #     "model_name": "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit",
    #     # "hf_space": "katanaml/sparrow-qwen2-vl-7b",
    #     # "hf_token": os.getenv('HF_TOKEN'),
    #     # Additional fields for local GPU inference
    #     # "device": "cuda", "model_path": "model.pth"
    # }
    #
    # # Use the factory to get the correct instance
    # factory = InferenceFactory(config)
    # model_inference_instance = factory.get_inference_instance()
    #
    # input_data = [
    #     {
    #         "file_path": "sparrow_parse/images/bonds_table.png",
    #         "text_input": "retrieve [{\"instrument_name\":\"str\", \"valuation\":\"int\"}]. return response in JSON format"
    #     }
    # ]
    #
    # # input_data = [
    # #     {
    # #         "file_path": None,
    # #         "text_input": "why earth is spinning around the sun?"
    # #     }
    # # ]
    #
    # # Now you can run inference without knowing which implementation is used
    # results_array, num_pages = extractor.run_inference(model_inference_instance, input_data, tables_only=False,
    #                                                    generic_query=False,
    #                                                    crop_size=0,
    #                                                    apply_annotation=False,
    #                                                    debug_dir="/Users/andrejb/Work/katana-git/sparrow/sparrow-ml/llm/data/",
    #                                                    debug=True,
    #                                                    mode=None)
    #
    # for i, result in enumerate(results_array):
    #     print(f"Result for page {i + 1}:", result)
    # print(f"Number of pages: {num_pages}")
