
import os
import sys
import logging
import requests
import tempfile
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def test_unstructured_connection(file_path, api_url=None):
    """
    Test connection to Unstructured API with a specific file.

    Args:
        file_path: Path to file to test
        api_url: Optional Unstructured API URL
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    # Default URL if not provided
    if not api_url:
        api_url = os.getenv("UNSTRUCTURED_API_URL", "http://localhost:9900/general/v0/general")

    logger.info(f"Testing Unstructured API connection: {api_url}")
    logger.info(f"Using file: {file_path}")

    # Get file content
    with open(file_path, "rb") as f:
        file_content = f.read()

    # Setup request
    headers = {
        "Accept": "application/json",
        "unstructured-api-key": os.getenv("UNSTRUCTURED_API_KEY", "")
    }

    # Try different parameter combinations
    parameter_sets = [
        {
            "strategy": "auto",
            "ocr_enabled": "true"
        },
        {
            "strategy": "hi_res",
            "ocr_enabled": "true",
            "ocr_languages": "eng",
            "hi_res_model_name": "yolox"
        },
        {
            "strategy": "fast",
            "ocr_enabled": "true",
            "ocr_languages": "eng",
        }
    ]

    success = False

    for i, params in enumerate(parameter_sets):
        logger.info(f"Trying parameter set {i+1}: {params}")

        with open(file_path, "rb") as f:
            files = {"files": (os.path.basename(file_path), f)}

            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    files=files,
                    data=params
                )

                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    continue

                elements = response.json()
                logger.info(f"Received {len(elements)} elements from API")

                # Check for text elements
                text_elements = [e for e in elements if e.get("text", "").strip()]
                logger.info(f"Found {len(text_elements)} elements with text")

                if text_elements:
                    success = True
                    logger.info("Successfully extracted text from document!")
                    logger.info(f"Parameter set {i+1} worked!")

                    # Log some sample text
                    for j, elem in enumerate(text_elements[:3]):
                        text_preview = elem.get("text", "")[:100].replace("\n", " ")
                        logger.info(f"Sample text {j+1}: {text_preview}...")

                    # Return the best working parameters
                    return params
                else:
                    logger.warning("No text elements found in the response.")

                    # Log element types received
                    type_counts = {}
                    for elem in elements:
                        elem_type = elem.get("type", "unknown")
                        type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
                    logger.warning(f"Element type counts: {type_counts}")

            except Exception as e:
                logger.error(f"Error calling API: {str(e)}")

    if not success:
        logger.error("Failed to extract text with any parameter set.")
        logger.info("Recommendations:")
        logger.info("1. Check if Unstructured API container is running properly")
        logger.info("2. Verify the PDF is not corrupted or password-protected")
        logger.info("3. Check if the PDF contains actual text or only images")
        logger.info("4. Try pre-processing the PDF with OCR software")

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Unstructured API connection")
    parser.add_argument("file_path", help="Path to the file to test")
    parser.add_argument("--api_url", help="Unstructured API URL", default=None)

    args = parser.parse_args()

    best_params = test_unstructured_connection(args.file_path, args.api_url)

    if best_params:
        print("\nBest working parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print("\nUse these parameters in your _call_unstructured_api method")
