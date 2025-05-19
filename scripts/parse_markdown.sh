#!/bin/bash

# RegulAIte - Direct Markdown Parser CLI
# This script sends a document directly to the Unstructured API container
# and prints the parsed output to the terminal

set -e

# Default values
API_URL="http://localhost:9900/general/v0/general"
STRATEGY="auto"
OCR_ENABLED="true"
LANGUAGES="auto"
INCLUDE_PAGE_BREAKS="true"
HIERARCHICAL_PDF="true"
EXTRACT_IMAGES="true"
EXTRACT_TABLES="true"
INCLUDE_METADATA="true"
FORMAT="json"  # Default output format: json, text, or elements

# Function to display usage information
show_help() {
    echo "Usage: $(basename "$0") [options] file_path"
    echo
    echo "Parse a Markdown (or other) document using the Unstructured API and print results to terminal."
    echo
    echo "Options:"
    echo "  -h, --help                Show this help message and exit"
    echo "  -s, --strategy STRATEGY   Parsing strategy: auto, hi_res, fast, ocr_only (default: auto)"
    echo "  -o, --ocr BOOLEAN         Enable OCR: true, false (default: true)"
    echo "  -l, --languages LANGS     Languages for OCR: auto, eng, deu, etc. (default: auto)"
    echo "  -p, --page-breaks BOOL    Include page breaks: true, false (default: true)"
    echo "  -i, --images BOOL         Extract images: true, false (default: true)"
    echo "  -t, --tables BOOL         Extract tables: true, false (default: true)"
    echo "  -m, --metadata BOOL       Include metadata: true, false (default: true)"
    echo "  -f, --format FORMAT       Output format: json, text, elements (default: json)"
    echo "  -u, --url URL             API URL (default: http://localhost:9900/general/v0/general)"
    echo
    echo "Example:"
    echo "  $(basename "$0") document.md"
    echo "  $(basename "$0") --format text document.pdf"
    echo "  inputs git:(documents) ✗ ../scripts/parse_markdown.sh -s auto -o true -l auto -p true -m true -f json originals/'règlement général de protection des données (RGPD)_CELEX_32016R0679_FR_LlamaParse.md' > './train/inputs/RGPD/LlamaParse Cloud + unstructured local/règlement général de protection des données (RGPD)_CELEX_32016R0679_FR_LlamaParse Cloud + unstructured local.json'"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -s|--strategy)
            STRATEGY="$2"
            shift 2
            ;;
        -o|--ocr)
            OCR_ENABLED="$2"
            shift 2
            ;;
        -l|--languages)
            LANGUAGES="$2"
            shift 2
            ;;
        -p|--page-breaks)
            INCLUDE_PAGE_BREAKS="$2"
            shift 2
            ;;
        -i|--images)
            EXTRACT_IMAGES="$2"
            shift 2
            ;;
        -t|--tables)
            EXTRACT_TABLES="$2"
            shift 2
            ;;
        -m|--metadata)
            INCLUDE_METADATA="$2"
            shift 2
            ;;
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        -u|--url)
            API_URL="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            ;;
        *)
            FILE_PATH="$1"
            shift
            ;;
    esac
done

# Check if a file path was provided
if [ -z "$FILE_PATH" ]; then
    echo "Error: No file path provided"
    show_help
fi

# Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File not found: $FILE_PATH"
    exit 1
fi

# Get the filename for display
# FILENAME=$(basename "$FILE_PATH")

# echo "Parsing file: $FILENAME"
# echo "API URL: $API_URL"
# echo "Parameters:"
# echo "  - Strategy: $STRATEGY"
# echo "  - OCR Enabled: $OCR_ENABLED"
# echo "  - Languages: $LANGUAGES"
# echo "  - Include Page Breaks: $INCLUDE_PAGE_BREAKS"
# echo "  - Extract Images: $EXTRACT_IMAGES"
# echo "  - Extract Tables: $EXTRACT_TABLES"
# echo "  - Include Metadata: $INCLUDE_METADATA"
# echo "  - Output Format: $FORMAT"
# echo

# Call the Unstructured API
response=$(curl -s -X POST "$API_URL" \
    -H "Accept: application/json" \
    -F "files=@$FILE_PATH" \
    -F "strategy=$STRATEGY" \
    -F "ocr_enabled=$OCR_ENABLED" \
    -F "languages=$LANGUAGES" \
    -F "include_page_breaks=$INCLUDE_PAGE_BREAKS" \
    -F "hierarchical_pdf=$HIERARCHICAL_PDF" \
    -F "extract_images=$EXTRACT_IMAGES" \
    -F "extract_tables=$EXTRACT_TABLES" \
    -F "include_metadata=$INCLUDE_METADATA")

# Check if the request was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to call the Unstructured API"
    exit 1
fi

# Process the output based on the format
case "$FORMAT" in
    json)
        # Pretty print the JSON output
        echo "$response" | jq
        ;;
    text)
        # Extract only the text from the elements
        echo "$response" | jq -r '.[].text | select(. != null and . != "")' | sed '/^$/d'
        ;;
    elements)
        # Print summary of the elements
        count=$(echo "$response" | jq '. | length')
        echo "Total elements: $count"
        echo
        
        # Count elements by type
        echo "Element types:"
        echo "$response" | jq -r '.[].type' | sort | uniq -c | sort -nr
        echo
        
        # Show the first few elements of each type (up to 3 of each)
        echo "Sample elements:"
        types=$(echo "$response" | jq -r '.[].type' | sort | uniq)
        
        for type in $types; do
            echo "  $type:"
            elements=$(echo "$response" | jq -r --arg type "$type" '.[] | select(.type == $type) | .text' | head -n 3)
            echo "$elements" | sed 's/^/    /' | sed 's/\\n/\n    /g' | head -c 500
            echo "..."
            echo
        done
        ;;
    *)
        echo "Unknown format: $FORMAT"
        exit 1
        ;;
esac
