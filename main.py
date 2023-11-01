import argparse
import os

from extractor import get_roi, extract_text, get_images_from_pdf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

default_path = "samples.pdf"

parser.add_argument("-f", "--file", default=default_path, help="path to .pdf document")
args = vars(parser.parse_args())

doc_path = args["file"]

if not os.path.exists(doc_path):
    raise IOError("File doesn't exist!")

if doc_path == default_path:
    print("HINT: Pass file path after -f or --file flag to provide path to your .pdf file.")

images = get_images_from_pdf(doc_path)
        
roi_collection = get_roi(images)

if roi_collection:
    for i, roi in roi_collection:
        text = extract_text(roi)
        print(f"SIGNATURE FOUNDED ON PAGE {i}, TEXT IS FOLLOWING:\n{text}")
        print("-" * 100)