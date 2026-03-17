import argparse
from oemer import ete
import oemer.symbol_extraction

# I save the reference to the original function
original_merge_nearby_bbox = oemer.symbol_extraction.merge_nearby_bbox

# I create a safe wrapper to handle edge cases with 0 or 1 items
def safe_merge_nearby_bbox(bboxes, distance, x_factor=1, y_factor=1):
    # I return immediately if clustering is impossible
    if len(bboxes) <= 1:
        return bboxes
    
    # I run the original clustering algorithm if there are 2 or more items
    return original_merge_nearby_bbox(bboxes, distance, x_factor, y_factor)

# I inject the safe function directly into the symbol_extraction module
oemer.symbol_extraction.merge_nearby_bbox = safe_merge_nearby_bbox

def generate_musicxml(image_path: str, output_dir: str = "./") -> str:
    # I clear previous session data from memory
    ete.clear_data()

    # I configure the extraction parameters
    args = argparse.Namespace(
        img_path=image_path,
        output_path=output_dir,
        use_tf=False,
        save_cache=False,
        without_deskew=True  # I skip deskewing to optimize for clear digital images
    )

    # I trigger the end-to-end OMR pipeline
    mxl_path = ete.extract(args)

    return mxl_path