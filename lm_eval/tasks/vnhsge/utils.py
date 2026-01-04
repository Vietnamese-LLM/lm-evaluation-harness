import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _helper(doc):
        # Filter out questions with images (empty strings or None are considered "no image")
        # In the python script: if (not q.get("Image_Question")) and (not q.get("Image_Answer"))
        img_q = doc.get("Image_Question")
        img_a = doc.get("Image_Answer")
        
        has_img_q = img_q is not None and len(img_q) > 0
        has_img_a = img_a is not None and len(img_a) > 0
        
        if has_img_q or has_img_a:
            return False
        return True
        
    return dataset.filter(_helper)

