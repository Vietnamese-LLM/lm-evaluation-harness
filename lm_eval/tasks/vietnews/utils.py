import datasets
from rouge_score import rouge_scorer

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _helper(doc):
        # Clean underscores to spaces for all text fields
        # In the python script: return text.replace("_", " ").strip()
        doc["title"] = doc["title"].replace("_", " ").strip()
        doc["abstract"] = doc["abstract"].replace("_", " ").strip()
        doc["article"] = doc["article"].replace("_", " ").strip()
        
        # Heuristic truncation to prevent extremely long documents from causing OOM
        # The original script used max_length=2048 tokens.
        # Assuming ~4 chars per token, 2048 * 4 = 8192. We'll use 10000 chars as a safe upper bound.
        # This helps stability with vLLM/engine backends.
        if len(doc["article"]) > 10000:
            doc["article"] = doc["article"][:10000]
            
        return doc
        
    return dataset.map(_helper)

ROUGE_SCORER = None

def rouge(references, predictions, **kwargs):
    """
    Returns ROUGE scores.
    """
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    
    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        ROUGE_SCORER = rouge_scorer.RougeScorer(rouge_types)
    scorer = ROUGE_SCORER
    
    # When called by metric_list in ConfigurableTask, references and predictions are lists of length 1 (per doc)
    # We should return the score for that single document.
    # Aggregation (mean) is handled by the harness.
    
    if len(references) == 1 and len(predictions) == 1:
        scores = scorer.score(references[0], predictions[0])
        return {type: scores[type].fmeasure for type in rouge_types}
    else:
        # Fallback for batch processing if ever used
        results = []
        for ref, pred in zip(references, predictions):
             scores = scorer.score(ref, pred)
             results.append({type: scores[type].fmeasure for type in rouge_types})
        return results

def rouge1(references, predictions, **kwargs):
    return rouge(references, predictions)["rouge1"]

def rouge2(references, predictions, **kwargs):
    return rouge(references, predictions)["rouge2"]

def rougeL(references, predictions, **kwargs):
    return rouge(references, predictions)["rougeL"]
