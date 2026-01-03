import os
import logging
import concurrent.futures
from tqdm import tqdm

eval_logger = logging.getLogger(__name__)

def build_judge_prompt(context, question, gold_answers, pred_answer):
    """
    Constructs the prompt for the judge model.
    """
    if isinstance(gold_answers, dict):
        gold_list = gold_answers.get("text", [])
    elif isinstance(gold_answers, list):
        gold_list = gold_answers
    else:
        gold_list = [gold_answers]

    gold_joined = "\n- ".join(gold_list) if gold_list else "(không có)"

    prompt = f"""Bạn là trợ lý đánh giá câu trả lời tiếng Việt cho bài toán hỏi đáp đọc hiểu.

Nhiệm vụ của bạn:
- Đọc đoạn văn, câu hỏi, danh sách câu trả lời tham chiếu (ground truth) và câu trả lời của mô hình.
- Chấm điểm ĐỘ ĐÚNG NGHĨA của câu trả lời mô hình so với các câu trả lời tham chiếu.
- Cho điểm từ 0 đến 1:
  - 1.0: nghĩa tương đương hoặc chỉ khác biệt nhỏ, chấp nhận được.
  - 0.5: đúng một phần, còn thiếu hoặc hơi sai.
  - 0.0: sai hoàn toàn, không liên quan, hoặc mâu thuẫn.
- Chỉ xuất ra MỘT SỐ thực duy nhất trong khoảng [0, 1], với tối đa 2 chữ số sau dấu phẩy.
- Không giải thích thêm, không in text nào khác.

Đoạn văn:
{context}

Câu hỏi:
{question}

Các câu trả lời tham chiếu:
- {gold_joined}

Câu trả lời của mô hình:
{pred_answer}

Hãy cho điểm (0 đến 1) mức độ đúng nghĩa của câu trả lời mô hình so với các câu trả lời tham chiếu.
Chỉ in duy nhất một số thực trong khoảng [0, 1].
"""
    return prompt

def call_judge_api(prompt):
    """
    Calls the judge model using OpenAI API (Worker function).
    """
    try:
        from openai import OpenAI
    except ImportError:
        return 0.0

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("JUDGE_BASE_URL")
    model = os.getenv("JUDGE_MODEL", "gpt-4o")
    
    if not api_key:
        return 0.0

    # Instantiate client per call to be safe with threads, or use a global one
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=16,
        )
        content = response.choices[0].message.content.strip()
        
        # Parse float
        for token in content.replace(",", ".").split():
            try:
                score = float(token)
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
    except Exception as e:
        eval_logger.error(f"Judge API error: {e}")
        
    return 0.0

def process_results(doc, results):
    """
    Prepare data for judging. 
    Instead of calling API here, we return the PROMPT.
    The aggregation function will handle the execution.
    """
    pred_answer = results[0].strip()
    context = doc["context"]
    question = doc["question"]
    gold_answers = doc["answers"]

    prompt = build_judge_prompt(context, question, gold_answers, pred_answer)

    return {
        "judge_score": prompt, # Return PROMPT, not score
        "exact_match": 1.0 if pred_answer in gold_answers.get("text", []) else 0.0
    }

def judge_aggregate(items):
    """
    Aggregation function: Receives a list of PROMPTS.
    Runs them in parallel and returns the mean score.
    """
    prompts = items
    if not prompts:
        return 0.0
        
    # Check API key once
    if not os.getenv("OPENAI_API_KEY"):
        eval_logger.warning("OPENAI_API_KEY not set. Returning 0.0 for judge score.")
        return 0.0

    # Number of threads (adjust based on your rate limits)
    max_workers = int(os.getenv("JUDGE_MAX_WORKERS", "10"))
    
    scores = []
    print(f"Starting batch judging for {len(prompts)} samples with {max_workers} threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress
        results = list(tqdm(executor.map(call_judge_api, prompts), total=len(prompts), desc="[ViQuAD] Judging"))
        scores = results

    return sum(scores) / len(scores)
