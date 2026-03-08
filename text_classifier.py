import ollama
from concurrent.futures import ThreadPoolExecutor

def classify_fracture(text, model_name="llama3.2:3b"):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': (
                    'Read this report from a medical professional and determine if it mentions the patient has a fracture.'
                    'To indicate a fracture, response with "1". Otherwise, respond with "0". Provide no other output.'
                    'There are multiple reports, but all doctors looked at the same patient and photos when making their decisions, so you should only give a single "1" or "0" as to whether or not the patient mentioned in the report has a fracture.'
                    f'List of reports: {text}'
                )
            }]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error classifying fracture: {e}")
        return None

def classify_avg(text, repeats=10, model_name="qwen2.5:0.5b"):
    results = []
    for _ in range(repeats):
        result = classify_fracture(text, model_name=model_name)
        if result is not None:
            try:
                results.append(int(result))
            except ValueError:
                # unexpected text – ignore this run
                print(f"[WARN] Unexpected response:|{result}|")
    return sum(results) / len(results)

def classify_fracture_avg_parallel(text, repeats=50, model_name="llama3.2:3b", workers=50):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(classify_fracture, text, model_name) for _ in range(repeats)]
        results = [f.result() for f in futures]

    ints = [int(r) for r in results if r is not None]
    return sum(ints) / len(ints) if ints else 0.0