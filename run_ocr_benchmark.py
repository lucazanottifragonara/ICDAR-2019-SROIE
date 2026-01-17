import ollama
import os
import json
import time

# Update these paths to where your SROIE data is
IMG_DIR = "./data/img"
KEY_DIR = "./data/key"
MODEL = "llama3.2-vision"

def run_sroie_benchmark(limit=10):
    client = ollama.Client(host='http://localhost:11434')
    images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')][:limit]
    
    overall_results = []
    
    print(f"--- SROIE Benchmark: Testing {len(images)} receipts on Ryzen 9 9900X ---")

    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMG_DIR, img_name)

        # Prefer .txt keys, but many datasets here use .json
        txt_path = os.path.join(KEY_DIR, f"{base_name}.txt")
        json_path = os.path.join(KEY_DIR, f"{base_name}.json")
        if os.path.exists(txt_path):
            key_path = txt_path
        elif os.path.exists(json_path):
            key_path = json_path
        else:
            print(f"Warning: ground truth not found for {base_name} (tried .txt and .json). Skipping.")
            continue

        # 1. Load the Ground Truth (The Correct Answer)
        with open(key_path, 'r', encoding='utf-8') as f:
            try:
                ground_truth = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: failed to parse ground truth for {base_name}. Skipping.")
                continue
            except Exception as e:
                print(f"Warning: error reading ground truth for {base_name}: {e}. Skipping.")
                continue

        # 2. Ask the Local AI
        print(f"Processing {img_name}...", end=" ", flush=True)
        start = time.time()
        
        response = client.chat(
            model=MODEL,
            messages=[{
                'role': 'user',
                'content': 'Extract "company", "date", and "total" from this receipt. Return ONLY JSON.',
                'images': [img_path]
            }],
            options={
                "num_thread": 12,     # Force use of 12 physical cores
                "temperature": 0      # Keep it deterministic for benchmarking
            }
        )
        
        elapsed = time.time() - start
        
        # 3. Parse AI Output
        try:
            # Clean AI output in case it wrapped JSON in markdown ```json blocks
            clean_output = response['message']['content'].strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean_output)
            # Some models return a list like [{...}] — pick the first dict if so
            if isinstance(parsed, dict):
                ai_data = parsed
            elif isinstance(parsed, list):
                ai_data = next((x for x in parsed if isinstance(x, dict)), {})
            else:
                ai_data = {}
        except Exception:
            ai_data = {}

        # 4. Simple Accuracy Check (Total Amount)
        ai_total = str(ai_data.get('total', 'MISSING')).replace("$", "").strip()
        gt_total = str(ground_truth.get('total', 'ERROR')).replace("$", "").strip()
        
        match = (ai_total == gt_total)
        print(f"{'✅ MATCH' if match else '❌ FAIL'} ({elapsed:.2f}s)")

        overall_results.append({
            "file": img_name,
            "match": match,
            "ai": ai_total,
            "truth": gt_total,
            "time": elapsed
        })

    # Summary
    matches = sum(1 for r in overall_results if r['match'])
    print(f"\n--- RESULTS ---")
    print(f"Accuracy: {(matches/len(overall_results))*100:.1f}%")
    print(f"Avg Speed: {sum(r['time'] for r in overall_results)/len(overall_results):.2f}s per page")

if __name__ == "__main__":
    run_sroie_benchmark(limit=5) # Start with 5 to test