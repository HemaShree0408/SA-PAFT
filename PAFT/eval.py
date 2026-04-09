import sys, json, torch, pandas as pd, numpy as np, gc, os
# Determine project root relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
paft_path = current_dir 

sys.path.insert(0, os.path.join(paft_path, 'src'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from llamafactory.data.loader import parse_example_content

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')

val_json_path = os.path.join(paft_path, 'data/hellaswag_val.json')
with open(val_json_path) as f:
    val_data = json.load(f)[:200]

prompts_xlsx_path = os.path.join(paft_path, 'prompts/robust_traindataset_hel.xlsx')
df = pd.read_excel(prompts_xlsx_path)
templates = df['text'].tolist()[:10]

def evaluate(ckpt_path, label):
    print(f"\n--- Evaluating {label} ---")
    base = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', torch_dtype=torch.float16, device_map='auto')
    model = PeftModel.from_pretrained(base, ckpt_path)
    model.eval()

    all_results = []
    for ex in val_data:
        parsed = parse_example_content(ex['instruction'])
        if parsed is None:
            continue
        correct = ex['output'].strip()
        per_prompt = []
        for t in templates:
            try:
                filled = t.format(**parsed)
                inputs = tokenizer(filled, return_tensors='pt',
                                   truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1, :]
                scores = {c: logits[tokenizer.encode(c, add_special_tokens=False)[0]].item()
                          for c in ['A','B','C','D']}
                pred = max(scores, key=scores.get)
                per_prompt.append(int(pred == correct))
            except:
                continue
        if per_prompt:
            all_results.append(per_prompt)

    # Convert to numpy for advanced indexing (Shape: [NumEx, NumPrompts])
    results_matrix = np.array(all_results)
    
    # Paper Metric 1: Mean Accuracy per prompt
    # First avg over examples for each prompt, then avg those.
    prompt_accuracies = np.mean(results_matrix, axis=0)
    
    # Paper Metric 2: Standard Deviation (lower is better)
    std_dev = np.std(prompt_accuracies)
    
    # Paper Metric 3: Minimum Accuracy (Worst-case)
    min_acc = np.min(prompt_accuracies)
    
    # Paper Metric 4: Top-90% (Stability)
    top_90 = np.mean(prompt_accuracies >= 0.90) * 100

    result = {
        'mean_acc': float(np.mean(prompt_accuracies)),
        'std_dev': float(std_dev),
        'min_acc': float(min_acc),
        'top_90_rate': float(top_90)
    }
    
    print(f"  Mean Acc:   {result['mean_acc']:.4f}")
    print(f"  Std Dev:    {result['std_dev']:.4f} (Robustness)")
    print(f"  Min Acc:    {result['min_acc']:.4f} (Worst-case)")
    print(f"  Top-90%:    {result['top_90_rate']:.1f}%")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return result

results = {}
results['baseline'] = evaluate(f'{paft_path}/output/paft_baseline', 'Standard SFT Baseline')
results['sa_paft']  = evaluate(f'{paft_path}/output/sapaft', 'SA-PAFT (Ours)')

print("\n" + "="*50)
print(f"{'Metric':<15} | {'Baseline':<10} | {'SA-PAFT':<10} | {'Diff'}")
print("-" * 50)
for m in ['mean_acc', 'std_dev', 'min_acc', 'top_90_rate']:
    b = results['baseline'][m]
    s = results['sa_paft'][m]
    diff = s - b
    # For Std, lower is better, so a negative diff is an improvement
    prefix = "+" if diff > 0 else ""
    print(f"{m:<15} | {b:<10.4f} | {s:<10.4f} | {prefix}{diff:.4f}")
print("="*50)

with open(f'{paft_path}/output/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to eval_results.json")
