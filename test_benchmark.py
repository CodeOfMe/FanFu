#!/usr/bin/env python3
"""
SOTA Benchmark for FanFu Converted Models - Fast Version.

Tests both models with focused benchmark questions via Ollama.
"""

import json
import re
import subprocess
import time
from pathlib import Path


BENCHMARK_QUESTIONS = {
    "MMLU Knowledge": [
        {"q": "What is the capital of France? A. London B. Paris C. Berlin D. Madrid. Return only the letter.", "a": "B"},
        {"q": "Which planet is known as the Red Planet? A. Venus B. Mars C. Jupiter D. Saturn. Return only the letter.", "a": "B"},
        {"q": "What is the chemical symbol for gold? A. Go B. Gd C. Au D. Ag. Return only the letter.", "a": "C"},
        {"q": "In Python, which keyword defines a function? A. func B. def C. function D. define. Return only the letter.", "a": "B"},
        {"q": "What is the square root of 144? A. 10 B. 11 C. 12 D. 13. Return only the letter.", "a": "C"},
        {"q": "Which data structure uses FIFO? A. Stack B. Queue C. Tree D. Graph. Return only the letter.", "a": "B"},
        {"q": "Who wrote Romeo and Juliet? A. Dickens B. Shakespeare C. Austen D. Twain. Return only the letter.", "a": "B"},
        {"q": "What is the binary of decimal 10? A. 1010 B. 1100 C. 1001 D. 1110. Return only the letter.", "a": "A"},
    ],
    "Math Reasoning": [
        {"q": "What is 15% of 200? Give the number only.", "kw": ["30"]},
        {"q": "A train travels 60 km/h for 2.5 hours. How far? Give the number only.", "kw": ["150"]},
        {"q": "What is 2 to the power of 10? Give the number only.", "kw": ["1024"]},
    ],
    "Code Generation": [
        {"q": "Write a Python function called fibonacci that returns the nth Fibonacci number. Only the function.", "kw": ["def", "fibonacci"]},
        {"q": "Write a Python one-liner to reverse a list called my_list.", "kw": ["[::-1]", "reverse"]},
    ],
    "Chinese Understanding": [
        {"q": "用一句话介绍北京", "kw": ["北京", "首都"]},
        {"q": "中国的四大发明是什么？", "kw": ["造纸", "火药", "指南针"]},
    ],
    "Logic Reasoning": [
        {"q": "A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball?", "kw": ["0.05", "5 cent"]},
        {"q": "5 machines make 5 widgets in 5 minutes. How long for 100 machines to make 100 widgets?", "kw": ["5"]},
    ],
}


def run_ollama(model, prompt, timeout=45):
    """Run query via Ollama."""
    try:
        r = subprocess.run(
            ["ollama", "run", model, "--nowordwrap"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def extract_answer(text):
    """Extract single letter answer."""
    if not text:
        return ""
    m = re.search(r'[Aa]nswer[:\s]*([A-D])', text)
    if m:
        return m.group(1)
    for line in reversed(text.split('\n')):
        m = re.search(r'\b([A-D])\b', line.strip())
        if m:
            return m.group(1)
    m = re.search(r'\b([A-D])\b', text)
    return m.group(1) if m else ""


def check_keywords(text, kws):
    """Check if any keyword present."""
    if not text:
        return False
    t = text.lower()
    return any(k.lower() in t for k in kws)


def run_suite(model_name, suite_name, questions):
    """Run a benchmark suite."""
    print(f"\n  {suite_name}")
    correct = 0
    details = []
    start = time.time()

    for i, q in enumerate(questions):
        response = run_ollama(model_name, q["q"])

        if "a" in q:
            predicted = extract_answer(response)
            is_correct = predicted == q["a"]
        else:
            is_correct = check_keywords(response, q["kw"])
            predicted = "✅" if is_correct else "❌"

        if is_correct:
            correct += 1
        status = "✅" if is_correct else ("⏱️" if not response else "❌")
        print(f"    Q{i+1}: {status}")

        details.append({
            "question": q["q"][:60],
            "correct": is_correct,
            "response_preview": response[:80] if response else "",
        })

    elapsed = time.time() - start
    return {
        "correct": correct, "total": len(questions),
        "accuracy": round(correct / len(questions) * 100, 1),
        "time": round(elapsed, 1), "details": details,
    }


def benchmark_model(model_name, ollama_name):
    """Run full benchmark on a model."""
    print(f"\n{'#'*60}")
    print(f"# {model_name} ({ollama_name})")
    print(f"{'#'*60}")

    results = {"model": model_name, "ollama_name": ollama_name, "suites": {}}
    total_correct = 0
    total_questions = 0
    total_time = 0

    for suite_name, questions in BENCHMARK_QUESTIONS.items():
        suite_result = run_suite(ollama_name, suite_name, questions)
        results["suites"][suite_name] = suite_result
        total_correct += suite_result["correct"]
        total_questions += suite_result["total"]
        total_time += suite_result["time"]

    results["overall"] = {
        "correct": total_correct,
        "total": total_questions,
        "accuracy": round(total_correct / total_questions * 100, 1) if total_questions > 0 else 0,
        "time": round(total_time, 1),
    }

    # Print summary
    print(f"\n  {'Suite':<25} {'Score':<10} {'Accuracy':<10} {'Time':<8}")
    print(f"  {'-'*53}")
    for name, s in results["suites"].items():
        print(f"  {name:<25} {s['correct']}/{s['total']:<8} {s['accuracy']}%{'':<5} {s['time']}s")
    print(f"  {'-'*53}")
    o = results["overall"]
    print(f"  {'OVERALL':<25} {o['correct']}/{o['total']:<8} {o['accuracy']}%{'':<5} {o['time']}s")

    return results


def main():
    models = {
        "qwen3.5": "huihui_ai/qwen3.5-abliterated:0.8B",
        "lfm2.5": "huihui_ai/lfm2.5-abliterated:latest",
    }

    all_results = {}
    for name, ollama_name in models.items():
        result = benchmark_model(name, ollama_name)
        all_results[name] = result

    # Save
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
