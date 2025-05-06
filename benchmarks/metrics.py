import csv
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util


def load_csv(filename: str, column_name: str, return_all: bool = False) -> list:
    """
    Load CSV file, returning either all data or a specific column.

    :param filename: Path to the CSV file
    :param column_name: Name of the column to extract (if not return_all)
    :param return_all: Flag to return all data (default: False)
    :return: List of either all rows (dicts) or column values (str)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            if return_all:
                return [row for row in reader]
            else:
                if column_name not in reader.fieldnames:
                    raise ValueError(f"Column '{column_name}' not found in {filename}")
                return [row[column_name].strip() for row in reader]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for '{filename}'.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def evaluate_yes_no(_, expected: str, predicted: str) -> float:
    """
    Score Yes/No predictions: 1 if expected and predicted are both yes/no and match, 0 otherwise.
    """
    expected = expected.strip().lower()
    predicted = predicted.strip().lower()
    if expected in ["yes", "no"] and predicted in ["yes", "no"]:
        return 1.0 if expected == predicted else 0.0
    return 0.0


def evaluate(expected, predicted, is_yes_no=None, hq_data=None):
    results = {
        "Exact Match": 0,
        "ROUGE-1": [],
        "ROUGE-2": [],
        "ROUGE-L": [],
        "BLEU": [],
        "Cosine Similarity": [],
        "YesNoScore": []
    }

    smoothing = SmoothingFunction()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    for idx, (e, p) in enumerate(zip(expected, predicted)):
        # Exact Match
        results["Exact Match"] += 1 if e.lower() == p.lower() else 0

        # ROUGE Scores
        rouge_scores = scorer.score(p, e)
        results["ROUGE-1"].append(rouge_scores["rouge1"].fmeasure)
        results["ROUGE-2"].append(rouge_scores["rouge2"].fmeasure)
        results["ROUGE-L"].append(rouge_scores["rougeL"].fmeasure)

        # BLEU Score
        results["BLEU"].append(sentence_bleu([e.split()], p.split()),smoothing_function=smoothing.method4)

        # Cosine Similarity
        query_embedding = sim_model.encode(e, convert_to_tensor=True)
        answer_embedding = sim_model.encode(p, convert_to_tensor=True)
        similarity = util.cos_sim(query_embedding, answer_embedding).item()
        results["Cosine Similarity"].append(similarity)

        # Yes/No Scoring
        if is_yes_no and hq_data:
            expected_answer = hq_data[idx]["Expected Answer"]
            yes_no_score = evaluate_yes_no(None, expected_answer, p)
            results["YesNoScore"].append(yes_no_score)
        else:
            results["YesNoScore"].append(0.0)

    # Average Metrics
    for metric in ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "Cosine Similarity", "YesNoScore"]:
        results[metric] = np.mean(results[metric]) if results[metric] else 0.0

    results["Exact Match"] /= len(expected)

    return results


# === Run Evaluation ===

# Load RAG responses and expected answers
hq_data = load_csv('/gpfs/wolf2/olcf/trn040/scratch/luki/dse697_LU_NS/data/HQ_examples.csv', 'Expected Answer', return_all=True)
hq_expected = [row["Expected Answer"] for row in hq_data]
rag_responses = load_csv('/gpfs/wolf2/olcf/trn040/scratch/luki/dse697_LU_NS/benchmarks/RAG_responses.csv', 'Generated Answer')

# Tag yes/no questions
is_yes_no = [
    any(kw in answer["Expected Answer"].lower() for kw in ["yes", "no"]) 
    for answer in hq_data
]

evaluation_results = evaluate(hq_expected, rag_responses, is_yes_no, hq_data)


# Write to .csv
with open("RAG_evaluation_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    for metric, value in evaluation_results.items():
        writer.writerow([metric, f"{value:.4f}"])
