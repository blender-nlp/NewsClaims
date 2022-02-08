# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""Official evaluation script for the MRQA Workshop Shared Task.
Adapted fromt the SQuAD v1.1 official evaluation script.
Usage:
    python official_eval.py dataset_file.jsonl.gz prediction_file.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path
from urllib.parse import urlparse
import argparse
import string
import re
import json
import gzip
import sys
import os
from collections import Counter
from copy import deepcopy

def cached_path(url_or_filename, cache_dir = None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = os.path.dirname(url_or_filename)
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    url_or_filename = os.path.expanduser(url_or_filename)
    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_predictions(prediction_file):
    with open(prediction_file) as f:
        predictions = json.load(f)
    return predictions


# def read_answers(gold_file):
#     answers = {}
#     with gzip.open(gold_file, 'rb') as f:
#         for i, line in enumerate(f):
#             example = json.loads(line)
#             if i == 0 and 'header' in example:
#                 continue
#             for qa in example['qas']:
#                 answers[qa['qid']] = qa['answers']
#     return answers

def read_answers(gold_file):
    answers = {}
    # with gzip.open(gold_file, 'rb') as f:
    #     for i, line in enumerate(f):
    #         example = json.loads(line)
    #         if i == 0 and 'header' in example:
    #             continue
    #         for qa in example['qas']:
    #             answers[qa['qid']] = qa['answers']
    data = json.load(open(gold_file, "r"))
    for i, item in enumerate(data):
        answers[str(i)] = [item["answer"]]
    return answers

# def read_answers(gold_file):
#     answers = {}
#     # with gzip.open(gold_file, 'rb') as f:
#     #     for i, line in enumerate(f):
#     #         example = json.loads(line)
#     #         if i == 0 and 'header' in example:
#     #             continue
#     #         for qa in example['qas']:
#     #             answers[qa['qid']] = qa['answers']
#     data = json.load(open(gold_file, "r"))
#     for i, item in enumerate(data):
#         answers[str(i)] = [item["answer"]]
#     return answers

def evaluate_x_attribute(predictions, gold, skip_no_answer=False):
    f1 = exact_match = total = 0

    for key in predictions:
        unique_gold = dict()
        unique_pred = dict()
        for claim in gold[key]:
            claim_text = claim["original_claim_provenance"].strip()
            if claim_text != "":
                if claim_text not in unique_gold:
                    unique_gold[claim_text] = [claim["x_varible"]]
                else:
                    unique_gold[claim_text].append(claim["x_varible"])
        
        for claim in predictions[key]:
            claim_text = claim["claim"].strip()
            if claim_text != "":
                if claim_text not in unique_pred:
                    unique_pred[claim_text] = [claim["positive"]["text"]]
                else:
                    unique_pred[claim_text].append(claim["positive"]["text"])
        
        assert len(unique_pred) == len(unique_gold)
        for key in unique_pred:
            assert len(list(set(unique_pred[key]))) == 1
            prediction = list(set(unique_pred[key]))[0]
            ground_truths = list((set(unique_gold[key])))
            total += 1
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths)            
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print("Total: ", total )
    return {'exact_match': exact_match, 'f1': f1}
        

def evaluate_claimer_afffiliation(predictions, skip_no_answer=False):
    claimer_f1 = claimer_exact_match = total = 0
    affiliation_f1 = affiliation_exact_match = 0

    for key in predictions:
        unique_claims = dict()
        for claim in predictions[key]:
            claim_text = claim["original_claim_provenance"].strip()
            if claim_text != "":
                if claim_text not in unique_claims:
                    unique_claims[claim_text] = dict()
                    unique_claims[claim_text]["claimer_gold"] = [claim["claimer"]]
                    unique_claims[claim_text]["claimer_predicted"] = [claim["claimer_extracted"]]
                else:
                    unique_claims[claim_text]["claimer_gold"].append(claim["claimer"])
                    unique_claims[claim_text]["claimer_predicted"].append(claim["claimer_extracted"])

        for key in unique_claims:
            claimer_ground_truths = list(set(unique_claims[key]["claimer_gold"]))
            assert len(list(set(unique_claims[key]["claimer_predicted"]))) == 1
            claimer_prediction = unique_claims[key]["claimer_predicted"][0]

            total += 1
            claimer_exact_match += metric_max_over_ground_truths(
            exact_match_score, claimer_prediction, claimer_ground_truths)            
            claimer_f1 += metric_max_over_ground_truths(
            f1_score, claimer_prediction, claimer_ground_truths)
                

    # for key in predictions:
    #     for claim in predictions[key]:
    #         claimer_ground_truths = [claim["claimer"]]
    #         claimer_prediction = claim["claimer_extracted"]

    #         affiliation_ground_truths = [claim["affiliation"]]
    #         affiliation_prediction = claim["affiliation_extracted"]

    #         total += 1
    #         claimer_exact_match += metric_max_over_ground_truths(
    #         exact_match_score, claimer_prediction, claimer_ground_truths)            
    #         claimer_f1 += metric_max_over_ground_truths(
    #         f1_score, claimer_prediction, claimer_ground_truths)

    #         affiliation_exact_match += metric_max_over_ground_truths(
    #         exact_match_score, affiliation_prediction, affiliation_ground_truths)            
    #         affiliation_f1 += metric_max_over_ground_truths(
    #         f1_score, affiliation_prediction, affiliation_ground_truths)

    claimer_exact_match = 100.0 * claimer_exact_match / total
    claimer_f1 = 100.0 * claimer_f1 / total

    affiliation_exact_match = 100.0 * affiliation_exact_match / total
    affiliation_f1 = 100.0 * affiliation_f1 / total

    return {'claimer_exact_match': claimer_exact_match, 'claimer_f1': claimer_f1, 'affiliation_exact_match': affiliation_exact_match, 'affiliation_f1': affiliation_f1}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation for MRQA Workshop Shared Task')
    parser.add_argument('dataset_file', type=str, help='Dataset File')
    parser.add_argument('predictions_file', type=str, help='Predictions File')
    parser.add_argument('--skip-no-answer', action='store_true')
    args = parser.parse_args()

    # answers = read_answers(cached_path(args.dataset_file))
    # predictions = read_predictions(cached_path(args.prediction_file))

    # predictions = json.load(open(args.predictions_file))
    # metrics = evaluate_claimer_afffiliation(predictions, args.skip_no_answer)

    predictions = json.load(open(args.predictions_file))
    gold = json.load(open(args.dataset_file))
    metrics = evaluate_x_attribute(predictions, gold, args.skip_no_answer)

    print(json.dumps(metrics))
