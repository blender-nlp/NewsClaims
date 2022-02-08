import json
import argparse
from sklearn.metrics import classification_report
from eval_F1 import exact_match_score, f1_score, metric_max_over_ground_truths
import numpy as np

def compute_span_f1(pred_span, gold_span):
    gold_start, gold_end = gold_span
    pred_start, pred_end = pred_span
    tp, fp, fn = 0, 0, 0
    if pred_end >= gold_end:
        tp = gold_end - max(pred_start, gold_start) + 1
    else:
        tp = pred_end - max(pred_start, gold_start) + 1
    tp = max(tp, 0)
    fp = pred_end-pred_start+1-tp
    fn = gold_end-gold_start+1-tp
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    if precision == 0.0 and recall == 0.0:
        return 0.0, precision, recall
    F1 = 2*precision*recall/(precision+recall)
    return F1, precision, recall

def score_claim(args):
    print("Evalutating Claim Detection Task")
    gold = json.load(open(args.gold_file))
    predictions = json.load(open(args.predictions_file))

    tp = 0.0
    total_prec = 0.0
    total_rec = 0.0

    for doc_id in gold:
        gold_sents = set(list(gold[doc_id].keys()))
        if doc_id in predictions:
            pred_sents = set(predictions[doc_id])
            tp += len(gold_sents.intersection(pred_sents))
            total_prec += len(pred_sents)
        
        total_rec += len(gold_sents)
    
    if total_prec != 0:
        prec = float(tp)/total_prec
    else:
        prec = 0
    rec = float(tp)/total_rec

    print("Prec: ", prec, tp, total_prec)
    print("Rec: ", rec, tp, total_rec)
    if tp != 0:
        print("F1: ", 2*prec*rec/(prec+rec))
    else:
        print("F1: ", 0.0)

def score_topic(args):
    print("Evalutating Topic Classification Task")
    gold = json.load(open(args.gold_file))
    predictions = json.load(open(args.predictions_file))

    gold_topics = list()
    predicted_topics = list()

    for doc_id in gold:
        assert doc_id in predictions
        for segment_id in gold[doc_id]:
            assert segment_id in predictions[doc_id]
            gold_topics.append(gold[doc_id][segment_id]["topic"])
            predicted_topics.append(predictions[doc_id][segment_id]["topic"])

    print(classification_report(gold_topics, predicted_topics, digits=4)) 

def score_claim_object(args):
    print("Evalutating Claim Object Extraction Task")
    gold = json.load(open(args.gold_file))
    predictions = json.load(open(args.predictions_file))
    EM = 0.0
    F1 = 0.0
    count = 0

    for doc_id in gold:
        for segment_id in gold[doc_id]:
            if "claim_object" in gold[doc_id][segment_id]:
                assert doc_id in predictions
                assert segment_id in predictions[doc_id]

                EM += metric_max_over_ground_truths(exact_match_score, predictions[doc_id][segment_id]["claim_object"], [gold[doc_id][segment_id]["claim_object"]])
                F1 += metric_max_over_ground_truths(f1_score, predictions[doc_id][segment_id]["claim_object"], [gold[doc_id][segment_id]["claim_object"]])
                count += 1
    
    print("EM: ", EM/count,  "F1: ", F1/count, "Count: ", count)

def score_stance(args):
    print("Evalutating Stance Detection Task")
    gold = json.load(open(args.gold_file))
    predictions = json.load(open(args.predictions_file))
    gold_stance = list()
    predicted_stance = list()

    for doc_id in gold:
        for segment_id in gold[doc_id]:
            if "stance" in gold[doc_id][segment_id]: # and "claim_object" in gold[doc_id][segment_id]:
                assert doc_id in predictions
                assert segment_id in predictions[doc_id]
                
                gold_stance.append(gold[doc_id][segment_id]["stance"])
                predicted_stance.append(predictions[doc_id][segment_id]["stance"])
    
    print(classification_report(gold_stance, predicted_stance, digits=4)) 

def score_claim_span(args):
    print("Evalutating Claim Span Detection Task")
    gold = json.load(open(args.gold_file))
    predictions = json.load(open(args.predictions_file))

    prec = list()
    recall = list()

    for doc_id in gold:
        for segment_id in gold[doc_id]:
            if "claim_span" in gold[doc_id][segment_id]:
                _, ex_prec, ex_recall = compute_span_f1(predictions[doc_id][segment_id]["claim_span"], gold[doc_id][segment_id]["claim_span"])
                prec.append(ex_prec)
                recall.append(ex_recall)

    final_prec = np.mean(prec)
    final_recall = np.mean(recall)
    print("P: ", final_prec, "R: ", final_recall, "F1: ", (2*final_prec*final_recall)/(final_prec+final_recall), "Count: ", len(prec))

def score_claimer(args):
    print("Evaluating Claimer Extraction Task")
    gold = json.load(open(args.gold_file))
    predictions = json.load(open(args.predictions_file))

    in_sent_f1 = list()
    out_of_sent_f1 = list()

    ans_f1 = list()
    no_claimer_tp = 0.0
    no_claimer_total_prec = 0.0
    no_claimer_total_rec = 0.0

    for doc_id in gold:
        for segment_id in gold[doc_id]:
            if "has_claimer" in gold[doc_id][segment_id]:
                assert doc_id in predictions
                assert segment_id in predictions[doc_id]
                
                if predictions[doc_id][segment_id]["has_claimer"] == False:
                    no_claimer_total_prec += 1.0
                    pred_claimer = "<AUTHOR>"
                else:
                    pred_claimer = predictions[doc_id][segment_id]["claimer"]
                if gold[doc_id][segment_id]["has_claimer"]:
                    if gold[doc_id][segment_id]["claimer_in_sentence"]:
                        in_sent_f1.append(metric_max_over_ground_truths(f1_score, pred_claimer, [gold[doc_id][segment_id]["claimer"]]))
                    else:
                        out_of_sent_f1.append(metric_max_over_ground_truths(f1_score, pred_claimer, [gold[doc_id][segment_id]["claimer"]]))
                    ans_f1.append(metric_max_over_ground_truths(f1_score, pred_claimer, [gold[doc_id][segment_id]["claimer"]]))
                else:
                    if predictions[doc_id][segment_id]["has_claimer"] == False:
                        no_claimer_tp += 1.0
                    no_claimer_total_rec += 1.0
    
    print("Ans F1: ", np.mean(ans_f1), " Count: ", len(ans_f1))
    prec = no_claimer_tp/no_claimer_total_prec
    recall = no_claimer_tp/no_claimer_total_rec
    no_ans_f1 = 2*prec*recall/(prec + recall)
    print("No ANS P:", prec, "No ANS R:", recall, "No ANS F1:", no_ans_f1, "Count: ", no_claimer_total_rec)
    print("Final F1: ", (len(ans_f1)*np.mean(ans_f1) + no_claimer_total_rec*no_ans_f1)/(len(ans_f1) + no_claimer_total_rec))
    print("In Sent F1: ", np.mean(in_sent_f1), "Count: ", len(in_sent_f1))
    print("Out of Sent F1: ", np.mean(out_of_sent_f1), "Count: ", len(out_of_sent_f1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--gold_file', type=str, help="Path to gold data")
    parser.add_argument('--predictions_file', type=str, help="path to predictions")
    parser.add_argument("--eval_claim", action='store_true', help="Whether to evaluate claim detection performance")
    parser.add_argument("--eval_topic", action='store_true', help="Whether to evaluate topic detection performance")
    parser.add_argument("--eval_claim_object", action='store_true', help="Whether to evaluate topic detection performance")
    parser.add_argument("--eval_claim_span", action='store_true', help="Whether to evaluate topic detection performance")
    parser.add_argument("--eval_stance", action='store_true', help="Whether to evaluate topic detection performance")
    parser.add_argument("--eval_claimer", action='store_true', help="Whether to evaluate topic detection performance")
    args = parser.parse_args()
    if args.eval_claim:
        score_claim(args)
    
    if args.eval_topic:
        score_topic(args)

    if args.eval_claim_object:
        score_claim_object(args)
    
    if args.eval_claim_span:
        score_claim_span(args)
    
    if args.eval_stance:
        score_stance(args)
    
    if args.eval_claimer:
        score_claimer(args)