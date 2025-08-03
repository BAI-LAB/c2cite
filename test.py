import logging
import os
import re
import json
import copy
import string
from nltk import sent_tokenize
from tqdm import tqdm
import numpy as np
from rouge import Rouge
import collections
from rouge_score import rouge_scorer, scoring
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import datasets as hf_datasets
import evaluate as hf_evaluate
import torch
import argparse


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com '

global autoais_model, autoais_tokenizer
autoais_model = None
autoais_tokenizer = None
qa_pipeline = None
get_docs_by_index = lambda i,docs: docs[i] if i < len(docs) else None 
ais_LLM = None

evaluate_device = 'cuda:5'
#QA_MODEL = "gaotianyu1350/roberta-large-squad"
QA_MODEL = "/yy21/qa_model"
#AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"
AUTOAIS_MODEL = "/yy21/autoais"

class BasicMetric:
    def __init__(self) -> None:
        pass

    def add_batch(self, data):
        pass

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        pass

    def compute(self) -> Dict[str, Any]:
        pass


from statistics import harmonic_mean

def normalize_answers(text):
  """QA style answer normalization. Similar to TriviaQA."""

  def remove_articles(s):
    return re.sub(r"\b(a|an|the)\b", " ", s)

  def replace_punctuation(s):
    to_replace = set(string.punctuation)
    return "".join(" " if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return " ".join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)

  return text


def strip_attribution_tokens(text):
  """Strip the attribution tokens from an answer."""
  return re.sub(r'\[ ([1-9]) ([^\[\]]*) \]',r'\2' , text)


def non_quoted(text):
  """Returns only the text that is outside of quoted spans."""
  return re.sub(r'\[ ([1-9]) ([^\[\]]*) \]', '' , text)


def only_quoted(text, sources='1-9', sep = ' '):
    """Returns only the text that is within of quoted spans."""
    return sep.join([x.group(1) for x in re.finditer(r'\[ [{}] ([^\[\]]*) \]'.format(sources), text)])


def quoted_sources(text):
  """Returns the list of input sources that were quoted in the answer."""
  return sorted(list(set([int(x.group(1)) for x in re.finditer(r'\[ ([1-9]) [^\[\]]* \]', text)])))


def score_all(data, scorer, aggr_measure, score_keys, preprocess_func=None, bootstrap=False):
  """
  Aggregates across all targets per sample.

  all_targets: list of list of strings
  all_predictions: list of strings
  """
  all_targets = [d['answer'] for d in data]
  all_predictions = [d['output'] for d in data]

  np.random.seed(1337)

  is_rouge_measure = 'rouge' in aggr_measure

  if preprocess_func is not None:
    scoring_func = lambda target, prediction: scorer.score(target=preprocess_func(target), prediction=preprocess_func(prediction))
  else:
    scoring_func = scorer.score

  aggregator = scoring.BootstrapAggregator()
  all_scores = [] if is_rouge_measure else dict((k,[]) for k in score_keys)
  for targets, prediction in zip(all_targets, all_predictions):
    # Max across references by aggr_measure
    if is_rouge_measure:
      max_scores = max([scoring_func(target, prediction) for target in targets], key=lambda x: x[aggr_measure].fmeasure)

      aggregator.add_scores(max_scores)
      all_scores.append(max_scores[aggr_measure].fmeasure*100)
    else:
      if aggr_measure == 'independent':
        max_scores = {}
        for key in score_keys:
          max_scores[key] = max([scoring_func(target, prediction)[key] for target in targets])
      else:
        max_scores = max([scoring_func(target, prediction) for target in targets], key=lambda x: x[aggr_measure])

      aggregator.add_scores(max_scores)
      for key in score_keys:
        all_scores[key].append(max_scores[key]*100)

  if not bootstrap:
    return all_scores

  result = aggregator.aggregate()
  postprocess_result = (lambda x: x.fmeasure*100) if is_rouge_measure else (lambda x: x*100)
  bootstrap_results = {}
  for key in score_keys:
    bootstrap_results[key] = (postprocess_result(result[key].mid), postprocess_result(result[key].low), postprocess_result(result[key].high))
  return bootstrap_results, all_scores

## ROUGE ##

score_all_rouge = functools.partial(score_all, scorer=rouge_scorer.RougeScorer(rouge_types=("rouge1", "rouge2", "rougeLsum", "rougeL")), aggr_measure='rougeLsum',  score_keys=("rouge1", "rouge2", "rougeLsum"), preprocess_func=strip_attribution_tokens, bootstrap=True)

## F1 ##

class _f1_scorer:
  def score(self, target, prediction):
    """Computes token F1 score for a single target and prediction."""
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = (collections.Counter(prediction_tokens) &
              collections.Counter(target_tokens))
    num_same = sum(common.values())
    if len(target_tokens) == 0 and len(prediction_tokens) == 0:
      return {'F1': 1.0, 'recall': 1.0, 'precision': 1.0}
    elif len(target_tokens) == 0 and len(prediction_tokens) > 0:
      return {'F1': 0.0, 'recall': 1.0, 'precision': 0.0}
    elif len(target_tokens) > 0 and len(prediction_tokens) == 0:
      return {'F1': 0.0, 'recall': 0.0, 'precision': 1.0}
    elif num_same == 0:
      return {'F1': 0.0, 'recall': 0.0, 'precision': 0.0}
    else:
      precision = 1.0 * num_same / len(prediction_tokens)
      recall = 1.0 * num_same / len(target_tokens)
      f1 = (2 * precision * recall) / (precision + recall)
      return {'F1': f1, 'recall': recall, 'precision': precision}


score_all_f1 = functools.partial(score_all, scorer=_f1_scorer(), aggr_measure='F1', score_keys=("F1", "recall", "precision"))


def preprocess_quotes_f1(text, sep=' ', sources='1-7'):
  text = only_quoted(text, sep=sep, sources=sources)
  return normalize_answers(text)


score_all_recall = functools.partial(score_all, scorer=_f1_scorer(), aggr_measure='recall', score_keys=("recall",))



def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True
    return False


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def load_auto_ais():
    global autoais_model, autoais_tokenizer
    print('Initializing eval model for citation precision and recall...') 
    autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, device_map=evaluate_device, )
    autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
    print('Done!')

def _run_nli_autoais(passage, claim, test = False):
    """
    Run inference for assessing AIS between a premise and .hypothesis
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py 
    """
    if not test:
        global autoais_model, autoais_tokenizer
        if not autoais_model:
            load_auto_ais()
        input_text = "premise: {} hypothesis: {}".format(passage, claim)
        input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
        with torch.inference_mode():
            outputs = autoais_model.generate(input_ids, max_new_tokens=10)
        result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference = 1 if result == "1" else 0
        return inference
    else:
        res = 114514

    return res


def compute_autoais(data,
                    qampari=False,
                    at_most_sents = 50,
                    at_most_citations=3,
                    entail_function = _run_nli_autoais):
    """
Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer


    ais_scores = []
    ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    for item in tqdm(data):
        # Get sentences by using NLTK
        if qampari:
            #print('now qampari...')
            sents = [item['query'] + " " + x.strip() for x in
                     item['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            sents = sent_tokenize(item['output'])[:at_most_sents]
        if len(sents) == 0:
            ais_scores.append(0.0)
            ais_scores_prec.append(0.0)  # len(sents))
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id]  # Citation removed and (if opted for) decontextualized
            joint_entail = -1  # Undecided

            # Find references
            #ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]  # In text citation id starts from 1
            matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", sent)
            ref = [int(num)-1 for match in matches for num in match.replace(' ', '').split(',')]
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
total_citations += len(ref)
                joint_passage = '\n'.join([(item['docs'][psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail = entail_function(joint_passage, target_sent)
                autoais_log.append({
                    #"question": item['question'],
                    "output": item['output'],
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                })

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = item['docs'][psgs_id]
                    nli_result = entail_function(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([item['docs'][pid] for pid in subset_exclude])
                        nli_result =entail_function(passage, target_sent)
                        if nli_result:  # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail
        sent_total += len(sents)
        ais_scores.append(entail / len(sents))
        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0)  # len(sents))
    if sent_mcite > 0 and sent_mcite_support > 0:
        print(
            "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
                100 * sent_mcite / sent_total,
                100 * sent_mcite_support / sent_mcite,
                100 * sent_mcite_overcite / sent_mcite_support
            ))

    return ({
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_prec": 100 * np.mean(ais_scores_prec),
    }, ais_scores, ais_scores_prec)


def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_qa(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """
    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        logging.warn("Warning: no QA pairs found in data")
        return {
            'QA-EM': 0,
            'QA-F1': 0,
            'QA-Hit': 0,
        }

    # Load model
    #logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    global qa_pipeline
    if not qa_pipeline:
        qa_pipeline = transformers.pipeline("question-answering", model=QA_MODEL, device = evaluate_device)
#logger.info("Done")

    # Get prediction
    #logger.info("Computing the QA-based accuracy...")
    em, f1, bins = [], [], []
    for item in tqdm(data):
        question = [qa_pair['question'] for qa_pair in item['qa_pairs']]
        #question = [item['qa_pairs'][0]['question']]
        context = item['output'] if len(item['output']) > 0 else " "
        results = qa_pipeline(question=question, context=remove_citations(context), handle_impossible_answer=True)
        loc_counter, loc_em, loc_f1 = 0, 0, 0

        for idx, res in enumerate(results):
            answers = item["qa_pairs"][idx]["short_answers"]
            prediction = res["answer"]

            loc_em += max([compute_exact(a, prediction) for a in answers])
            loc_f1 += max([compute_f1(a, prediction) for a in answers])
            loc_counter += 1

        em.append(loc_em / loc_counter)
        f1.append(loc_f1 / loc_counter)
        bins.append(loc_em == loc_counter)

    return {
        'QA-EM': 100 * np.mean(em),
'QA-F1': 100 * np.mean(f1),
        'QA-Hit': 100 * np.mean(bins)
    }


def compute_claims(data):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        #logger.info("Loading AutoAIS model...")
        # autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16,
                                                              device_map=evaluate_device)
        # autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto",offload_folder= "/data/hongbang/zsf/projects/ALCE/ALCE/model/t5_xxl_true_nli_mixture/offload1")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
    #logger.info("Computing claims...")
    scores = []
    for item in tqdm(data):
normalized_output = remove_citations(item['output'])
        entail = 0
        claims = item["qa_pairs"]
        for claim in claims:
            entail += _run_nli_autoais(normalized_output, claim)
        scores.append(entail / len(claims))
    return 100 * np.mean(scores), scores


def compute_qampari_f1(data, cot=False):
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for item in data:
        if cot:
            if ":" in item['output']:
                o = ':'.join(item['output'].split(":")[1:])  # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item['output']

        preds = [normalize_answer(x.strip()) for x in remove_citations(o).rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0]  # delete empty answers
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item['answer']]
flat_answers = [item for sublist in answers for item in sublist]
        prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)

        rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
        rec_top5.append(min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(answers)))
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0)
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec": 100 * np.mean(prec),
        "qampari_rec": 100 * np.mean(rec),
        "qampari_rec_top5": 100 * np.mean(rec_top5),
        "qampari_f1": 100 * np.mean(f1),
        "qampari_f1_top5": 100 * np.mean(f1_top5),
    }


def compute_str_em(data):
"""Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """
    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0

    acc = []
    for item in data:
        loc_acc = []
        if len(item['qa_pairs']) == 0:
            continue
        loc_acc.append(exact_presence(item['qa_pairs'][0]['short_answers'], item["output"]))
        """for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))"""
        acc.append(float(np.mean(loc_acc)))
    return (100 * np.mean(acc) if len(acc) > 0 else 0, acc)


def compute_mauve(data):
    """Compute Mauve score."""

    logging.info("Computing MAUVE...")
    human_data = []
    model_data = []
    for item in data:
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words
        human_data.append(
            ' '.join((item['query'] + " " + item['answer'].strip()).split()[:100]).rstrip(string.punctuation))
model_data.append(
            ' '.join((item['query'] + " " + item['output'].strip()).split()[:100]).rstrip(string.punctuation))

    import mauve
    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name="gpt2-large"
    )
    return out.mauve * 100


def compute_rouge_l(data):
    total = len(data)
    res = {
                "r": 0.0,
                "p": 0.0,
                "f": 0.0
            }
    for item in data:
        # print(f"output:{item['output']}, \nanswer:{item['answer']}")
        if item['output'] and item['answer']:
            rouge = Rouge()
            scores = rouge.get_scores(item['output'], item['answer'])
            res['r'] += scores[0]['rouge-l']['r']
            res['p'] += scores[0]['rouge-l']['p']
            res['f'] += scores[0]['rouge-l']['f']
        else:
            print('Warning: no hypothesis or references')
res['r'] /= total
    res['p'] /= total
    res['f'] /= total

    return res


def compute_length(data):
    return sum(len(item['output'].split(' '))for item in data)/(len(data))


metric_list = {
    'cite_pr': compute_autoais,
    'asqa_acc': compute_qa,
    'eli5_acc': compute_claims,
    'qampari': compute_qampari_f1,
    'short_ans': compute_str_em,
    # 'fluence': compute_mauve,
    'rouge': compute_rouge_l,
    'length': compute_length,
    'rouge_all': score_all_rouge,
    'semqa_f1': score_semqa_f1, # 相当于precision
    'semqa_short': score_semqa_short_recall, # 相当于recall
}

data_list = {
    'cite_pr': {'output': None, 'docs': None, 'query': None},
    'asqa_acc': {'output': None,'qa_pairs': None, 'query': None},
    'eli5_acc': {'output': None, 'qa_pairs': None},
    'qampari': {'output': None, 'answer': None},
    'short_ans': {'qa_pairs': None, 'output': None},
    # 'fluence': {'query': None, 'answer': None, 'output': None},
    'rouge': {'output': None, 'answer': None},
'length': {'output': None},
    'rouge_all': {'answer': None, 'output': None},
    'semqa_f1': {'answer': None, 'output': None, 'docs': None},
    'semqa_short':{'output': None, 'qa_pairs': None},
    'semqa': {}
}



class AttributeMetric:
    def __init__(self, config):
        self.task = 'attribute'
        self.metrics = config['metric']
        self.flag = False
        self.data = {
            'cite_pr': [],
            'asqa_acc': [],
            'eli5_acc': [],
            'qampari': [],
            'short_ans': [],
            'fluence': [],
            'rouge': [],
            'length': [],
            'rouge_all': [],
            'semqa_f1': [],
            'semqa_short': [],
            'semqa': [],
        }
    
    def add_batch(self, data): #(output, qa_pairs, answer, docs, query)
        for metric in self.metrics:
            self.data[metric].append({k:v for k, v in data.items() if k in data_list[metric]})
        
    def compute(self):
        ans = {}
        for metric in self.metrics:
assert metric in metric_list, logging.info("Invalid metric")
            if metric == 'cite_pr' and 'qampari' in self.metrics:
                ans[metric] = metric_list[metric](data = self.data[metric], qampari = True)
            else:
                ans[metric] = metric_list[metric](data = self.data[metric])

        return ans

"""def main():

    def split_docs_and_answer(input_str):

        tag = "</think>"
        index = input_str.find(tag)
        if index != -1:
            return input_str[index + len(tag):]
        else:
            return "" 
        

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()

    test_data = []
    with open(args.dir, "r", encoding="utf-8") as fuck:
        with open(f'/yy21/MoE-PEFT/dataset/front_output/{args.dataset}.json', "r", encoding="utf-8") as f:
data = json.load(f)
            for idx, line in enumerate(fuck):
                opt = json.loads(line)
                
                ori_output = re.sub(r'\[ref_(\d+)\]', r'[\1]', opt['response'])
                if args.dataset == 'asqa':
                   qa_pairs = data[idx]['qa_pairs']
                elif args.dataset == 'eli5':
                    qa_pairs = data[idx]['claims']
                answer = data[idx]['answer']
                query = data[idx]['question']

                #output = split_docs_and_answer(ori_output)
                ori_docs = []
                for i in range(5):
                    ori_docs.append(data[idx]['docs'][i]['text'])
                fk = {
                    'qa_pairs' : qa_pairs,
                    'answer' : answer,
                    'query' : query,
                    'docs' : ori_docs,
                    'output' : ori_output
                }
                test_data.append(fk)
            ans, rec, pre = compute_autoais(test_data)
if args.dataset == 'asqa':
                ans2, acc = compute_str_em(test_data)
            elif args.dataset == 'eli5':
                ans2, acc = compute_claims(test_data)
            
            print(ans2)
            print(ans)
    with open(args.dir, "r", encoding="utf-8") as fuck,\
        open(args.dir[:-6] + '_output.jsonl', "w", encoding="utf-8") as outputf:
            for idx, line in enumerate(fuck):
                opt = json.loads(line)
                opt['accuracy'] = acc[idx]
                opt['precision'] = pre[idx]
                opt['recall'] = rec[idx]
                outputf.write(json.dumps(opt, ensure_ascii=False) + '\n')"""
def main():

    def split_docs_and_answer(input_str):

        tag = "</think>"
        index = input_str.find(tag)
        if index != -1:
            return input_str[index + len(tag):]
        else:
            return "" 
        

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()

    test_data = []
    with open (args.dir, "r", encoding = "utf-8") as file:
        with open(f'/yy21/MoE-PEFT/dataset/front_output/{args.dataset}.json', "r", encoding="utf-8") as f:
            fuck = json.load(file)
            data = json.load(f)
            cnt = 0
            for idx, opt in enumerate(fuck):
                if opt['response'] == None:
                   cnt += 1
                   opt['response'] = ""
                ori_output = re.sub(r'\[ref_(\d+)\]', r'[\1]', opt['response'])
                if args.dataset == 'asqa':
                   qa_pairs = data[idx]['qa_pairs']
                elif args.dataset == 'eli5':
                    qa_pairs = data[idx]['claims']
                answer = data[idx]['answer']
                query = data[idx]['question']

                #output = split_docs_and_answer(ori_output)
ori_docs = []
                for i in range(5):
                    ori_docs.append(data[idx]['docs'][i]['text'])
                fk = {
                    'qa_pairs' : qa_pairs,
                    'answer' : answer,
                    'query' : query,
                    'docs' : ori_docs,
                    'output' : ori_output
                }
                test_data.append(fk)
            print(cnt)
            input()
            ans, rec, pre = compute_autoais(test_data)
            if args.dataset == 'asqa':
                ans2, acc = compute_str_em(test_data)
            elif args.dataset == 'eli5':
                ans2, acc = compute_claims(test_data)
            
            print(ans2)
            print(ans)
    with open(args.dir, "r", encoding="utf-8") as fuck,\
        open(args.dir[:-6] + '_output.json', "w", encoding="utf-8") as outputf:
            for idx, opt in enumerate(fuck):
                opt['accuracy'] = acc[idx]
                opt['precision'] = pre[idx]
                opt['recall'] = rec[idx]
                outputf.write(json.dumps(opt, ensure_ascii=False) + '\n')


if __name__ == "__main__":

    main()