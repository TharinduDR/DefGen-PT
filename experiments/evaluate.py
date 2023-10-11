import pyter
import itertools
import statistics

from datasets import load_metric
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


def bertscore(ref, gen):
    bertscore_metric = load_metric('bertscore')
    bert_scores = bertscore_metric.compute(predictions=gen, references=ref, lang="pt")
    return statistics.mean(bert_scores['f1'])

def bleurt_score(ref, gen):
    bleurt = load_metric("bleurt", 'bleurt-20')
    bleurt_scores = bleurt.compute(predictions=gen, references=ref)
    return statistics.mean(bleurt_scores)


def bleu(ref, gen):
    '''
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    ref_bleu = []
    gen_bleu = []
    for l in gen:
        gen_bleu.append(l.split())
    for i, l in enumerate(ref):
        ref_bleu.append([l.split()])
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    return score_bleu


def ter(ref, gen):
    '''
    Args:
        ref - reference sentences - in a list
        gen - generated sentences - in a list
    Returns:
        averaged TER score over all sentence pairs
    '''
    if len(ref) == 1:
        total_score = pyter.ter(gen[0].split(), ref[0].split())
    else:
        total_score = 0
        for i in range(len(gen)):
            total_score = total_score + pyter.ter(gen[i].split(), ref[i].split())
        total_score = total_score / len(gen)
    return total_score
