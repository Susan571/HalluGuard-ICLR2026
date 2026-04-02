import os
import numpy as np
import pickle as pkl
import evaluate
from rouge_score import rouge_scorer
import math
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from metric import *
from plot import *

USE_Roberta = False
USE_EXACT_MATCH = True
##### 导入ROUGE评估函数计算ROUGE-L指标
###### 导入roberta_large模型计算sentence similarity
rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
if USE_Roberta:
    SenSimModel = SentenceTransformer('../data/weights/nli-roberta-large')


##### 打印结果信息, resultDict is a list of dict
def printInfo(resultDict):
    print(len(resultDict))
    for item in resultDict:
        for key in item.keys():
            print(key)
        exit()



#### 计算LLM模型输出answer的准确率
def getAcc(resultDict, file_name):
    correctCount = 0
    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # print("GT:", ansGT)
        # print("Generation:", generations)
        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        if "coqa" in file_name or "TruthfulQA" in file_name:
            additional_answers = item["additional_answers"]
            rougeScores = [getRouge(rougeEvaluator, generations, ansGT) for ansGT in additional_answers]
            rougeScore = max(rougeScore, max(rougeScores))
        if rougeScore>0.5:
            correctCount += 1
    print("Acc:", 1.0*correctCount/len(resultDict))



##### 计算皮尔逊相关系数
def getPCC(x, y):
    rho = np.corrcoef(np.array(x), np.array(y))
    return rho[0,1]



##### 计算度量指标的AUROC
def getAUROC(resultDict, file_name):
    Label = []
    Score = []
    Perplexity = []
    Energy = []
    LexicalSimilarity = []
    SentBertScore = []
    Entropy = []
    EigenIndicator = []
    EigenIndicatorOutput = []
    NTKS3Indicator = []
    NTKS3IndicatorOutput = []

    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # print("GT:", ansGT)
        # print("Generation:", generations)
        Perplexity.append(-item["perplexity"])
        Energy.append(-item["energy"])
        Entropy.append(-item["entropy"])
        LexicalSimilarity.append(item["lexical_similarity"])
        SentBertScore.append(-item["sent_bertscore"])
        EigenIndicator.append(-item["eigenIndicator"])
        EigenIndicatorOutput.append(-item["eigenIndicatorOutput"])
        NTKS3Indicator.append(-item["ntks3Indicator"])
        NTKS3IndicatorOutput.append(-item["ntks3IndicatorOutput"])


        if USE_Roberta:
            similarity = getSentenceSimilarity(generations, ansGT, SenSimModel)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                similarities = [getSentenceSimilarity(generations, ansGT, SenSimModel) for ansGT in additional_answers]
                similarity = max(similarity, max(similarities))
            if similarity>0.9:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
        elif USE_EXACT_MATCH:
            similarity = compute_exact_match(generations, ansGT)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                similarities = [compute_exact_match(generations, ansGT) for ansGT in additional_answers]
                similarity = max(similarity, max(similarities))
            if similarity==1:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
        else:
            rougeScore = getRouge(rougeEvaluator, generations, ansGT)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                rougeScores = [getRouge(rougeEvaluator, generations, ansGT) for ansGT in additional_answers]
                rougeScore = max(rougeScore, max(rougeScores))
            if rougeScore>0.5:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(rougeScore)


######### 计算AUROC + F1 + TPR@FPR ###########
    all_methods = {
        "Perplexity": Perplexity,
        "Energy": Energy,
        "Entropy": Entropy,
        "LexicalSim": LexicalSimilarity,
        "SentBertScore": SentBertScore,
        "EigenScore": EigenIndicator,
        "EigenScore-Output": EigenIndicatorOutput,
        "NTK-S3 (HalluGuard)": NTKS3Indicator,
        "NTK-S3-Output": NTKS3IndicatorOutput,
    }
    thresholds_dict = {}
    for name, method_scores in all_methods.items():
        fpr, tpr, thresholds = roc_curve(Label, method_scores)
        AUROC = auc(fpr, tpr)
        thresh = get_threshold(thresholds, tpr, fpr)
        thresholds_dict[name] = thresh

        f1 = getF1(Label, method_scores, thresh)
        tpr_at_5 = getTPRatFPR(fpr, tpr, 0.05)
        tpr_at_10 = getTPRatFPR(fpr, tpr, 0.10)

        print(f"--- {name} ---")
        print(f"  AUROC:       {AUROC:.4f}")
        print(f"  F1:          {f1:.4f}")
        print(f"  TPR@5%FPR:   {tpr_at_5:.4f}")
        print(f"  TPR@10%FPR:  {tpr_at_10:.4f}")
        try:
            VisAUROC(tpr, fpr, AUROC, name, file_name.split("_")[1])
        except Exception:
            pass


######## 计算皮尔逊相关系数 ###############
    print("\n--- Pearson Correlation Coefficients ---")
    for name, method_scores in all_methods.items():
        rho = getPCC(Score, method_scores)
        print(f"  PCC-{name}: {rho:.4f}")



######### 计算幻觉检测准确率(TruthfulQA)
    if "TruthfulQA" in file_name:
        for name, method_scores in all_methods.items():
            thresh = thresholds_dict[name]
            acc = getTruthfulQAAccuracy(Label, method_scores, thresh)
            print(f"TruthfulQA {name} Accuracy: {acc:.4f}")



def getF1(labels, scores, threshold):
    """Compute F1 score at the given threshold."""
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def getTPRatFPR(fpr_arr, tpr_arr, target_fpr):
    """Interpolate TPR at a specific FPR operating point."""
    fpr_arr = np.array(fpr_arr)
    tpr_arr = np.array(tpr_arr)
    if len(fpr_arr) == 0:
        return 0.0
    idx = np.searchsorted(fpr_arr, target_fpr, side='right') - 1
    idx = max(0, min(idx, len(fpr_arr) - 1))
    return float(tpr_arr[idx])


def get_threshold(thresholds, tpr, fpr):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    return thresholdOpt



def getTruthfulQAAccuracy(Label, Score, thresh):
    count = 0
    for ind, item in enumerate(Score):
        if item>=thresh and Label[ind]==1:
            count+=1
        if item<thresh and Label[ind]==0:
            count+=1
    return count/len(Score)



def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


if __name__ == "__main__":
    file_name = "../data/output/llama-7b-hf_coqa_1/0.pkl"
    # file_name = "../data/output/llama-7b-hf_triviaqa_3/0.pkl"
    # file_name = "../data/output/llama-7b-hf_nq_open_1/0.pkl"
    # file_name = "../data/output/llama-7b-hf_SQuAD_1/0.pkl"

    # file_name = "../data/output/opt-6.7b_triviaqa_0/0.pkl"
    # file_name = "../data/output/opt-6.7b_nq_open_2/0.pkl"
    # file_name = "../data/output/opt-6.7b_coqa_0/0.pkl"
    # file_name = "../data/output/opt-6.7b_SQuAD_0/0.pkl"

    # file_name = "../data/output/llama-13b-hf_coqa_0/0.pkl"
    # file_name = "../data/output/llama-13b-hf_triviaqa_0/0.pkl"
    # file_name = "../data/output/llama-13b-hf_nq_open_3/0.pkl"
    # file_name = "../data/output/llama-13b-hf_SQuAD_0/0.pkl"

    # file_name = "../data/output/llama-7b-hf_TruthfulQA_7/0.pkl"

    # file_name = "../data/output/llama2-7b-hf_coqa_0/0.pkl"
    # file_name = "../data/output/llama2-7b-hf_nq_open_0/0.pkl"

    # file_name = "../data/output/falcon-7b_coqa_0/0.pkl"
    # file_name = "../data/output/falcon-7b_nq_open_0/0.pkl"


    f = open(file_name, "rb")
    resultDict = pkl.load(f)
    # printInfo(resultDict)
    getAcc(resultDict, file_name)
    getAUROC(resultDict, file_name)

