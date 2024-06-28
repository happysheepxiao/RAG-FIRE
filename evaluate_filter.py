import torch
import jsonlines
from nltk import sent_tokenize
from ralm.scorer import UniEvaluator
from prettytable import PrettyTable
import numpy as np
import argparse

def convert_to_json(output_list, src_list=None):
    json_data = []
    for i in range(len(output_list)):
        cur = {}
        cur['system_output'] = output_list[i]
        if src_list is not None:
            cur['source'] = src_list[i]
        json_data.append(cur)
    return json_data


def print_scores(scores):
    table = PrettyTable(['Dimensions','Score'])
    print('\nEvaluation scores are shown below:')
    dims = list(scores[0].keys())
    for dim in dims:
        cur_score = 0
        for i in range(len(scores)):
            cur_score += scores[i][dim]
        table.add_row([dim, round(cur_score / len(scores), 6)])
    print(table)


def add_question(output, src=None):
    
    input_with_question = []
    for i in range(len(output)):
        cur_input = 'question: Is this question relevant to the document? </s> question: ' + output[i] + ' </s> document: ' + src[i]
        input_with_question.append(cur_input)

    return input_with_question


def evaluate(scorer, data, print_result=False):

    n_data = len(data)
    eval_scores = [{} for _ in range(n_data)]

    print('Evaluating {} samples !!!'.format(n_data))

    # Calculate average sentence-level scores for facutal consistency
    src_list, output_list = [], []
    n_sents = [] # the number of sentences in the claim

    for i in range(n_data):
        source = data[i]['source']
        system_outputs = sent_tokenize(data[i]['system_output'])

        n_sents.append(len(system_outputs))
        for j in range(len(system_outputs)):
            src_list.append(source)
            output_list.append(system_outputs[j])


    input_list = add_question(output=output_list, src=src_list)
    sent_score = scorer.score(input_list)
    
    # Get average score for each sample
    start_idx = 0
    score = []
    for cur_n_sent in n_sents:
        score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]) / cur_n_sent)
        start_idx += cur_n_sent
    
    # import pdb
    # pdb.set_trace()


    for i in range(n_data):
        eval_scores[i]["relevance"] = score[i]

    if print_result == True:
        print_scores(eval_scores)

    return eval_scores


def main(args):

    with jsonlines.open(args.input_path, 'r') as reader:
        data_test = list(reader)

    question_list = []
    document_list = []
    label_list = []
    for i in range(len(data_test)):
        question = data_test[i]["src"].split(" </s> ")[1].split(": ")[1]
        document = data_test[i]["src"].split(" </s> ")[2].split(": ")[1]
        question_list.append(question)
        document_list.append(document)
        if data_test[i]["tgt"] == "Yes":
            label_list.append(1)
        else:
            label_list.append(0)


    # Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=question_list, src_list=document_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    scorer = UniEvaluator(model_name_or_path=args.model_path, max_length=1024, device=device)

    eval_scores = evaluate(scorer, data, print_result=True)

    data_list = []
    for i in range(len(data)):
        data_list.append({'score': eval_scores[i]['relevance'], 'label': label_list[i], 'prompt': data_test[i]['src']})

    with jsonlines.open(args.output_path, 'w') as writer:
        writer.write_all(data_list)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="", help='Path to data')
    parser.add_argument('--output_path', type=str, default="", help='Path to data')
    parser.add_argument('--model_path', type=str, default="", help='Path to data')

    args = parser.parse_args()

    main(args)