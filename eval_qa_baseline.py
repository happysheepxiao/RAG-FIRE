import os
import argparse
import json
import re
import string

import torch
from tqdm import tqdm

from ralm.file_utils import print_args
from ralm.model_utils import load_model_and_tokenizer

import jsonlines

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def build_qa_prompt(example, num_docs=1):
    if num_docs == 0:
        question_text = normalize_question(example["question"])
        ex_prompt = f"Answer these questions:\nQ: {question_text}\nA:"
    elif num_docs == 1:
        q = normalize_question(example["question"])
        title = example['ctxs'][0]['title']
        text = example['ctxs'][0]['text']
        ex_prompt = f"{title}\n\n{text}\n\nBased on this text, answer these questions:\nQ: {q}\nA:"

        # examples = [
        #     {"answer": ["1950"], "question": "when was puerto rico added to the usa"},
        #     {"answer": ["Jan Hammer"], "question": "who sings the theme song for miami vice"},
        #     {"answer": ["41,629"], "question": "what is the population of farmington new mexico"},
        #     {"answer": ["Minnesota"], "question": "where is walnut grove on little house on the prairie"},
        #     {"answer": ["Pink rhododendron"], "question": "which is the state flower of himachal pradesh"}
        # ]

        # example_prompt = ""
        # for example in examples:
        #     question = example["question"]
        #     answer = example["answer"][0]
        #     example_prompt += f"Q: {question}\nA: {answer}\n\n"
        
        # ex_prompt = f"{example_prompt}Based on the following texts, answer the question below:\n\n{title}\n\n{text}\n\nQ: {q}\nA:"


    else:
        q = normalize_question(example["question"])
        docs_text = "\n\n".join([f"{ctx['title']}\n\n{ctx['text']}" for ctx in example["ctxs"][:num_docs]])

        ex_prompt = f"{docs_text}\n\nBased on these texts, answer these questions:\nQ: {q}\nA:"


        # examples = [
        #     {"answer": ["1950"], "question": "when was puerto rico added to the usa"},
        #     {"answer": ["Jan Hammer"], "question": "who sings the theme song for miami vice"},
        #     {"answer": ["41,629"], "question": "what is the population of farmington new mexico"},
        #     {"answer": ["Minnesota"], "question": "where is walnut grove on little house on the prairie"},
        #     {"answer": ["Pink rhododendron"], "question": "which is the state flower of himachal pradesh"}
        # ]

        # example_prompt = ""
        # for example in examples:
        #     question = example["question"]
        #     answer = example["answer"][0]
        #     example_prompt += f"Q: {question}\nA: {answer}\n\n"

        # ex_prompt = f"{example_prompt}Based on the following texts, answer the question below:\n\n{docs_text}\n\nQ: {q}\nA:"


    return ex_prompt


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


def text_has_answer(answers, text) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def get_answer_from_model_output(outputs, tokenizer, prompt):
    generation_str = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)

    if generation_str.startswith("[gMASK]sop "):
        generation_str = generation_str[len("[gMASK]sop "):][len(prompt):]
    else:       
        generation_str = generation_str[len(prompt):]
    answer = generation_str.split("\n")[0]
    return answer, generation_str


def evaluate_dataset(
        model, tokenizer, device, eval_dataset, max_length, num_docs=0, output_dir=None, max_tokens_to_generate=10
):
    idx = 0
    num_correct = 0
    num_has_answer = 0
    num_too_long = 0
    sample_prompt = None

    answer_list = []
    for ex in (tq := tqdm(eval_dataset, desc=f"EM:  0.0%")):
        if "answers" in ex:
            answers = ex["answers"]
        else:
            answers = ex["answer"]

        prompt = build_qa_prompt(ex, num_docs=num_docs)
        if idx == 0:
            sample_prompt = prompt
        has_answer = text_has_answer(answers, prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[-1] > max_length - max_tokens_to_generate:
            num_too_long += 1
            input_ids = input_ids[..., -(max_length - max_tokens_to_generate):]

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=max_tokens_to_generate)

        prediction, generation = get_answer_from_model_output(outputs, tokenizer, prompt)

        is_correct = any([exact_match(prediction, answer) for answer in answers])
        # is_correct = text_has_answer(answers, prediction)

        # import pdb
        # pdb.set_trace()

        answer_list.append({"is_correct": is_correct, "prediction": prediction, "answers": answers, "question": ex["question"], "prompt": prompt})

        idx += 1
        if is_correct:
            num_correct += 1
        if has_answer:
            num_has_answer += 1
        tq.set_description(f"EM: {num_correct / idx * 100:4.1f}%")

    em = num_correct / idx * 100
    has_answer = num_has_answer / idx * 100
    print(f"EM: {em:.2f}%")
    print(f"% of prompts with answer: {num_has_answer / idx * 100:.1f}%")
    if output_dir is not None:
        d = {"em": em, "has_answer": has_answer, "num_examples": idx, "too_long": num_too_long}
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")
        if sample_prompt is not None:
            with open(os.path.join(output_dir, "example_prompt.txt"), "w") as f:
                f.write(sample_prompt)
        with jsonlines.open(os.path.join(output_dir, "answer.json"), "w") as writer:
            writer.write_all(answer_list)



def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    # with open(dataset_path) as f:
    #     return json.load(f)

    import jsonlines
    with jsonlines.open(dataset_path, 'r') as reader:
        data = [obj for obj in reader]
        return data


def main(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir)
    print_args(args, output_dir=args.output_dir)

    print("Loading model:", args.model_name)
    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )
    # import pdb
    # pdb.set_trace()

    if "chatglm2" in args.model_name or "chatglm3" in args.model_name:
        model_max_length = config.seq_length
    elif "chatglm" in args.model_name:
        model_max_length = config.max_sequence_length
    else:
        model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
    print(model_max_length)

    eval_dataset = load_dataset(args.dataset_path)

    evaluate_dataset(
        model, tokenizer, device, eval_dataset,
        max_length=model_max_length,
        num_docs=args.num_docs,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_docs", type=int, default=0)

    # Dataset params
    parser.add_argument("--dataset_path", type=str)

    args = parser.parse_args()

    main(args)
