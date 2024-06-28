from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration
from tqdm import tqdm
import torch
import argparse

def main(args):

    model_path = args.model_path

    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model.to(device)

    count = 1
    bsz = 24

    total_len = args.total_len
    gen_max_len = args.total_len
    gen_min_len = args.gen_min_len
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    path = args.data_path

    with open(path + 'test.source') as source:
        lines = source.readlines()
    total_num = len(lines)
    with open(path + 'test.source') as source, open(path + 'test.out', 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in tqdm(source, total=total_num):
            if count % bsz == 0:
                with torch.no_grad():
                    dct = tokenizer.batch_encode_plus(slines, max_length=total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    summaries = model.generate(
                        input_ids=dct["input_ids"].to(device),
                        attention_mask=dct["attention_mask"].to(device),
                        max_length=gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=gen_min_len + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        early_stopping=True,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for hypothesis in dec:
                    hypothesis = hypothesis.replace("\n", " ")
                    fout.write(hypothesis + '\n')
                    fout.flush()

                slines = []
            sline = sline.strip()
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
            count += 1
        if slines != []:
            with torch.no_grad():
                dct = tokenizer.batch_encode_plus(slines, max_length=total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                summaries = model.generate(
                    input_ids=dct["input_ids"].to(device),
                    attention_mask=dct["attention_mask"].to(device),
                    max_length=gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=gen_min_len + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=True,
                )
                dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for hypothesis in dec:
                    hypothesis = hypothesis.replace("\n", " ")
                    fout.write(hypothesis + '\n')
                    fout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--total_len', type=int, default=1024, help='Total length')
    parser.add_argument('--gen_max_len', type=int, default=140, help='Maximum generation length')
    parser.add_argument('--gen_min_len', type=int, default=55, help='Minimum generation length')
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams')
    parser.add_argument('--length_penalty', type=float, default=2.0, help='Length penalty')
    parser.add_argument('--data_path', type=str, default="", help='Path to data')
    parser.add_argument('--model_path', type=str, default="", help='Path to model')

    args = parser.parse_args()

    main(args)