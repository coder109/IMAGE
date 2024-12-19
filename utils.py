import os
import torch
import json
from tqdm import tqdm
import argparse
import re

# List of dicts -> l_o_d
def rewrite_into_json(l_o_d, file_name):
    with open(file_name, "w") as f:
        json.dump(l_o_d, f, ensure_ascii=False)

def write_into_json(l_o_d, file_name):
    with open(file_name, "a+") as f:
        json.dump(l_o_d, f, ensure_ascii=False)

def write_into_jsonl(l_o_d, file_name):
    with open(file_name, "a+") as outfile:
        for entry in l_o_d:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

def rewrite_into_jsonl(l_o_d, file_name):
    with open(file_name, "w") as outfile:
        for entry in l_o_d:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

def save_in_jsonl(l_o_d, file_name):
    if os.path.exists(file_name):
        rewrite_into_jsonl(l_o_d, file_name)
    else:
        write_into_jsonl(l_o_d, file_name)

def save_in_json(l_o_d, file_name):
    if os.path.exists(file_name):
        rewrite_into_json(l_o_d, file_name)
    else:
        write_into_json(l_o_d, file_name)
        
def sample_json(in_file, out_file, sample_num=1000):
    l_o_d = json.load(open(in_file, "r", encoding="utf-8"))
    save_in_json(l_o_d[:sample_num], out_file)

def load_dataset(file_path: str):
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        curr_line = f.readline()
        while curr_line != "":
            results.append(curr_line.replace("\n", ""))
            curr_line = f.readline()
    return results

def construct_result_block(src: str, tgt: str, hyp: str):
    return {"src": src, "tgt": tgt, "hyp": hyp}

def find_all_idx_of_char(my_str, my_char):
    idx_list = []
    idx = 0
    for c in my_str:
        if c == my_char:
            idx_list.append(idx)
        idx += 1
    return idx_list

def is_contain_alphas(my_str):
    for c in my_str:
        if c.isalpha():
            return True
    return False

def filter_result(file_path: str):
    result_answ = []
    with open(file_path, "r", encoding="utf-8") as f:
        contents = f.readlines()
    for content in tqdm(contents):
        # Convert str to dict
        curr_dict = eval(content)

        # Extract hyp field
        curr_text = curr_dict["hyp"]

        print(curr_text)

        # Remove all Chinese Characters
        curr_text = re.sub('[\u4e00-\u9fa5]', '', curr_text)

        # Extract content after ":"
        if ':' in curr_text:
            curr_text = curr_text[curr_text.index(':')+1:]

        # Extract content between " and "
        if '\"' in curr_text:
            quote_idx_list = find_all_idx_of_char(curr_text, '\"')
            if len(quote_idx_list) % 2 == 1:
                pass
            else:
                # Iterate over content between two quote symbols
                for list_idx in range(0, len(quote_idx_list), 2):
                    curr_quote_idx = quote_idx_list[list_idx]
                    suff_quote_idx = quote_idx_list[list_idx+1]
                    sub_str = curr_text[curr_quote_idx+1:suff_quote_idx]
                    if is_contain_alphas(sub_str):
                        curr_text = sub_str
                        break
                        
        curr_dict.update({"hyp": curr_text})
        result_answ.append(curr_dict)
    save_in_jsonl(result_answ, file_path)

if __name__ == "__main__":
    pass
