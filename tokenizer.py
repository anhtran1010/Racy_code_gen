from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os 
import json

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()

corpus = []
code_dict = []
for file_name in os.listdir("corpus_no_comment"):
    # file_path = os.path.join("corpus", file_name)
    # with open(file_path, "r") as cf:
    #     lines = cf.readlines()
    full_path = os.path.join("corpus_no_comment", file_name)
    corpus.append(full_path)
    with open(full_path, 'r') as fn:
        source_code = fn.read()
    if source_code == []:
        print(full_path)
    code_dict.append((file_name,source_code))

# for sub_dir in os.listdir("Indigo3_bug"):
#     for file_name in os.listdir(os.path.join("Indigo3_bug", sub_dir)):
#         write_path = "Indigo3_bug/"+sub_dir+"/"+file_name
#         corpus.append(write_path)      

# for sub_dir in os.listdir("Indigo3_nobug"):
#     for file_name in os.listdir(os.path.join("Indigo3_nobug", sub_dir)):
#         write_path = "Indigo3_nobug/"+sub_dir+"/"+file_name
#         corpus.append(write_path)  

tokenizer.train(corpus, trainer)
# print(tokenizer.get_vocab())
tokenizer.save("buggy_code_tokens.json")

with open("race_codes.json", "w") as cf:
    json.dump(code_dict, cf)