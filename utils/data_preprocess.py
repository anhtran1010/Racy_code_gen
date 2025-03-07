from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os 
import json

def create_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()

    corpus = []
    for file_name in os.listdir("/home/abtran/Racy_code_gen/corpus_no_comment"):
        # file_path = os.path.join("corpus", file_name)
        # with open(file_path, "r") as cf:
        #     lines = cf.readlines()
        full_path = os.path.join("/home/abtran/Racy_code_gen/corpus_no_comment", file_name)
        if os.path.isfile(full_path) and "signaling" not in file_name:
            if "yes" in file_name:
                with open(full_path, 'r') as fn:
                    source_code = fn.readlines()
                if source_code == []:
                    print(full_path)
                else:
                    corpus.append(full_path)

    # for sub_dir in os.listdir("/home/abtran/Racy_code_gen/Indigo3_bug"):
    #     for file_name in os.listdir(os.path.join("/home/abtran/Racy_code_gen/Indigo3_bug", sub_dir)):
    #         write_path = "/home/abtran/Racy_code_gen/Indigo3_bug/"+sub_dir+"/"+file_name
    #         corpus.append(write_path)      

    tokenizer.train(corpus, trainer)
    # print(tokenizer.get_vocab())
    tokenizer.save("buggy_code_tokenizer.json")

def create_corpus():
    corpus = []
    for file_name in os.listdir("/home/abtran/Racy_code_gen/corpus_no_comment"):
        # file_path = os.path.join("corpus", file_name)
        # with open(file_path, "r") as cf:
        #     lines = cf.readlines()
        full_path = os.path.join("/home/abtran/Racy_code_gen/corpus_no_comment", file_name)
        if os.path.isfile(full_path) and "signaling" not in file_name:
            if "yes" in file_name:
                with open(full_path, 'r') as fn:
                    source_code = fn.read()
                if source_code == []:
                    print(full_path)
                else:
                    corpus.append((file_name,source_code))
    with open("racy_codes.json", "w") as rf:
        json.dump(corpus, rf)
    # for sub_dir in os.listdir("/home/abtran/Racy_code_gen/Indigo3_bug"):
    #     for file_name in os.listdir(os.path.join("/home/abtran/Racy_code_gen/Indigo3_bug", sub_dir)):
    #         write_path = "/home/abtran/Racy_code_gen/Indigo3_bug/"+sub_dir+"/"+file_name
    #         with open(full_path, 'r') as fn:
    #             source_code = fn.readlines()
    #         corpus.append(source_code)      
    
# create_tokenizer()
create_corpus()