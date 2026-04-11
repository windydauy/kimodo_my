import torch
from transformers import AutoTokenizer
from peft import PeftModel

from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel

base_path = "./huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
mntp_path = "./huggingface/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/snapshots/31474e395ada192e8ed1586db6be79fb3b70c9c0"
output_path = "./local_models/llama3_mntp_merged"

tokenizer = AutoTokenizer.from_pretrained(mntp_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = LlamaBiModel.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(model, mntp_path)
model = model.merge_and_unload()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"saved merged model to: {output_path}")