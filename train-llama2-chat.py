import torch, transformers
from huggingface_hub import login

import pyreft
import torch, transformers
import argparse

from utils import get_device

from pyreft import (
    ReftModel,
    get_intervention_locations
)

def parse_args():
    parser = argparse.ArgumentParser(description='PyReft and Training arguments.')

    # Hugging Face args
    parser.add_argument('--model_type', type=str, choices=["llama-2-7b", "llama-2-13b"], default="llama-2-7b", help='LLM to finetune')
    parser.add_argument('--model_max_length', type=int, default=2048, help="Max number of input tokens.")
    parser.add_argument('--access_token_read', type=str, help="Token used to login on HuggingFace", required=True)
    
    # PyReFT args
    parser.add_argument("--low_rank_dimension", type=int, help="Rank of the low-ranked matrix")
    parser.add_argument('--layers', type=str, help='List of layers delimited by comma on which interventions are applied.', default="3,9,18,24", required=True)
    parser.add_argument('-p', '--prefix', type=int, help="Number of first p tokens on which interventions are applied.")
    parser.add_argument('-s', '--suffix', type=int, help="Number of last s tokens on which interventions are applied.") 
    parser.add_argument('--reft-model-type', type=str, choices=['loreft', 'direft'], default='loreft')

    # Training args
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3) # ???
    parser.add_argument("--warmup_ratio", type=float, default=0.00)
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--logging_steps", type=int, default=1000)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    print(f"Training on {device}...")
    
    # Login on HuggingFace
    access_token_read = args.access_token_read
    login(token = access_token_read)

    # Get HF model path
    if args.model_type == "llama-2-7b":
        model_path = "meta-llama/Llama-2-7b-hf"
    elif args.model_type == "llama-2-13b":
        model_path = "meta-llama/Llama-2-13b-hf"
    else:
        raise ValueError(f"{args.model_type} is undefined.")
    
    print(f"HF model path: {model_path}")

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=args.model_max_length, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token

    # Pretrained LLM
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device)

    # Define representations
    layers = [int(layer) for layer in args.layers.split(',')]
    print(f'Layers: {layers}')

    representations = []
    for layer in layers:
        repr = dict({
            "layer": layer,
            "component": "block_output",
            "low_rank_dimension": args.low_rank_dimension,
            "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size, low_rank_dimension=args.low_rank_dimension),
        })
        representations.append(repr)

    # Initialize ReFT model
    reft_config = pyreft.ReftConfig(representations=representations)
    print(f'ReFT config: {reft_config}')

    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)
    reft_model.print_trainable_parameters()

    # # TODO: Get data loader
    # data_module = ...

    # # train
    # training_args = transformers.TrainingArguments(
    #     num_train_epochs=args.num_epochs, output_dir=args.output_dir, per_device_train_batch_size=args.batch_size, 
    #     learning_rate=args.lr, logging_steps=40, report_to=[])
    # trainer = pyreft.ReftTrainerForCausalLM(
    #     model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)
    # _ = trainer.train()

    # # save model
    # reft_model.set_device("cpu") # send back to cpu before saving.
    # reft_model.save(
    #     save_directory="./reft_to_share",
    # )




    


    
