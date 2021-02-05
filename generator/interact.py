"""
Script for interacting with the generator model via the terminal.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

def main():
    # load the args & config
    parser = argparse.ArgumentParser("Interact with the generator model")
    parser.add_argument("--modelpath", default="gpt2-large", required=False, help="Path to the Huggingface Transformers GPT-2 model to load.")
    parser.add_argument("--force-cpu", action="store_true", required=False, help="Force the device to cpu even if a supported GPU is present.")
    
    args = parser.parse_args()
    
    print()
    print("Running with arguments:")
    print(args)
    print()
    
    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.modelpath)
    model = AutoModelForCausalLM.from_pretrained(args.modelpath)
    
    if args.force_cpu:
        device = torch.device("cpu")
    else:    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device == "cuda":
        model = model.half()
    model.to(device)
    model.eval()
    
    #Start the interaction loop
    dialog_history = []
    while True:
        user_input = input(">> User: ")
        if user_input == "exit":
            break
        
        #Add last input to the dialog history and concatenate to create model input
        dialog_history.append(user_input)
        model_input = '"' + '" "'.join(dialog_history) + '"' + tokenizer.eos_token
        
        #Tokenize the model input
        model_input_ids = tokenizer.encode(model_input, return_tensors="pt")
        model_input_ids = model_input_ids.to(device)
        
        #Generate the result
        result_ids = model.generate(model_input_ids,
                                    max_length=128,
                                    pad_token_id=tokenizer.eos_token_id,
                                    top_k=50,
                                    top_p=0.95,
                                    do_sample=True,
                                    temperature=1.2,
                                    num_beams=1,
                                    early_stopping=True,
                                    num_return_sequences=1)
        
        result_ids = result_ids.to("cpu")
        
        #Print the result(s)
        for i in range(result_ids.shape[0]):
            print("Generator: {}".format(tokenizer.decode(result_ids[i], skip_special_tokens=True)))
            print()


if __name__ == "__main__":
    main()