"""
Script for interacting with the generator model via the terminal.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

def main():
    # load the args & config
    parser = argparse.ArgumentParser("Interact with the generator model")
    parser.add_argument("--modelpath", default="output/checkpoint-7000", required=False, help="Path to the Huggingface Transformers GPT-2 model to load.")
    parser.add_argument("--force-cpu", action="store_true", required=False, help="Force the device to cpu even if a supported GPU is present.")
    # TODO: support a larger set of options such as setting inference hyperparams.
    # Also support a user menu to set most of these options during runtime, like
    # https://github.com/AbrahamSanders/seq2seq-chatbot/blob/master/seq2seq-chatbot/chat_command_handler.py

    args = parser.parse_args()
    
    print()
    print("Running with arguments:")
    print(args)
    print()
    
    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.modelpath)
    model = AutoModelForCausalLM.from_pretrained(args.modelpath)
    
    narrative_token, dialog_token = tokenizer.additional_special_tokens
    
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
    turn_counter = 0
    post_token = dialog_token
    dialog_history = []
    while True:
        if post_token == dialog_token:
            user_input = input(">> User: ")
            turn_counter += 1
            if user_input == "exit":
                break
        
        #Add last input to the dialog history and concatenate to create model input
        if turn_counter % 3 == 0 and post_token == dialog_token:
            post_token = narrative_token
        else:
            post_token = dialog_token
        dialog_history.append(dialog_token + user_input + tokenizer.eos_token + post_token)
        model_input = "".join(dialog_history)
        
        #Tokenize the model input
        model_input_ids = tokenizer.encode(model_input, return_tensors="pt")
        model_input_ids = model_input_ids.to(device)
        
        #Generate the result
        #TODO: verify that increasing temperature proportionally to input length
        #      actually does help the model avoid repetition loops.
        result_ids = model.generate(model_input_ids,
                                    max_length=tokenizer.model_max_length,
                                    pad_token_id=tokenizer.pad_token_id,
                                    length_penalty=0.1,
                                    top_k=50,
                                    top_p=0.95,
                                    do_sample=True,
                                    temperature=1.5 + 0.005 * model_input_ids.shape[1],
                                    num_beams=6,
                                    early_stopping=True,
                                    num_return_sequences=1)
        
        result_ids = result_ids.to("cpu")
        
        #Print the result(s)
        for i in range(result_ids.shape[0]):
            generated_text = tokenizer.decode(result_ids[:, model_input_ids.shape[-1]:][i], 
                                              skip_special_tokens=True)
            if i == 0:
                dialog_history.append(generated_text)
                
            if post_token == narrative_token:
                generated_text = "***{}***".format(generated_text)
            print("Generator: {}".format(generated_text))
            print()


if __name__ == "__main__":
    main()