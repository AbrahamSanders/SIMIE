"""
Script for interacting with the generator model via the terminal.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import numpy as np

from interact_helpers import preprocess_input, postprocess_output

def main():
    # load the args & config
    parser = argparse.ArgumentParser("Interact with the generator model")
    parser.add_argument("--modelpath", default="output/checkpoint-21000", required=False, help="Path to the Huggingface Transformers GPT-2 model to load.")
    parser.add_argument("--force-cpu", action="store_true", required=False, help="Force the device to cpu even if a supported GPU is present.")
    parser.add_argument("--prompt-narrative-prob", type=float, default=0.3, required=False, help="Probability that the model will get prompted to generate narrative at each turn.")
    parser.add_argument("--max-input-tokens", type=int, default=512, required=False, help="Maximum number of tokens to use as input. Dialog history gets trimmed from the back to accommodate this.")
    parser.add_argument("--print-raw", action="store_true", required=False, help="Print the raw model input and output for debugging purposes.")
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
    narrative_token = tokenizer.additional_special_tokens[0]
    dialog_history = []
    while True:
        user_input = input(">> User: ")
        if user_input == "--exit":
            break
        if user_input == "--reset":
            dialog_history.clear()
            print("Dialog history cleared.")
            print()
            continue
        if user_input == "--print-raw":
            args.print_raw = not args.print_raw
            print("print_raw set to %s." % args.print_raw)
            continue
        
        if bool(np.random.binomial(1, args.prompt_narrative_prob)):
            generate(args, model, device, tokenizer, dialog_history, user_input, prompt_narrative=True)
        else:    
            generate(args, model, device, tokenizer, dialog_history, user_input)
        
        #If a narrative is generated, generate a follow-up dialog response to the user input.
        if dialog_history[-1].startswith(narrative_token):
            generate(args, model, device, tokenizer, dialog_history, user_input, prompt_dialog=True)
            #remove the second copy of the user input from the dialog history.
            dialog_history.pop(-2)
        
def generate(args, model, device, tokenizer, dialog_history, user_input=None, prompt_narrative=False, prompt_dialog=False):
    if prompt_narrative and prompt_dialog:
        raise ValueError("One of prompt_narrative or prompt_dialog may be true, not both.")
    has_continuation_prompt = prompt_narrative or prompt_dialog
        
    narrative_token, dialog_token = tokenizer.additional_special_tokens
    
    #If provided, preprocess user input and add to the dialog history
    if user_input is not None:
        processed_input = preprocess_input(user_input, narrative_token, 
                                           dialog_token, tokenizer.eos_token)
        dialog_history.append(processed_input)
    
    #Tokenize the model input, trimming the dialog history to stay below the maximum length.
    while True:
        model_input = "".join(dialog_history)
        if has_continuation_prompt:
            model_input += narrative_token if prompt_narrative else dialog_token
        
        model_input_ids = tokenizer.encode(model_input, return_tensors="pt")
        if model_input_ids.shape[-1] <= args.max_input_tokens:
            break
        dialog_history.pop(0)
    
    if args.print_raw:
        print(model_input)
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
                                temperature=1.5 + 0.003 * model_input_ids.shape[-1],
                                num_beams=6,
                                early_stopping=True,
                                num_return_sequences=1)
    
    result_ids = result_ids.to("cpu")
    
    #Print the result(s)
    for i in range(result_ids.shape[0]):
        response_start_idx = model_input_ids.shape[-1]
        if has_continuation_prompt:
            response_start_idx -= 1
            
        generated_text = tokenizer.decode(result_ids[:, response_start_idx:][i], 
                                          skip_special_tokens=False)
        if i == 0:
            dialog_history.append(generated_text)
        
        if args.print_raw:
            processed_output = generated_text
        else:
            processed_output = postprocess_output(generated_text, narrative_token, 
                                                  dialog_token, tokenizer.eos_token)
        print("Generator: {}".format(processed_output))
        print()

if __name__ == "__main__":
    main()
