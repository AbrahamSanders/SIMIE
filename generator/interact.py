"""
Script for interacting with the generator model via the terminal.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import numpy as np

from identities import Identities
import interact_helpers

def main():
    # load the args & config
    parser = argparse.ArgumentParser("Interact with the generator model")
    parser.add_argument("--modelpath", default="models/gpt2-xl-dialog-narrative", required=False, 
                        help="Path to the Huggingface Transformers GPT-2 model to load. (default: %(default)s)")
    parser.add_argument("--force-cpu", action="store_true", required=False, 
                        help="Force the device to cpu even if a supported GPU is present.")
    parser.add_argument("--prompt-narrative-prob", type=float, default=0.2, required=False, 
                        help="Probability that the model will get prompted to generate narrative at each turn. (default: %(default)s)")
    parser.add_argument("--max-input-tokens", type=int, default=350, required=False, 
                        help="Maximum number of tokens to use as input. Dialog history gets trimmed from the back to accommodate this. (default: %(default)s)")
    parser.add_argument("--print-raw", action="store_true", required=False, 
                        help="Print the raw model input and output for debugging purposes.")
    parser.add_argument("--speaker-tracking", action="store_true", default=True,
                        help="Enable speaker tracking through narrative prompts.")
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
    
    identities = Identities()
    
    #Start the interaction loop
    if args.speaker_tracking:
        identities.user = input("Enter your name: ")
        identities.generator = input("Enter a name for the generator: ")
    narrative_token = tokenizer.additional_special_tokens[0]
    dialog_history = []
    
    while True:
        identities.reset()
        generate_count = 1
        user_input = input(">> %s: " % identities.user)
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
        if user_input == "--auto-generate":
            if len(dialog_history) == 0:
                print("The first turn must be taken by the user.")
                continue
            user_input = None
            generate_count = 10

        for _ in range(generate_count):
            #When generating more than once, the generator will be impersonating the user every other step
            #so the identities need to be swapped.
            if generate_count > 1:
                identities.swap()

            #Generate the next step
            if bool(np.random.binomial(1, args.prompt_narrative_prob)):
                generate(args, model, device, tokenizer, dialog_history, identities, user_input, prompt_narrative=True)
            else:    
                generate(args, model, device, tokenizer, dialog_history, identities, user_input)
            
            #If a narrative is generated, generate a follow-up dialog response.
            if dialog_history[-1].startswith(narrative_token):
                generate(args, model, device, tokenizer, dialog_history, identities, prompt_dialog=True)

        
def generate(args, model, device, tokenizer, dialog_history, identities, user_input=None, prompt_narrative=False, prompt_dialog=False):
    if prompt_narrative and prompt_dialog:
        raise ValueError("One of prompt_narrative or prompt_dialog may be true, not both.")
    has_continuation_prompt = prompt_narrative or prompt_dialog
        
    narrative_token, dialog_token = tokenizer.additional_special_tokens
    
    #If provided, preprocess user input and add to the dialog history.
    speaker_tracking_reply = False
    if user_input is not None:
        processed_input = interact_helpers.preprocess_input(user_input, narrative_token, 
                                                            dialog_token, tokenizer.eos_token)
        for processed_segment in processed_input:
            if args.speaker_tracking:
                if processed_segment.startswith(narrative_token):
                    processed_segment = interact_helpers.force_third_person(processed_segment, identities.user)
                #Give the user a speaker tracking prompt if at least one dialog utterance
                #exists in the user input. If multiple dialog utterances exist, only give one
                #speaker tracking prompt.
                elif not speaker_tracking_reply:
                    dialog_history.append(narrative_token + identities.user + " said," + tokenizer.eos_token)
                    #The model should also receive a speaker tracking prompt since it should reply to the user's
                    #dialog utterance(s).
                    speaker_tracking_reply = True
            dialog_history.append(processed_segment)
        
    if args.speaker_tracking:
        #speaker_tracking_reply indicates if the model should receive a speaker tracking prompt
        #when speaker tracking is enabled. By default it is set to true if the user input exists
        #and contains at least one dialog (see immediately above). 
        #However there are some additional conditions to check:
        
        #Condition (A): If the model is receiving a dialog prompt, we always want to give it
        #a speaker tracking prompt even in the absence of a user input with dialog.
        if prompt_dialog:
            speaker_tracking_reply = True
        #Condition (B): In the absence of user input we can infer if the previous generated step was a dialog,
        #in which case we want to give the model a speaker tracking prompt.
        if user_input is None and len(dialog_history) > 0 and not dialog_history[-1].startswith(narrative_token):
            speaker_tracking_reply = True
            
        #The following conditions OVERRIDE the previous ones:
            
        #Condition (C): If the model is receiving a narrative prompt, it makes no sense to give it
        #a speaker tracking prompt since the model will not output a dialog utterance.
        if prompt_narrative:
            speaker_tracking_reply = False
        #Condition (D): It is possible that the model previously generated the speaker tracking prompt
        #after being prompted to generate narrative. In this case, we don't want to give it a duplicate prompt.
        if len(dialog_history) > 0 and interact_helpers.is_speaker_tracking_prompt(
                dialog_history[-1], identities.generator, narrative_token, tokenizer.eos_token):
            speaker_tracking_reply = False
        
        #Give the model a speaker tracking prompt if all the conditions are met.
        if speaker_tracking_reply:
            action = " said," if identities.is_swapped else " replied,"
            dialog_history.append(narrative_token + identities.generator + action + tokenizer.eos_token)
    
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
                                #length_penalty=0.1,
                                top_k=50,
                                top_p=0.95,
                                do_sample=True,
                                temperature=1.5 + 0.002 * model_input_ids.shape[-1],
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
            processed_output = interact_helpers.postprocess_output(generated_text, narrative_token, 
                                                                   dialog_token, tokenizer.eos_token)
        print("{0}: {1}".format(identities.generator, processed_output))
        print()

if __name__ == "__main__":
    main()
