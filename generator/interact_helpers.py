"""
Utility methods for use during interaction
"""

def preprocess_input(user_input, narrative_token, dialog_token, eos_token):
    """
    Adds dialog, narrative, and eos special tokens to user input string before tokenizer encoding.
    Text surrounded with '***' signals a narrative passage. For example:
        
    "***She turned around*** What did you say?"
    becomes
    narrative_token + "She turned around" + eos_token + dialog_token + "What did you say?" + eos_token
    """
    segments = user_input.split("***")
    for i in range(len(segments)):
        seg = segments[i].strip()
        if seg != "":
            segments[i] = (narrative_token if i % 2 == 1 else dialog_token) + seg + eos_token
        else:
            segments[i] = seg
            
    segments = [seg for seg in segments if seg != ""]
    
    if len(segments) == 0:
        segments.append(dialog_token + eos_token)
        
    return segments

def postprocess_output(model_output, narrative_token, dialog_token, eos_token):
    """
    Performs the reverse of preprocess_input. Removes dialog, dialog, and eos special tokens from
    model output after tokenizer decoding. Text between a narrative_token and eos_token gets
    surrounded with '***'.
    """
    #Replace those eos tokens which immediately follow a narrative_token with "***"
    narrative_token_idx = -len(narrative_token)
    while True:
        narrative_token_idx = model_output.find(narrative_token, 
                                                narrative_token_idx + len(narrative_token))
        if narrative_token_idx == -1:
            break
        eos_token_idx = model_output.find(eos_token, narrative_token_idx)
        if eos_token_idx > -1:
            model_output = (model_output[:eos_token_idx] + "***" +
                            model_output[(eos_token_idx + len(eos_token)):])
    
    #Substitute all the remaining special tokens
    model_output = (model_output.replace(narrative_token, " ***")
                    .replace(dialog_token, " ")
                    .replace(eos_token, "")
                    .strip())
    
    return model_output

def force_third_person(segment, identity):
    if segment is None:
        return segment
    return (segment.replace(" I ", " %s " % identity)
                    .replace(" i ", " %s " % identity)
                    .replace("|>I ", "|>%s " % identity)
                    .replace("|>i ", "|>%s " % identity)
                    .replace(" I<|", " %s<|" % identity)
                    .replace(" i<|", " %s<|" % identity))

def is_speaker_tracking_prompt(segment, identity, narrative_token, eos_token):
    if segment is not None:
        if segment.startswith("%s%s said," % (narrative_token, identity)):
            return True
        if segment.startswith("%s%s replied," % (narrative_token, identity)):
            return True
        if segment.startswith(narrative_token) and segment.endswith("%s said,%s" % (identity, eos_token)):
            return True
        if segment.startswith(narrative_token) and segment.endswith("%s replied,%s" % (identity, eos_token)):
            return True
    return False