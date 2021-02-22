"""
Utility methods for use during training & evaluation
"""

def load_dataset(dataset_file, narrative_token, dialog_token, eos_token):
    dataset = []
    sequence = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line == "":
                if len(sequence) > 0:
                    dataset.append("".join(sequence))    
                    sequence.clear()
            else:
                prefix_token = narrative_token if line[:3] == "[N]" else dialog_token
                sequence.append(prefix_token + line[5:] + eos_token)
                
    return dataset