from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, abort, send_from_directory
from flask_restful import Resource, Api, reqparse
import argparse
import uuid
import numpy as np
import torch

from interact import generate
from identities import Identities

parser = argparse.ArgumentParser("Run the interaction server")
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
parser.add_argument("--speaker-tracking", action="store_true", required=False,
                    help="Enable speaker tracking through narrative prompts.")
parser.add_argument("--num-beams", type=int, default=6, required=False,
                    help="Number of beams to use for beam search generation.")
parser.add_argument("--show-beams", action="store_true", required=False, 
                    help="Print all beams when using beam search generation.")
parser.add_argument("--port", "-p", default="8080", required=False, type=int, help="Port to run server on.")

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
narrative_token = tokenizer.additional_special_tokens[0]
sessions = {}

app = Flask(__name__)
api = Api(app)

class Session(Resource):
    def post(self):
        session_id = uuid.uuid4().hex
        sessions[session_id] = []
        return session_id


class Interaction(Resource):
    def __init__(self):
        self.reqparser = reqparse.RequestParser()
        self.reqparser.add_argument("user_input", type=str, location="json", required=True)
        self.reqparser.add_argument("session_id", type=str, location="json",  required=True)
        #batch_reqparser.add_argument("max_len", type=int, default=60, required=False)
        #batch_reqparser.add_argument("num_beams", type=int, default=4, required=False)
        #batch_reqparser.add_argument("temperature", type=float, default=1.0, required=False)

    def post(self):
        reqargs = self.reqparser.parse_args()
        user_input = reqargs["user_input"]
        session_id = reqargs["session_id"]
        #max_len = reqargs["max_len"]
        #num_beams = reqargs["num_beams"]
        #temperature = reqargs["temperature"]

        if session_id not in sessions:
            abort(404)
            return

        dialog_history = sessions[session_id]

        responses = []
        if bool(np.random.binomial(1, args.prompt_narrative_prob)):
            results = generate(args, model, device, tokenizer, dialog_history, identities, user_input, prompt_narrative=True)
        else:    
            results = generate(args, model, device, tokenizer, dialog_history, identities, user_input)
        responses.extend(results)

        #If a narrative is generated, generate a follow-up dialog response.
        if dialog_history[-1].startswith(narrative_token):
            results = generate(args, model, device, tokenizer, dialog_history, identities, prompt_dialog=True)
            responses.extend(results)
        return responses

class UI(Resource):
    def get(self):
        return send_from_directory(".", "chat_ui.html")

api.add_resource(Session, "/session")     
api.add_resource(Interaction, "/interaction")
api.add_resource(UI, "/chat_ui/")
app.run(debug=False, port=args.port, host="0.0.0.0")