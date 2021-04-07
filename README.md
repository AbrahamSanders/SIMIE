# SIMIE
![SIMIE](images/simie_banner.png)

## What is SIMIE?
SIMIE is an experiment in narrative extrapolation from dialog using GPT-2.

## Motivation
Chatbots model human conversation, but don't picture the scene the way we do in our mind's eye when we talk.
Can machines learn to imagine a story behind a conversation? Books might be a great place to start!

## Narrative-aligned dialog dataset
The dataset used for training is built using a [fork of ricsinaruto/gutenberg-dialog](https://github.com/AbrahamSanders/gutenberg-dialog), which has been extended to augment dialogs with surrounding narrative passages.

For example, this is a dialog sequence extracted from `THE CASH BOY` by `Horatio Alger, Jr.` using `gutenberg-dialog`:

```
296.txt:  We’ll go upstairs,
296.txt:  This will be your room,
296.txt:  I hope you will soon feel at home here, I’ll go down and see if I can find something to eat.
296.txt:  What does this mean?
```

After enabling narrative alignment, narrative passages immediately surrounding and nested within this dialog sequence are included.
A new prefix indicates if each line is narrative (`[N]`) or dialog (`[D]`):

```
296.txt:  [N]: At the front door, instead of knocking--there was no bell--Graves drew a rusty key from his pocket and inserted it in the lock. They found themselves in a small entry, uncarpeted and dingy. 
296.txt:  [D]: We’ll go upstairs,
296.txt:  [N]: Arrived on the landing, he threw open a door, and ushered in our hero. 
296.txt:  [D]: This will be your room,
296.txt:  [N]: Frank looked around in dismay. 
296.txt:  [N]: It was a large, square room, uncarpeted, and containing only a bed, two chairs and a washstand, all of the cheapest and rudest manufacture. 
296.txt:  [D]: I hope you will soon feel at home here, I’ll go down and see if I can find something to eat.
296.txt:  [N]: He went out, locking the door behind him 
296.txt:  [D]: What does this mean?
```

The narrative passages paint a picture of the scene in which the dialog takes place.

## Build the dataset
1. Clone the [fork of ricsinaruto/gutenberg-dialog](https://github.com/AbrahamSanders/gutenberg-dialog)
2. From the clone directory, in a terminal run:
   ```
   python code/main.py --dialog_gap=200 --include_surrounding_narratives --languages=en --run_all
   ```
   If the books fail to download, try changing the default gutenberg mirror via the `GUTENBERG_MIRROR` environment variable as described [here](https://github.com/AbrahamSanders/gutenberg-dialog#1-download--d).

3. The dataset can be found in `data/filtered/en/dialogs.txt`. Additionally, the tool creates a train / validation (dev) / test split of the dialogs which can be found in the same directory, named `train.txt`, `dev.txt`, and `test.txt`.

## Chat with the pre-trained model: gpt2-xl finetuned on the dataset
1. Download the model [gpt2-xl-dialog-narrative](https://drive.google.com/file/d/1vnY9CjgZSuuZdOpCUcDc7xhKf48SSHfD/view?usp=sharing) and extract to directory `generator/models/gpt2-xl-dialog-narrative`.

2. Run `interact.py`:
   ```
   python interact.py
   ```

   Additional command line options can be used:
   
   ```
   python interact.py --help

   usage: Interact with the generator model [-h] [--modelpath MODELPATH]
                                            [--force-cpu]
                                            [--prompt-narrative-prob PROMPT_NARRATIVE_PROB]
                                            [--max-input-tokens MAX_INPUT_TOKENS]
                                            [--print-raw] [--speaker-tracking]

   optional arguments:
   -h, --help            show this help message and exit
   --modelpath MODELPATH
                           Path to the Huggingface Transformers GPT-2 model to
                           load. (default: models/gpt2-xl-dialog-narrative)
   --force-cpu           Force the device to cpu even if a supported GPU is
                           present.
   --prompt-narrative-prob PROMPT_NARRATIVE_PROB
                           Probability that the model will get prompted to
                           generate narrative at each turn. (default: 0.2)
   --max-input-tokens MAX_INPUT_TOKENS
                           Maximum number of tokens to use as input. Dialog
                           history gets trimmed from the back to accommodate
                           this. (default: 350)
   --print-raw           Print the raw model input and output for debugging
                           purposes.
   --speaker-tracking    Enable speaker tracking through narrative prompts.
   ```

   ## Explore examples generated with the pre-trained model:
   See: [examples](examples)