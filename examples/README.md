# Examples
This directory contains select examples of narrative-enabled conversations with the [gpt2-xl](https://huggingface.co/gpt2-xl) model fine-tuned on the narrative-aligned dialog dataset.

## Cherry Picks :cherries:
The [cherry_picks](cherry_picks) directory contains example dialogs which are *mostly* consistent with the context throughout, and contain
*mostly* human-like response utterances and/or narratives.
 - [footman.txt **](cherry_picks/footman.txt): A short dialog between a boy named Jack and his employer Sir George. Jack works in the fields
 but thinks he would be happier as a [footman](https://en.wikipedia.org/wiki/Footman).
 - [revolver.txt **](cherry_picks/revolver.txt): Later on, Jack attempts to shoot Sir George (see above) over an unspecified disagreement.
 - [tomatoes.txt](cherry_picks/tomatoes.txt): An unnamed narrator buys a bag of tomatoes from his grocer, but first haggles over the price
 and gets an unexpected bargain.

## Lemon Picks :lemon: (coming soon)
The [lemon_picks](lemon_picks) directory contains examples which display any of the many known failure modes common to generative language models
including but not limited to:
 - Response utterances and/or narratives which directly conflict with themselves or immediately preceding responses.
 - Topical and/or factual inconsistency with the context.
 - Unnecessary repetition.
 - Incoherent or grammatically incorrect responses.
 - Boring, generic, high-entropy responses (See: [Neural Chatbots Are Dumb - article by Richard Csaky](https://medium.com/@richardcsaky/neural-chatbots-are-dumb-65b6b40e9bd4)).

** This dialog uses third-person narrative-based speaker tracking.