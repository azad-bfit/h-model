How Language Models Work: Study Notes
1. Logit
- Odds = p / (1 − p). Logit = ln(odds) = ln(p / (1 − p)).
- It maps a probability (0 to 1) onto any real number: 0.5 → 0, 0.8 → +1.39, 0.2 → −1.39, 0.9 → +2.20.
- Inverse (sigmoid): p = 1 / (1 + e^(−z)).
- Worked example: p = 0.8 → odds = 0.8 ÷ 0.2 = 4 → ln(4) ≈ 1.39.
- In logistic regression, logit(p) is a linear function of the inputs. In neural networks, the raw final-layer outputs are called logits.
2. Softmax
- Turns a list of logits into probabilities that sum to 100%.
- Steps: (1) compute e^score for each class, (2) add them up, (3) divide each by the total.
- Example, scores Cat 2, Dog 1, Bird 0 → e^x = 7.39, 2.72, 1; total 11.11 → 66.5%, 24.5%, 9.0%.
- Sigmoid handles one yes/no logit. Softmax handles many classes. It exaggerates differences between scores.
3. Tokenizer
- Splits text into tokens (words or subwords) and maps each to an integer ID from a fixed vocabulary.
- Models only understand numbers, so this is the first step. The model’s output IDs are converted back to text by the same tokenizer.
- Subword tokens are the middle ground: whole words give a huge vocabulary and fail on new words, single letters give very long sequences.
- A token is roughly 3 to 4 English characters, or about ¾ of a word. Pricing and context windows are counted in tokens.
4. Tensor
- A grid of numbers: scalar (0D), vector (1D), matrix (2D), tensor (3D or more).
- Shape is the size along each dimension.
- Token IDs for a 6-token sentence form a tensor of shape (6). Logits have shape (6, vocab_size). A batch of 32 sentences has shape (32, 6, vocab_size).
- GPUs are built for fast math on big tensors. Weights and gradients are tensors too.
5. Self-attention
- Lets each token look at the other tokens and decide which ones matter, such as “it” → “cat”.
- Steps:
  1. Each token vector produces a Query (what am I looking for), Key (what do I contain), and Value (what do I give).
  2. Score every pair: Query · Key.
  3. Apply softmax to the scores.
  4. Mix the Values using those weights.
- Formula: Attention = softmax(QKᵀ / √d) V. The √d keeps numbers from growing too large.
- Multi-head attention runs many attention sets in parallel, each learning different relationships.
- Causal masking: in GPT-style models, a token can only see earlier tokens.
- This is the core of the Transformer architecture.
6. Where “grammar” comes from
- There is no grammar rulebook inside the model. Grammar patterns emerge from training.
- Different heads tend to learn useful links, such as verb → subject (“run” → “dogs”), adjective → noun, pronoun → referent.
- The model is never told what a noun or verb is. These patterns are strengthened because they help predict the next token.
- Caveat: many heads do mixed or hard-to-interpret jobs, so neat labels are a simplification.
7. How a model is trained
1. Start with random weights (billions of numbers in tensors).
2. Tokenize huge amounts of text.
3. Predict the next token (logits → softmax → probabilities).
4. Measure the error with cross-entropy loss, which is high if the true next word got low probability.
5. Backpropagation computes the gradient of the loss for every weight.
6. Gradient descent nudges each weight to reduce the loss, with step size set by the learning rate.
7. Repeat for billions of snippets.
   - This is pre-training, a form of self-supervised learning (no human labels needed).
   - Afterward: fine-tuning on good Q&A examples, then RLHF or similar, where ratings of responses steer the model toward better answers.
8. Backpropagation and gradient descent: worked example
- Model: prediction = w × x, with x = 2, target = 10, start w = 1.
- Prediction = 2, loss = (2 − 10)² = 64.
- Chain rule backward: d(loss)/d(prediction) = 2(2 − 10) = −16. d(prediction)/dw = x = 2. So d(loss)/dw = −32.
- Update: new w = w − lr × gradient = 1 − 0.05 × (−32) = 2.6.
- Continuing: w = 2.6, 3.56, 4.14, … → 5 (the ideal value), loss → 0, and steps shrink as the slope flattens.
- Backpropagation is how gradients are calculated. Gradient descent is how they are used to update weights.
- Too small a learning rate crawls, too large overshoots. The Adam optimizer adapts step sizes per weight.
9. Open-source repos for learning
Suggested order:
1. micrograd (github.com/karpathy/micrograd): backprop and gradient descent from scratch.
2. minbpe (github.com/karpathy/minbpe): a minimal BPE tokenizer.
3. nanoGPT (github.com/karpathy/nanoGPT): a complete GPT, with model, attention, and training loop in about 300 lines each.
4. build-nanogpt (github.com/karpathy/build-nanogpt): the same model built step by step, with a video lecture.
5. LLMs-from-scratch (github.com/rasbt/LLMs-from-scratch): beginner-friendly notebooks covering tokenizer, attention, GPT, pretraining, and fine-tuning.
6. nanochat (github.com/karpathy/nanochat): a full-stack ChatGPT-like implementation covering tokenization, pretraining, finetuning, evaluation, inference, and a web UI in one minimal codebase. deepakness
- For production-grade code, github.com/huggingface/transformers is large and abstract, so it’s harder to learn from.
10. nanoGPT model.py walkthrough
Structure: LayerNorm → CausalSelfAttention + MLP → Block → GPT.
Setting	Typical value	Meaning
block_size	1024	max context length
vocab_size	~50,000	tokens known
n_layer	12	stacked Blocks
n_head	12	heads per layer
n_embd (C)	768	vector size per token


- CausalSelfAttention: one linear layer makes Q, K, V together. Split C into heads, then scores = QKᵀ / √head_size, apply the future mask (−∞), softmax, multiply by V, merge heads, and apply an output projection.
- MLP: expand C → 4C, GELU, shrink back to C. It processes each token individually after attention has mixed information.
- Block: x = x + attn(ln(x)), then x = x + mlp(ln(x)). Residual connections give gradients a direct path back. LayerNorm keeps values stable.
- GPT: token embedding (wte) + position embedding (wpe) → N Blocks → final LayerNorm → lm_head produces logits. Cross-entropy loss is computed against targets (the input shifted by one). wte and lm_head share weights.
- Shapes (1 sentence, 6 tokens): IDs (1, 6) → embeddings (1, 6, 768) → after each Block (1, 6, 768) → logits (1, 6, 50000).
- generate(): crop to context, run the model, take the last position’s logits, divide by temperature, optionally keep top-k, softmax, sample, append, repeat.
  - Low temperature is safe and repetitive. High temperature is creative and riskier.
- In train.py: find loss.backward() (backprop) and optimizer.step() (update).
- Study tip: print x.shape after each step, and print the attention weights for a short sentence.
11. Tokenizers and languages
- Modern tokenizers can represent any text, because byte-level BPE starts from the 256 byte values, so there are no unknown words.
- But merges are learned from the tokenizer’s training text. If it was mostly English, English gets efficient tokens and other languages get split into many small pieces.
- Hindi typically needs about 2 to 4 times more tokens than English on older tokenizers, for three reasons:
  1. Less Hindi in the tokenizer’s training text.
  2. Devanagari characters take 3 bytes in UTF-8 versus 1 for English letters.
  3. Matras and conjuncts are combinations of several Unicode characters.
- Effects: higher cost (priced per token), fewer sentences in the context window, slower generation, and slightly lower quality.
- Newer tokenizers have bigger multilingual vocabularies and narrow the gap, though English is usually still cheapest.
- Test it: use tiktoken to compare “How are you?” with “आप कैसे हैं?”.
12. Training a Hindi model from scratch
- A Hindi-trained tokenizer gives common Hindi words and syllables their own tokens, so token counts become comparable to English.
- Steps:
  1. Collect a large, cleaned Hindi corpus (Hindi Wikipedia, news, books, public datasets such as CC-100, OSCAR, IndicCorp).
  2. Train a tokenizer on it (minbpe, Hugging Face tokenizers, or SentencePiece) with a 32k to 64k vocabulary.
  3. Tokenize the corpus.
  4. Set vocab_size in GPTConfig. Nothing else in model.py changes.
  5. Train with train.py.
- Handle carefully: normalize Unicode (NFC/NFKC), check that the pre-tokenization regex doesn’t cut Devanagari syllables badly, mix in some English and code, and keep byte fallback on.
- The hard part is data and compute, not the tokenizer.
  - A small model (tens of millions of parameters) on a few GB of Hindi runs on one GPU. It’s good for learning, but limited.
  - A genuinely useful model needs billions of tokens and many GPUs, and high-quality Hindi text is scarcer than English.
- Cheaper alternative: extend an existing multilingual model by adding Hindi tokens and continuing pre-training on Hindi text. This needs far less data and compute.
- Learning plan: take a few hundred MB of Hindi, train a 16k-token tokenizer, compare token counts against GPT-2’s tokenizer, then train a small nanoGPT-sized model and sample from it.