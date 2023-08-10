### Decoder models
[![Decoder models]
(https://img.youtube.com/vi/d_ixlCubqQw/0.jpg)]
(https://youtu.be/d_ixlCubqQw)

**Def**:Decoder models use only the decoder of a Transformer model. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. These models are often called **auto-regressive models**.

- The pretraining of decoder models usually revolves around predicting the next word in the sentence.
- These models are best suited for tasks involving text generation.
- Representatives of this family of models include:
    - GPT2
    - CRTL
    - GPT
    - TranformerXL
  