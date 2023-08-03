## A bit of Transformer history
Here are some reference points in the (short) history of Transformer models:

![history](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono.svg)


The [Transformer architecture](https://arxiv.org/abs/1706.03762) was introduced in June 2017. The focus of the original research was on translation tasks. This was followed by the introduction of several influential models, including:

* **June 2018:** [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), the first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results

* **October 2018:** [BERT](), another large pretrained model, this one designed to produce better summaries of sentences (more on this in the next chapter!)

* **February 2019:** [GPT-2](), an improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns

* **October 2019:** [DistilBERT](), a distilled version of BERT that is 60% faster, 40% lighter in memory, and still retains 97% of BERT’s performance



*  **May 2020, GPT-3**, an even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning)

they can be grouped into three categories:

* GPT-like (also called *auto-regressive* Transformer models)
* BERT-like (also called *auto-encoding* Transformer models)
* BART/T5-like (also called *sequence-to-sequence* Transformer models)


### Transformers are language models
All the Transformer models mentioned above (GPT, BERT, BART, T5, etc.) have been trained as language models. This means they have been trained on large amounts of raw text in a **self-supervised** fashion. **Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model**. That means that humans are not needed to label the data!

This type of model develops a statistical understanding of the language it has been trained on, but it’s not very useful for specific practical tasks. Because of this, the general pretrained model then goes through a process called transfer learning. During this process, the model is fine-tuned in a **supervised way** — that is, using human-annotated labels — on a given task.

An **example** of a task is predicting the next word in a sentence having read the n previous words. This is called causal language modeling because the output depends on the past and present inputs, but not the future ones.

![from huggingface](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/causal_modeling.svg)
-------
**masked language modeling:**  which the model predicts a masked word in the sentence.


this image from hugging face:
![from huggingface](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/masked_modeling.svg)


### Transformers are big models
the general strategy to achieve better performance is by increasing the models’ sizes as well as the amount of data they are pretrained on.
![from huggingface](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/model_parameters.png)





### Transfer Learning
[![Transfer Learning]
(https://img.youtube.com/vi/BqqfQnyjmgg/0.jpg)]
(https://youtu.be/BqqfQnyjmgg)

   -  **Pretraining:** is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.
![pretraining](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining.svg)
   - **Fine-tuning:** is the training done after a model has been pretrained. To perform fine-tuning, you first acquire a pretrained language model, then perform additional training with a dataset specific to your task.<br/>
**For example**, one could leverage a pretrained model trained on the English language and then fine-tune it on an arXiv corpus, resulting in a science/research-based model. The fine-tuning will only require a limited amount of data: the knowledge the pretrained model has acquired is “transferred,” hence the term transfer learning.
![fine-tune](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/finetuning.svg)

### General architecture



The model is primarily composed of two blocks:
- **Encoder (left):**The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
- **Decoder (right):** The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

![encoder-decoder](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks.svg)
Each of these parts can be used independently, depending on the task:

- **Encoder-only models:** Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
- **Decoder-only models:** Good for generative tasks such as text generation.
- **Encoder-decoder models** or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.


## Attention layers

## Summary

Transformer models have a crucial feature known as attention layers. These layers allow the model to focus on specific words in a sentence, ignoring others, which is essential for tasks involving natural language understanding.

The "Attention Is All You Need" paper introduced the Transformer architecture, emphasizing the significance of attention layers. At its core, the attention layer enables the model to understand the context and meaning of each word by paying special attention to neighboring words in a sentence.

For example, when translating text from English to French, the model needs to consider adjacent words to properly translate verbs or determine the gender of nouns. This context-driven approach becomes increasingly important as the sentence's complexity and grammar rules grow.

In conclusion, attention layers are vital for effective natural language processing, as they allow the model to grasp the meaning of words within their context and improve the performance of various language-related tasks.
### The original architecture
```markdown
+---------------------------------------------+
|               Encoder (left)                |
+---------------------------------------------+
                  ↑
            Self-Attention Layer
                  ↑
               Feed-Forward
                  ↑
            Layer Normalization
                  ↑
                 Add
                  ↑
+---------------------------------------------+
|              Decoder (right)                |
+---------------------------------------------+
                  ↑
    Self-Attention (Masked) Layer
                  ↑
     Cross-Attention Layer (Encoder-Decoder)
                  ↑
               Feed-Forward
                  ↑
            Layer Normalization
                  ↑
                 Add
                  ↑
+---------------------------------------------+
|                 Output                     |
+---------------------------------------------+
```

![arch](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers.svg)


In this simplified representation:

- The Transformer consists of an Encoder on the left and a Decoder on the right.
- The Encoder processes the input sequence, while the Decoder generates the output sequence.
Both the Encoder and Decoder contain multiple layers, each consisting of a self-attention mechanism, feed-forward neural network, and layer normalization.
- The self-attention layer in the Decoder uses masking to prevent attending to future positions during training (to maintain auto-regression).
The Decoder also has an additional cross-attention layer that attends to the Encoder's output, allowing it to leverage information from the input sequence.
- Keep in mind that the actual architecture is more complex, and the diagram above provides a high-level overview of the key components of the Transformer model. To get a more detailed visual representation, you can search for Transformer architecture diagrams online.

[![The Transformer architecture]
(https://img.youtube.com/vi/H39Z_720T5s/0.jpg)]
(https://youtu.be/H39Z_720T5s)




