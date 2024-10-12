Papers for nlp practitioner 📖

### `1. Foundational Papers (Year:1990s-2000s):`

* (1997) [Long Short-Term Memory (LSTM)](https://bioinf.jku.at/publications/older/2604.pdf) :bulb:
  - This paper introduces LSTMs, a crucial building block for modern NLP models. Understanding LSTMs is essential for many later papers.
* (2000) [Maximum Entropy Markov Models for Information Extraction and Segmentation (MEMMs)](https://www.seas.upenn.edu/~strctlrn/bib/PDF/memm-icml2000.pdf) :bulb:
  - This paper introduces MEMMs, a common machine learning technique used in traditional NLP tasks like information extraction.
* (2000) [A Statistical Part-of-Speech Tagger](https://arxiv.org/pdf/cs/0003055.pdf)
  - This paper demonstrates a powerful HMM-based POS tagger, introducing many core concepts,tips and tricks for building such classical systems.
* (2003) [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) :bulb:
  - This paper is a foundational work in deep learning for NLP, introducing an early effective neural network-based language model.
* (2003) [Feature-rich part-of-speech tagging with a cyclic dependency network](https://nlp.stanford.edu/pubs/tagging.pdf)
  - This paper proposes powerful linguistic features for building a high-performing POS-tagging system some number of powerful linguistic features for building a (then) SOTA POS-tagging system.
* (2004) [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013) :vhs:
  - This paper introduces ROUGE, a widely used evaluation metric for summarization tasks(ROUGE is used to this day on a variety of sequence transduction tasks).
* (2010) [From Frequency to Meaning: Vector Space Models of Semantics](https://arxiv.org/pdf/1003.1141.pdf)
  - This paper provides a great survey of existing vector space models for learning word meaning from text. A wonderful survey of existing vector space models for learning semantics in text. 
* (2012) [An Introduction to Conditional Random Fields (CRFs)](http://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)
  - This paper offers an in-depth overview of CRFs, a common sequence labeling model used in NLP tasks like POS tagging and named entity recognition. 

<hr>

### `2. Core NLP Techniques (Year:2000s-2010s):`
* `Part-of-Speech Tagging`: Papers from 2000, 2003, and 2015 explore different approaches to POS tagging, showcasing the evolution from HMMs to neural networks.
* `Parsing`: Papers from 2003, 2006, 2014 (dependency parsing), 2014 (constituency parsing), and 2015 showcase different parsing techniques, including transition-based dependency parsing and the rise of neural parsers.
* `Named Entity Recognition (NER)`: Papers from 2005 and 2015 explore NER techniques, including the use of CRFs and Bidirectional LSTMs.
* `Coreference Resolution`: Papers from 2010, 2015, and 2016 demonstrate coreference resolution approaches, including sieve-based methods and early deep learning applications.
* `Sentiment Analysis`: Papers from 2012, 2013 (introducing Stanford Sentiment Treebank), and 2014 showcase sentiment analysis techniques, including Naive Bayes models and Recursive Neural Tensor Networks.
* `Natural Logic/Inference`: Papers from 2007, 2008, and 2014 explore logic-based approaches to textual inference and the application of deep learning techniques like recurrent neural networks.
* `Machine Translation (MT)`: Papers from 1993 (introducing IBM machine translation models), 2002 (introducing BLEU evaluation metric), 2003 (phrase-based MT), 2014 (introducing sequence-to-sequence learning for MT), 2015 (attention mechanisms in MT), and 2016 (various advancements in neural MT) provide a historical perspective and showcase the evolution of MT techniques.
* `Semantic Parsing`: Papers from 2013 (learning from question-answer pairs), 2014 (paraphrase models), and 2015 (building parsers from scratch) explore semantic parsing techniques.

<hr>

### `3. Deep Learning Advancements (Year:2010s-2020s):`

* `2014: Sequence to Sequence Learning with Neural Networks`: This foundational paper introduces the sequence-to-sequence architecture, a cornerstone of modern NLP. It's particularly relevant for tasks like machine translation and text summarization.
* `2014: A Large Annotated Corpus for Learning Natural Language Inference`: This paper introduces the Stanford Natural Language Inference corpus, a crucial resource for training and evaluating models on textual inference tasks.
* `2015: Bidirectional LSTM-CRF Models for Sequence Tagging`: This paper showcases the effectiveness of combining LSTMs with CRFs for sequence labeling tasks like POS tagging and NER.
* `2016: How NOT To Evaluate Your Dialogue System`: This paper highlights the importance of careful evaluation in dialogue systems, emphasizing the limitations of unsupervised metrics.
* `2017: A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue`: This paper demonstrates the effectiveness of copy mechanisms in sequence-to-sequence models for task-oriented dialogue.
* `2018: Deep contextualized word representations (ELMO)`: This paper introduces ELMO, a powerful pre-trained language model that provides contextualized word embeddings, significantly improving performance on various NLP tasks.
* `2018: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`: BERT revolutionized NLP by introducing a pre-trained language model that can be fine-tuned for various tasks.
* `2019: XLNet: Generalized Autoregressive Pretraining for Language Understanding`: XLNet further advances pre-trained language models, addressing limitations of BERT.
* `2019: Unsupervised Data Augmentation for Consistency Training`: This paper introduces a technique for data augmentation that can be helpful when training models on limited data.
* `2020: A Generative Model for Joint Natural Language Understanding and Generation`: This paper presents a generative model that can perform both NLU and NLG tasks jointly.
* `2020: BART`: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension: BART is another powerful pre-trained language model that has been applied to various NLP tasks.   
* `2020: DeFormer`: Decomposing Pre-trained Transformers for Faster Question Answering: This paper explores techniques to improve the efficiency of pre-trained transformers for question answering tasks.