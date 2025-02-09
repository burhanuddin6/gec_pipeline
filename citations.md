# Papers

- [1](A Low-Resource Approach to the Grammatical Error Correction of Ukrainian.pdf)

- [2](A Multilayer Convolutional Encoder-Decoder Neural Network for Grammatical Error Correction.pdf)

- [4] (A Tagged Corpus and a Tagger for Urdu.pdf)

- [5] (Attention Is All You Need.pdf)

- [6] (C16-ERRANT- Alignment.pdf)

- [7] (Corpora Generation for Grammatical Error Correction.pdf)

- [8] (ERRANT - Automatic Annotation and Evaluation of Error Types for Grammatical Error Correction.pdf)

- [9] (GenERRate - Generating Errors for Use in Grammatical Error Detection.pdf)

- [10] (Generating Inflectional Errors for Grammatical Error Correction in Hindi.pdf)

- [12] (Ground Truth for Grammatical Error Correction Metrics.pdf)

- [13] (Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data.pdf)

- [15] (Proposed Model for Arabic Grammar Error Correction Based on CNN.pdf)

- [16] (Russian - Grammar Error Correction in Morphologically Rich Languages.pdf)

- [17] (Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction.pdf)

- [18] (Tensor2Tensor for Neural Machine Translation.pdf)

- [19] (Using Wikipedia Edits in Low Resource Grammatical Error Correction.pdf)

- [20] (WikiAtomicEdits - A Multilingual Corpus of Wikipedia Edits.pdf)

- [21] mT5 - AMassively Multilingual Pre-trained Text-to-Text Transformer.pdf

- [22] (A Comparative Evaluation of Deep and Shallow Approaches to the Automatic Detection of Common Grammatical Errors)

- [23] (A Discriminative Language Model with Pseudo-Negative Samples)
# Similarities

*   The need to address the challenges of low-resource languages. [10] [16] [19] [1]
*   The goal of developing GEC systems that generalize to human errors. [10] [19] [1] [13]
*   The importance of understanding error patterns and distributions. [10] [1] [16] [15]
*   The use of data augmentation techniques through synthetic data generation. [10] [1] [7] [17]
*   The use of Wikipedia edits as a source of natural errors. [10]  [19]


Our research on Urdu Grammatical Error Correction (GEC) employs a synthetic data generation approach, which involves creating a dataset of correct and incorrect sentence pairs to train a model, given the limited resources for Urdu. This approach has similarities with other research efforts in GEC, particularly for low-resource languages. Here's a comparison of each aspect of the our research with other papers:

*   **Synthetic Data Generation:**

    *   **Our Research:** Generates a synthetic dataset by learning error types from the Urdu WikiEdits dataset and inflicting these errors on a large corpus of Urdu sentences.
    *   **Similar Approaches:**
        *   **[10]** generate a parallel corpus of synthetic errors by inserting errors into grammatically correct sentences using a rule-based process, focusing specifically on inflectional errors for Hindi. Their work also scrapes Wikipedia edit history as a real-world data source to test the trained model. This is similar to the Urdu GEC research, which also uses WikiEdits to learn error types.
        *   **[7]** provide a general approach to generate large corpora, using Wiki Edits and round-trip machine translation. They also augment data with common errors found in the Wiki edit approach and introduce them using stochastic methods.
        *   **[9]**: In this paper, artificial ungrammatical data has been used in NLP and we survey its use in the field, focussing mainly on grammatical error detection.
        *   **[23]** and **[22]** attempt to learn a model which discriminates between grammatical and ungrammatical sentences, and both use synthetic negative data which is obtained by distorting sentences from the British National Corpus (BNC) (Burnard, 2000).

    
*   **Error Learning from WikiEdits:**

    *   **Our Research:** Extracts correct and incorrect sentence pairs from the Urdu WikiEdits dataset to learn different types of errors.
    *   **Similar Approaches:**
        *   **[7]** use the Wiki edits dataset, which benefits from a more accurate distribution of natural grammatical errors made by humans.
        *   **[19]**: Wikipedia edits are extracted using Wiki Edits (Grundkiewicz and Junczys-Dowmunt, 2014), profiled with ERRANT, and filtered with reference to the gold GEC data.
        *    **[20]**: WikiAtomicEdits is a dataset of atomic edits extracted from Wikipedia edit history, which can be used to generate synthetic data for GEC tasks. Its not available for Urdu.
    
*   **Error Annotation Pipeline**
    *   **Annotation Substitution, Insertion, and Deletion:** specific steps for these processes, focusing on POS tags and word features to create errors, as outlined in approach.txt
    *   **Similar Approaches:**
        *   **[22]** introduce grammatical errors of the following four types into BNC sentences: context-sensitive spelling errors, agreement errors, errors in-volving a missing word and errors involving an extra word.
        *   **[6]**: ERRANT is a tool for automatically annotating parallel data with error types. We use the alignment algorithm from it to align the sentences in the synthetic dataset for urdu and classify the edits as Insertion, Deletion, Substitution.

*   **Error Infliction:**

    *   **Our Research:** Uses a corpus of Urdu sentences to inflict learned errors and generate synthetic error-generated pairs.
    *   **Similar Approaches:**
        *   **[7]** introduce spelling errors probabilistically in the source sequences at a rate of 0.003 per character, randomly selecting deletion, insertion, replacement, or transposition of adjacent characters for each introduced error.
        *   **[10]**: We create a parallel corpus of synthetic errors by inserting errors into grammatically correct sentences using a rule­based process, focus­ing specifically on inflectional errors.

*   **Transformer Model and Training:**

    *   **Our Research:** Employs the mT0 transformer model, a fine-tuned version of mT5, designed to follow human instructions across languages in a zero-shot setting.
    *   **Similar Approaches:**
        *   **[21]**: mT5 inherits all of the benefits of T5 (described in section 2), such as its general-purpose text-to-text format, its design based on insights from a large-scale em-pirical study, and its scale. To train mT5, we in-troduce a multilingual variant of the C4 dataset called mC4.
        *   **Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data**: Need to check this

*   **Limitations (Synthetic Data Bias):**

    *   **Our Research:** Recognizes that the synthetic dataset might not capture all nuances and error patterns seen in real-world text.
    *   **Similar Approaches:**
        *   **[22]**: Wagner et al. (2009) report a drop in accuracy for their classification methods when ap-plied to real learner texts as opposed to held-out syn-thetic test data, reinforcing the earlier point that ar-tificial errors need to be tailored for the task at hand.

*   **Use of MT5:**
    We are using mt5 to leverage its multilingual capabilities and zero-shot learning.
    The use of transformer models like mT5 is consistent with current state-of-the-art practices in NLP.


*   **Similar Languages:**
    Our literature review includes research on other morphologically rich languages like Hindi and Russian, which have similar challenges to Urdu in terms of GEC.