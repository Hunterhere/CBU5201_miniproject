#

This project is for CBU5201 coursework under BUPT & QM joint programme. We constructed a dataset that told either a true story or a false story, in a mix of Chinese and English, with a total of 100 audio pieces.  

The pipeline's a priori assumption is that the audio in the data set itself contains *enough information* to determine whether the story told by the audio is real or deceptive. Therefore, under this assumption, the model does not add additional audio transcripts as input, but only gives judgments based on audio data. In general, the pipeline uses a model based on the Transformer architecture to handle audio classification tasks, including data enhancement, data preprocessing, model training and evaluation. The original data set was expanded to **four times** after transformation, totaling **8GB** for about **10** hours, of which 80% was used as a training set and 20% as a test set, *avoiding potential data leakage risks*. The process uses a pre-trained Whisper model to extract audio **log-mel spectrogram** features and **fine-tune** the `whisper-tiny` model to classify them through a custom SpeechClassifier. **Cross entropy** is defined as a loss function in training, and **AdamW** optimizer is used to improve the parameters, and finally the model performance is evaluated with **Accuracy** and **F1 score**.

After 2 hours of fine tuning on A40 (48GB) GPU using the best training parameter combination, the accuracy of the model on the deceptive story is **0.85**, f1 score is **0.77**, the **avg accuracy** is **0.80**, and the **avg f1 score** is **0.79**. It can be concluded that the model achieves baseline performance and can distinguish story categories well.

**NOTE**  

**1. Download [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) and [dataset](https://huggingface.co/datasets/Hunterhere/CBU0521DD_stories_expanded) from Hugging Face first**  
**2. Please run `data_augment.ipynb` to regenerate full datasets**  
