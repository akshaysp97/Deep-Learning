# Spoken Language Identification
In this project, we deal with recognizing the language of a given speech utterance. The data consists of auido files in three laguages mainly English, Hindi and Mandarin. 
We approach the problem by,
- Extracting Mel-frequency cepstral coeficients (MFCCs) from audio, which will be employed as features.
- Handling short periods of silence by implemeting a custom loss function to mask the silence.
- Implementing a GRU model, and train it to classify the languages.
