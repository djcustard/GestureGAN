# GestureGAN
### Application of AI in ElectroAcoustic Music

**It should be noted that Google Colab documents run from the Dataset on my Google Drive, a rerouting of the Dataset will be done for public access at some point!

Here is a collection of the code which contributes towards the GAN. 

Firstly, the Dataset is provided from a Kaggle competition. User Daisukelab presents a Preprocessed Mel-Spectrogram representation of audio clips, which are in the form of Numpy arrays.

link: https://www.kaggle.com/daisukelab/fat2019_prep_mels1#trn_noisy_best50s.csv

These arrays are depedent on the length of the audio clips. In our case we want these to all be the same length so I wrote some code to unload the pickle and shorten these arrays to be 1 second long. Program I wrote is provided as a file called ReshapeTheNumpy.ipynb.

Original Numpy format: (128, AUDIO_LENGTH, 3);
New Numpy format: (128, 128, 3);

I originally wanted to use AudioSet, however this dataset only provided csv files of timestamps within YouTube videos. These timestamps represented sounds of a particular condition which was ideal for this interpretation. The FAT2019 Prep-Mels Dataset will have to use a classifier engine to organise these sounds. At the moment I plan to break these down into folder of their particular categories and then train the GAN on each. At this current time I am unsure how this will work with the plan for interpolation between categories, but this will soon be fixed.

AudioSet: https://research.google.com/audioset/

To prove that the numpy arrays can be played back into sound, I adapted Daisukelab's code that he used to generate the dataset.

The model is built upon a version of DC-GAN outlined by lecturer Jeff Heaton.

DC-GAN with Images: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_07_2_Keras_gan.ipynb

This application uses images which are represented as 3 dimensional numpy arrays. To tackle this we must adapt this model to accept 1 dimensional arrays that are outputted from our STFT.
