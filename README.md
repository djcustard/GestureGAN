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

To prove that the numpy arrays can be played back into sound, I tried to adapt Daisukelab's code that he used to generate the dataset. In this program he generates a 2D numpy array, but then transforms this to 3D when converting the STFT representation from Mono to Colour. Potentially this could be benefinicial to the acceptance of the dataset with DCGAN. Because this Generative Adversarial Network is built for RGB images anyway, perhaps it is worth just downgrading the output arrays back to Mono so then they can be processed back into sound.

Yet to be succesful in transferring Mel-Specs back to Audio. Might take advantage of this python program that will allow me the ability to create STFT and inverse it.

Tim Sainburg's implementation of kastnerkyle's work: https://timsainburg.com/python-mel-compression-inversion.html

Because of my inability to convert the MelSpec processed dataset back to sound, I took the raw wav files, reduced their lengths to one second and then processe them through Tim Sainburg's implementation. The audio length code I wrote is called AdjustAudioClip and the implementation of kastnerkyle's code is found in the STFT program.

The resulting Dataset I now have is Numpy arrays of 2 dimensions. Now I must try to implement the Generative Adversarial Network model.

The problem with this is that the arrays are not in a shape that could fit into the image GAN. However the MelSpecs that Daisukelab produced before are perfect for this. I will contact him to ask about the possibility of converting these specs back to sound.


//THE GENERATIVE ADVERSARIAL NETWORK MODEL THAT I HAVE PREVIOUSLY EXPLORED//

The model is built upon a version of DC-GAN outlined by lecturer Jeff Heaton.

DC-GAN with Images: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_07_2_Keras_gan.ipynb

This application uses images which are represented as 3 dimensional numpy arrays. To tackle this we must adapt this model to accept 1 dimensional arrays that are outputted from our STFT.

Currently following a 1D function implementation GAN. Keras model Sequential first layer must take it's input shape. Use Dense function to define this.

How to Develop A 1D Function Generative Adversarial Network: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/

Following on from this, I am trying to develop a short term version of this system to produce waveforms. Replacing the 1D function with a waveform. This work has seen me starting to train a discriminator model. This taking various points from the real waveform and a randomly generated waveform (noise.) 


Because of my inability to be able to succesfully regenerate audio from the STFT images I recieve from Daisukelab's code, I returned to the GANSynth code. From reading their papers before, I understand that they are fully able to perform Audio->Spec-> Audio tasks. Inside their Python program, Specgrams_Helper, they define function's named "waves_to_specgrams" and "specgrams_to_waves". I will look into these methods soon.
