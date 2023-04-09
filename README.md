# CS6741_Replication
Replication Project for CS6741 - Topics in NLP @ Cornell University
TEMOS - Text-to-Motion Synthesis

The project goal is to implement the architecture proposed in the paper "https://arxiv.org/abs/2204.14109" from ECCV 2022. 

Brief summary about the paper.

The paper provides a network architecture mostly based on VAE with Transformer Encoder & Decoder to learn the joint space of text and human motion represented by their skeletal motion (sequence of skeleton pose). In this github repository, we have several major parts which describe the following
1. Data 
  - As the project handles text and human motion, it requires additional pre-processing steps to convert the original motion capture data into human model based forward kinematics. This code handles the conversion of the original human motion data into motion model (KIT-mmm) based forward kinematics data.
  
2. Model
  - The model has two different pathways to create joint space between text and motion
    1. Variational Auto Encoder (VAE) architecture for motion representation learning
      In this pipeline, transformer encoder-decoder structure was used to learn the latent space of motion model.
    2. Pre-trained language model based encoder-decoder architecture for text to motion synthesis
      In this pipeline, pre-trained language model is used to map text to their embeddings. This embeddings become the input of encoder-decoder structure where the output is the corresponding motion.
      
  - The two pathways mentioned above were trained jointly to match the latent space of the pathway 1. with the latent space of pathway 2. (Normal distribution)
  
