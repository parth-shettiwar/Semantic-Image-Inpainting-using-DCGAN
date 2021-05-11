## Semantic Image Inpainting using DCGAN
CS736 Project:  
1)Parth Shettiwar: 170070021  
2)Prajval Nakrani: 17D070014  
3)Harekrissna Rathod, 17D070001  

This is a pytorch implementation of the paper [Semantic Image Inpainting with Deep Generative Models](https://arxiv.org/abs/1607.07539)    

The DCGAN training was done for 50 epochs using reference from [Pytorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) to get the trained weights used for training optimal latent vector.  

Following are the results from the implmentation after running for 1500 iterations in finding optimal latent space vector.  

![Results](Results/result.png)  
  
**Running Instructions:**  
1)For training: run python training.py after putting appropriate path for dataset [CelebA](https://www.kaggle.com/jessicali9530/celeba-dataset)
in the code.  
2)For post processing to perform poisson blending, give the path for input image and output image in the code and run python poisson_blending.py.  
