# RealFaceGAN: Real-Time Adversarial Image Editing
## Introduction
RealFaceGAN is our project developed within the scope of the CS585 - Deep Generative Networks course at the Computer Engineering Department, Bilkent University. Leveraging the power of Generative Adversarial Networks (GANs) and other machine learning techniques, RealFaceGAN offers capabilities in image enhancement, object removal, and online makeup application. This project finds its applications in entertainment, digital art, and social media, among others. The advancement in deep-learning models, particularly in generating realistic images, forms the core of the RealFaceGAN model.

## Features
Object Manipulation: 
Add or remove objects from images with ease.

Appearance Modification: 
Alter the appearance of subjects in images to fit the user's requirements.

Image Segmentation and Recognition: 
Utilizes image segmentation, facial recognition, and image synthesis technologies.

Real-Time Editing: 
Empowers users to edit images in real-time through a user-friendly interface.

Natural Language Queries: 
Incorporates natural language processing to understand and execute user commands for image editing.


## Related Work
RealFaceGAN builds upon the foundations set by several key studies and projects in the field:

StyleGAN: 
Utilizes pre-trained StyleGAN networks, expanding the latent space to achieve a balance between low distortion and high editability.

TransStyleGAN: 
Integrates attention-based transformers with StyleGAN, enhancing the reconstruction and editing quality through the introduction of the W++ space.

InterFaceGAN: 
Leverages the latent semantics interpretation provided by GANs for semantic face editing.

Dataset
The CelebA Dataset, containing over 200,000 face images of various celebrities with 40 binary attribute annotations per image, is used for training and testing purposes.

## Proposed Method
RealFaceGAN introduces a novel approach by:

Learning an encoder model to find latent vectors of real images.
Utilizing the InterFaceGAN framework for detailed editing capabilities.
Generating an edited version of a real image using StyleGAN's generator.
Introducing a natural language query to latent vector editing function, enhancing accessibility and user experience.

Developed by Mousa Farshkar Azari and Gün Kaynar at Bilkent University for the CS585 - Deep Generative Networks course.
