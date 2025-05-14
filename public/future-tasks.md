# Overview
This document contains tasks which are interesting but beyond the scope of a one-man mission.

[] finding the PDG attacked image noise equivilent / add random noise benchmark for the pgd attack / isote the mask / translate this into params of gaussian / equivilent gaussian noise etc.
[] variants of cmp - alternative ways of making descrete choices
[] use activation maximisation to understand the network
[] implement https://captum.ai/ for saliency map kind stuff; maybe it will tell some information loss thing
[] Andriy: Did you use the same batch size for the different learning rates? Please do a control where you change the learning rate AND the batch size so that their ratio is fixed (the same) in all cases.
[] test the convex assumption on the old network; convexity; add loss surface viz; use Hessian Matrix; For a function to be convex, the Hessian should be positive semi-definite everywhere. If the Hessian has negative eigenvalues, then there are directions in which the function is concave.
[] add deep dream; increase the loss, get the model away from the optimal point and see what happens
[] implement decoders to measure the loss of imformation along forward pass
[] egen value analysis / decision surface like that in the Linnea's report to see the landscape of intermediate representations; andriy has sent some paper about this
[] test whether making the decision binary with regulation itself make the model more robust to adversarial attack? when all the other tests are done
[] multiple receptive fields adversarial attacks
[] distribution favoring loss term, rather than simple cut; already doing this to some extent
[] would adding noise benefit the flex layer
[] add VAE -> make each image some distribution attributes; we then decode the logits (decisisions) as if they are images
[] think of alternative ways to make the model generate more concrete decisions; maybe something about embeddings; DropConnect or variational dropout
[] Test with both tensor being conv in the selection
[] later - maybe add a linear layer at the end for making the bimodal distributions (after first trail)
[] just as a vague thought - think of doing something with the activation functions; two different activations functions; or a new activation function; think about this; maybe speak to andriy
[] Andriy: HOWEVER, the most classical phenomenon associated with repeated stimulus presentation is adaptation (due to short-term depression). In this scenario, OR-like neurons will tend to become AND-like, whereas AND-like neurons will remain AND-like. The functional “purpose” of this phenomenon (adaptation) is not very clear. There are various theories. It would be indeed interesting to examine what would happen if only the OR neurons are flexible. Moreover, biological similarity aside, if this happens to give a better-performing network, then this is a win. Please try it.
[] investigate into the informaiton loss of the network across layers using mutual information
[] experiment with skip connections thus smoother minimiser
[] experiment: with random initialised binariness and conv ratio
[] experiment: insert the flex box at different position and observe the change of the hessian and plot the color / maybe by block / maybe replace not blocks but certain neurons set and see the flex
[] experiment: here maybe add the binariness shifting behaviour where the number of layer is controlled etc
[] experiment: how well the model perform vs vgg using same number or parameters
[] experiment: control the layer to plot in the gradient plot ws well as thr number of parameters
[] code: since hard sigmoid work much better and the model has preference over dichotomy at differetn places / maybe make the sharpness of the sigmoid function learned? / maybe the model should learn the sharpness of each scaling factor
[] code the init with diff proportion of conv pool ratio etc / only for the threshold mechanism
[] code: test adding the layer / ratio using the learned ratio / binariness / to show that it actually increases the performance etc
[] in a flexible neuron like this, can the loss landscape be possibly smooth?  under what condition may this happen? / think about it / maybe it happens when the model is initialised with the chosen numeber of operations at each layer / maybe bruteforce this using evolutionary algorithms ?
[] is flex a form of regulation: flex data augmentation less benefits articulated better

# Deep Interpolations
[] get data from: https://aws.amazon.com/marketplace/pp/prodview-r3vtavkhdgjli#overview
[] test deep interpolation using MONAI
[] think of How human vision cope with false positives and maybe take some principles to here