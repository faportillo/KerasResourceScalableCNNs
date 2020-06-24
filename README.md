# KerasResourceScalableCNNs
Code used for my Master's thesis at the University of California Davis.
Using the tf.Keras library, this project creates and runs Channel-scaled resource scalable models. Link to full thesis will be published upon approval by the university. 

**Abstract:**
State-of-the-art image recognition systems use sophisticated Convolutional Neural Networks (CNNs) that are designed and trained to identify numerous object classes. Inference using such complex networks is fairly resource intensive, prohibiting their deployment on resource-constrained edge devices. In this context, we make two observations: First, the ability to classify an exhaustive list of categories is excessive for the demands of most IoT applications. Furthermore, designing a new custom-designed CNN architecture for each new IoT application is impractical, due to the inherent difficulty in developing competitive models and time-to-market pressure. The observations motivate us to consider if one can utilize an existing structurally-optimized CNN model to automatically construct a competitive CNN for a given IoT application whose objects of interest are a fraction of categories that the original CNN was designed to classify, such that the model’s inference resource requirement is proportionally scaled down. We use the term *resource scalability* to refer to this concept, and develop a methodology for automated synthesis of resource scalable CNNs from an optimized baseline CNN. The synthesized CNN has sufficient learning capacity for handling the given IoT application requirements, and yields competitive accuracy. The proposed approach is fast, and unlike the presently common practice of CNN design, does not require iterative rounds of training trial and error. Experimental results showcase the efficacy of the approach, and highlight its complementary nature with respect to existing model compression techniques.

---
**Methodology**:

The basic concept is to take an existing neural network model, Ψ,  and reduce the number of parameters in that model to produce, Ψ'

Introduces the concept of "Micro-Layer" and Macro-Layer".

 - **Micro-Layer**:  a convolutional unit that includes only one ﬁlter-bank (i.e., one set of kernels). Removing a micro-layer does not necessarily bisect the network into two separate parts. Micro-layers can be composed in parallel, and they cannot include any sub-layers.

 - **Macro-Layer**: a CNN layer that includes one or more micro-layers. Removing each macro-layer bisects the CNN into two separate parts. By deﬁnition, a CNN cannot have parallel macro-layers, since such a structure would simply be a larger macro-layer. 
 ![enter image description here](https://drive.google.com/uc?export=view&id=1PSP18p8We8MM9XDLAVOUKPKr1qDeduPN
)

Reduces size of model to fit a desired parameter count by scaling the depth of each convolution layer in an organized manner subject to two constraints.

 - Bottleneck Avoidance: In general, as we move from shallower macro-layers towards deeper ones, the number of channels in the consecutive macro-layers increases. We want to preserve this quality while changing the depth of micro-layers. 
 
 - Aﬃne Scaling: In deriving Ψ' from Ψ, we would like to preserve the contribution ratio of each micro-layer to the feature maps generated in the corresponding macro-layer.

Uses what we call "Scope-Aware Inference" to also determine if an input image belongs to the set of images the model was trained to classify (i.e. identify items as miscellaneous)

**Process**:
![enter image description here](https://drive.google.com/uc?export=view&id=19KdP9hHF03PK_VYojeqh_KV7K3C_-RFR
)

---

**Implementation and Results**:
This methodology was implemented on GoogLeNet and MobileNet. The resulting table shows a comparison with Ψ and Ψ' for multiple numbers of classes.

 - [GoogLeNet](https://arxiv.org/abs/1409.4842)
![enter image description here](https://drive.google.com/uc?export=view&id=1W4XVg4CgJp25sZ3hwreJ_D8Xu0GVCuaX
)
 


 - [MobileNet](https://arxiv.org/abs/1704.04861)
![enter image description here](https://drive.google.com/uc?export=view&id=1YVyfnma7K-EpMxcjIZBGbxkA2Lysx8jf
)


Results: Baseline (Unscaled models vs Depth-Scaled Models)

![enter image description here](https://drive.google.com/uc?export=view&id=1AysGWr_PEfmHL5Qtmkop9Yj1nt7XDl1s )

---
**Integration/Comparison with existing model compression methods (Pruning and Quantization):**

The results scaled models are then further reduced using the following process:

![enter image description here](https://drive.google.com/uc?export=view&id=1-xIBUvw_WRX6xOtEhce-1KIn7hyLeUDi
)

The results are as follows:

![enter image description here](https://drive.google.com/uc?export=view&id=1RLsyv2D_L3mlSwv8oLxmtDGP-RKP_j-I
)

