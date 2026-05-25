**Applied Physics 293 Explainable AI**  
**Instructor: Surya Ganguli**  
**Stanford University** 

**General review/perspective articles**

* Review articles  
  * [Foundation models in neuroscience](https://www.neuroai.science/p/foundation-models-for-neuroscience)  
  * [A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models](https://arxiv.org/abs/2407.02646)  
    * See also [ICML 2025 Tutorial on Mechanistic Interpretability for Language Models](https://ziyu-yao-nlp-lab.github.io/ICML25-MI-Tutorial.github.io/)  
  * [Mechanistic Interpretability for AI Safety: A Review](https://arxiv.org/abs/2404.14082)  
  * [Post-hoc Interpretability for Neural NLP: A Survey](https://dl.acm.org/doi/10.1145/3546577)  
  * [The Shapley value in machine learning](https://arxiv.org/abs/2202.05594)  
  * [The Quest for the Right Mediator: Mechanistic Interpretability via Causal Mediation Analysis](https://arxiv.org/abs/2408.01416)  
  * [Counterfactual Explanations and Algorithmic Recourses for Machine Learning: A Review](https://arxiv.org/abs/2010.10596)  
  * [Towards Unified Attribution in Explainable AI, Data-Centric AI, and Mechanistic Interpretability](https://arxiv.org/abs/2501.18887)  
  * [Explaining by removing: A unified framework for model explanation](https://arxiv.org/abs/2011.14878)  
  * [Training Data Influence Analysis and Estimation: A Survey](https://arxiv.org/abs/2212.04612)  
  * [A Primer on the Inner Workings of Transformer-based Language Models](https://arxiv.org/abs/2405.00208)  
  * [The Explainability of Transformers: Current Status and Directions](https://www.mdpi.com/2073-431X/13/4/92)  
  * [Circuit analysis research landscape](https://www.neuronpedia.org/graph/info#section-the-landscape-of-interpretability-methods)  
* Perspective pieces  
  * [Position: Principles of Animal Cognition to Improve LLM Evaluations](https://openreview.net/forum?id=gCPJFcHskT)  
  * [Testing methods of neural systems understanding](https://www.sciencedirect.com/science/article/pii/S1389041723000906)  
  * [Multilevel Interpretability Of Artificial Neural Networks: Leveraging Neuroscience](https://arxiv.org/abs/2408.12664)  
  * [Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?](https://aclanthology.org/2020.acl-main.386/)  
  * [Assessing skeptical views of interpretability research](https://web.stanford.edu/~cgpotts/blog/interp/)  
* Roadmaps  
  * [How To Become A Mechanistic Interpretability Researcher](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)  
  * [Open problems in mechanistic interpretability](https://arxiv.org/abs/2501.16496)   
  * [AI Security Institute Call for Interpretability Research](https://alignmentproject.aisi.gov.uk/research-area/interpretability)  
* Paper lists  
  * [List of Explainable AI papers](https://github.com/BirkhoffG/Explainable-ML-Papers)  
  * [Awesome Interpretability in Large Language Models](https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models)  
  * [Opinionated list of mechanistic interpretability papers](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite)  
* Conferences  
  * [2nd New England mechanistic interpretability workshop](https://nemiconf.github.io/summer25/)

**Motivations: Foundation models in neuroscience: big data, big models, but understanding?**

* Task trained models in neuroscience across the years  
  * [A back-propagation programmed network that simulates posterior parietal neurons](https://www.nature.com/articles/331679a0)  
  * [What Does the Retina Know about Natural Scenes?](https://direct.mit.edu/neco/article/4/2/196/5632/What-Does-the-Retina-Know-about-Natural-Scenes)  
  * [The emergence of multiple retinal cell types through efficient coding of natural movies](https://proceedings.neurips.cc/paper/2018/hash/d94fd74dcde1aa553be72c1006578b23-Abstract.html)  
  * [Emergence of simple-cell receptive fields by learning a sparse code for natural images](https://www.nature.com/articles/381607a0)  
  * [Performance-optimized hierarchical models predict neural responses in higher visual cortex](https://www.pnas.org/doi/10.1073/pnas.1403112111)  
  * [Context-dependent computation by recurrent dynamics in prefrontal cortex](https://www.nature.com/articles/nature12742)  
* Complex models fit to neural data, including foundation models  
  * EEG  
    * [Neuro-GPT: Towards A Foundation Model for EEG](https://arxiv.org/abs/2311.03764)  
  * fMRI  
    * [Self-Supervised Learning of Brain Dynamics from Broad Neuroimaging Data](https://arxiv.org/abs/2206.11417)  
    * [BrainLM: A foundation model for brain activity recordings](https://openreview.net/forum?id=RwI7ZEfR27)  
  * Single-cell electrophysiology  
    * [Inferring single-trial neural population dynamics using sequential auto-encoders](https://www.nature.com/articles/s41592-018-0109-9)  
    * [Interpreting the retinal neural code for natural scenes: From computations to neurons](https://www.sciencedirect.com/science/article/pii/S0896627323004671)  
    * [A Unified, Scalable Framework for Neural Population Decoding](https://poyo-brain.github.io/)  
    * [Multi-session, multi-task neural decoding from distinct cell-types and brain regions](https://openreview.net/forum?id=IuU0wcO0mo)  
    * [Generalizable, real-time neural decoding with hybrid state-space models](https://arxiv.org/abs/2506.05320v1)  
    * [Representation learning for neural population activity with neural data transformers](https://arxiv.org/abs/2108.01210)  
    * [Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity](https://papers.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf)  
    * [Towards a "universal translator" for neural dynamics at single-cell, single-spike resolution](https://arxiv.org/abs/2407.14668)  
    * [Neural encoding and decoding at scale](https://openreview.net/forum?id=vOdz3zhSCj)  
    * [Foundation model of neural activity predicts response to new stimulus types](https://www.nature.com/articles/s41586-025-08829-y)  
    * [Compact deep neural network models of visual cortex](https://www.biorxiv.org/content/10.1101/2023.11.22.568315v1)  
* Basic theories of transfer learning explaining how data from other sessions/subjects/species might help  
  * [An analytic theory of generalization dynamics and transfer learning in deep linear networks](https://arxiv.org/abs/1809.10374)  
  * [Features are fate: a theory of transfer learning in high-dimensional regression](https://icml.cc/virtual/2025/poster/43897)

**Feature attribution**: How does a network output depend on input features? 

* Perturbation based approaches  
  * [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)  
  * [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)  
  * [The many Shapley values for model explanation](https://arxiv.org/abs/1908.08474)  
* Gradient based approaches  
  * [Deep inside convolutional networks: visualizing saliency maps](https://arxiv.org/abs/1312.6034)  
  * [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) (Integrated gradients)  
  * [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)  
  * [Time-series attribution maps with regularized contrastive learning](https://arxiv.org/abs/2502.12977)  
  * [TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation](https://arxiv.org/abs/2506.05035)  
* Approximation based approaches  
  * [Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938v3) (LIME)  
  * [Significance Tests for Neural Networks](https://arxiv.org/abs/1902.06021)  
* Unified view and perspectives  
  * [Which Explanation Should I Choose? A Function Approximation Perspective](https://arxiv.org/abs/2206.01254)  
  * [From Shapley Values to Generalized Additive Models and back](https://arxiv.org/abs/2209.04012)

**Data Attribution**: Which training data points support a test prediction? 

* [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)  
* [Data Shapley: Equitable Valuation of Data for Machine Learning](https://arxiv.org/abs/1904.02868)  
* [Datamodels: Predicting Predictions from Training Data](https://arxiv.org/abs/2202.00622)  
* Scaling up  
  * [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/abs/2308.03296)  
  * [TRAK: Attributing Model Behavior at Scale](https://arxiv.org/abs/2303.14186)  
  * [DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models](https://arxiv.org/abs/2310.00902)

**Discovery of Concepts**

* [Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://arxiv.org/abs/1711.11279)  
* [Towards automatic concept based explanations](https://proceedings.neurips.cc/paper/2019/hash/77d2afcb31f6493e350fca61764efb9a-Abstract.html)  
* [We Can't Understand AI Using our Existing Vocabulary](https://arxiv.org/abs/2502.07586)  
* [Neural representational geometry underlies few-shot concept learning](https://www.pnas.org/doi/10.1073/pnas.2200800119)  
* [A mathematical theory of semantic development in deep neural networks](https://www.pnas.org/doi/10.1073/pnas.1820226116)

**Introduction to Interpretability in transformers**

* Introductory material  
  * [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
  * [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)  
  * 3blue1brown videos:  
    * [Ch 5: Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M)  
    * [Ch 6: Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  
    * [Ch 7: How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=8)  
  * [Formal algorithms for transformers](https://arxiv.org/abs/2207.09238)  
* Connections to earlier and simpler ideas  
  * [Attention and kernel smoothing](http://bactra.org/notebooks/nn-attention-and-transformers.html)  
  * [Kernel regression, data adaptive filters, and attention](https://docs.google.com/presentation/d/1VuEJ2-bYgP34hBTIY_hS7ZV_jD_jdBlqpCPS6Mqp-k4/edit?slide=id.g32e350e7ee7_0_0#slide=id.g32e350e7ee7_0_0)   
* Early Interpretation of transformers (Induction Heads)  
  * [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)  
    * [One-layer transformers aren’t equivalent to a set of skip-trigrams](https://www.alignmentforum.org/posts/b5HNYh9ne5vEkX5ag/one-layer-transformers-aren-t-equivalent-to-a-set-of-skip)  
    * [Some common confusion about induction heads](https://www.lesswrong.com/posts/nJqftacoQGKurJ6fv/some-common-confusion-about-induction-heads)  
  * [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)  
  * [Induction heads illustrated](https://www.perfectlynormal.co.uk/blog-induction-heads-illustrated)  
* RASP interpretation  
  * [Thinking Like Transformers](https://arxiv.org/abs/2106.06981)  
  * [Tracr: Compiled Transformers as a Laboratory for Interpretability](https://arxiv.org/abs/2301.05062)  
* Connections to modern Hopfield model  
  * [Hopfield networks is all you need](https://arxiv.org/abs/2008.02217)  
  * [Dense associative memory for pattern recognition](https://arxiv.org/abs/1606.01164)  
  * [On a model of associative memory with huge storage capacity](https://link.springer.com/article/10.1007/s10955-017-1806-y)  
  * [Exponential capacity of dense associative memories](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.077301)  
  * [The Capacity of Modern Hopfield Networks under the Data Manifold Hypothesis](https://arxiv.org/abs/2503.09518)

**Sparse Autoencoders**

* [Toy models of superposition](https://arxiv.org/abs/2209.10652)   
* [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)  
* [Interpreting Attention Layer Outputs with Sparse Autoencoders](https://arxiv.org/abs/2406.17759)  
* [Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control](https://arxiv.org/abs/2405.08366)  
* [Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concepts](https://arxiv.org/abs/2505.16004)  
* [The Geometry of Concepts: Sparse Autoencoder Feature Structure](https://arxiv.org/abs/2410.19750)  
* [CRISP: Persistent Concept Unlearning via Sparse Autoencoders](https://arxiv.org/abs/2508.13650)  
* Scaling up  
  * [Scaling and evaluating sparse autoencoders](https://cdn.openai.com/papers/sparse-autoencoders.pdf)  
  * [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

**Causal analysis, editing and control** 

* Perturbation based approaches  
  * [Direct and Indirect Effects](https://arxiv.org/pdf/1301.2300)  
  * [Investigating gender bias in language models using causal mediation analysis](https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html)  
  * [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (Causal tracing)  
  * [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)  
  * [Neuron Shapley: Discovering the Responsible Neurons](https://arxiv.org/abs/2002.09815)  
* Gradient based approaches  
  * [Attribution patching: Activation patching at industrial scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)  
* Approximation based approaches  
  * [Decomposing and Editing Predictions by Modeling Model Computation](https://arxiv.org/abs/2404.11534) (COAR)  
* Causal abstractions  
  * [Causal abstractions of neural networks](https://arxiv.org/abs/2106.02997)  
  * [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://arxiv.org/abs/2303.02536)  
  * [An Interpretability Illusion for Subspace Activation Patching](https://arxiv.org/abs/2311.17030)   
  * [A Reply to Makelov et al. (2023)'s "Interpretability Illusion" Arguments](https://arxiv.org/abs/2401.12631)  
  * [The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](https://arxiv.org/abs/2507.08802)  
* More model editing  
  * [Editing factual knowledge in language models](https://arxiv.org/abs/1810.03292)  
  * [Fast model editing at scale](https://arxiv.org/abs/2110.11309)  
  * [Does localization inform editing? Surprising differences](https://arxiv.org/abs/2110.11309)   
* Model steering  
  * [Representation engineering: A top-down approach to AI transparency](https://arxiv.org/abs/2310.01405)  
  * [The Geometry of Truth: Emergent Linear Structure](https://arxiv.org/abs/2310.06824)  
  * [Truth is universal: Robust detection of lies in LLMs](https://arxiv.org/abs/2407.12831)  
  * [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)  
  * [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341)  
  * [Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning](https://arxiv.org/abs/2507.16795)

**Evaluation of model explanations**

* [Sanity checks for saliency maps](https://arxiv.org/abs/1810.03292)  
* [OpenXAI: Towards a Transparent Evaluation of Model Explanations](https://arxiv.org/abs/2206.11104)  
* [MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151)  
* [Towards Unifying Interpretability and Control: Evaluation via Intervention](https://arxiv.org/abs/2411.04430)  
* [Causal Scrubbing: a method for rigorously testing interpretability hypotheses](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)

**Circuit discovery** 

* [Initial Circuits Thread](https://distill.pub/2020/circuits/)  
* [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)  
* [How does GPT-2 compute greater-than?](https://arxiv.org/abs/2305.00586)  
* [Sparse Feature Circuits: Discovering/Editing Interpretable Causal Graphs in LLMs](https://arxiv.org/abs/2403.19647)  
* [Does Circuit Analysis Interpretability Scale? Multiple Choice Capabilities in Chinchilla](https://arxiv.org/abs/2307.09458)  
* [Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997)  
* [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)  
* [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)  
* [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944)  
* [Circuit Tracer](https://www.anthropic.com/research/open-source-circuit-tracing)

**Computational complexity issues in interpretability** 

* [The Computational Complexity of Circuit Discovery for Inner Interpretability](https://arxiv.org/abs/2410.08025)  
* [Local vs. Global Interpretability: A Computational Complexity Perspective](https://arxiv.org/abs/2406.02981)  
* [Model interpretability through the lens of computational complexity](https://arxiv.org/abs/2406.02981)

**Comparing representations across models** 

* [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)  
* [Linearly Mapping from Image to Text Space](https://arxiv.org/abs/2209.15162)  
* [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)

**Discovering and understanding interesting behaviors**

* Behavior discovery through “psychology” experiments on LLMs  
  * [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (In-context learning)  
  * [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) (chain-of-thought unfaithfulness)  
  * [Taken out of context: On measuring situational awareness in LLMs](https://arxiv.org/abs/2309.00667)  
  * [Alignment faking in large language models](https://arxiv.org/abs/2412.14093)  
  * [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424)  
  * [Introducing Docent: A system for analyzing and intervening on agent behavior](https://transluce.org/introducing-docent)  
* Understanding specific, interesting behaviors  
  * [On the Emergence of Linear Analogies in Word Embeddings](https://arxiv.org/pdf/2505.18651)  
  * [Language Models use Lookbacks to Track Beliefs](https://arxiv.org/abs/2505.14685)  
  * [Language Models Share Latent Grammatical Concepts Across Diverse Languages](https://arxiv.org/abs/2501.06346)  
  * [Incremental Sentence Processing Mechanisms in Autoregressive Language Models](https://arxiv.org/abs/2412.05353)  
  * [Emergent World Representations: Exploring a Sequence Model on a Synthetic Task](https://arxiv.org/abs/2210.13382)  
  * [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217)  
  * [Acquisition of chess knowledge in AlphaZero](https://www.pnas.org/doi/10.1073/pnas.2206625119)

**Cautionary tales in explainabilty**

* [Impossibility theorems for feature attribution](https://www.pnas.org/doi/10.1073/pnas.2304406120)  
* [The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective](https://arxiv.org/abs/2202.01602)  
* [Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models](https://arxiv.org/abs/2402.04614)  
* Adversarial attacks on Interpretations  
  * [Interpretation of Neural Networks is Fragile](https://arxiv.org/abs/1710.10547)  
  * [Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods](https://arxiv.org/abs/1911.02508)

**Automated Interpretability Agents**

* [A Multimodal Automated Interpretability Agent](https://arxiv.org/abs/2404.14394)


**Reasoning**

* [On the Hardness of Faithful Chain-of-Thought Reasoning in Large Language Models](https://arxiv.org/abs/2406.10625)  
* [Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models](https://arxiv.org/abs/2505.13774)   
* [Thought Anchors: Which LLM Reasoning Steps Matter?](https://arxiv.org/abs/2506.19143)  
* [All for One: LLMs Solve Mental Math at the Last Token With Information Transferred From Other Tokens](https://arxiv.org/pdf/2509.09650)  
* [Neuron Activation as a Unified Lens to Explain Chain-of-Thought Eliciting Arithmetic Reasoning of LLMs](https://arxiv.org/pdf/2406.12288)