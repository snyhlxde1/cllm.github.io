From programming to providing legal and health advice, large language models (LLMs) are revolutionizing the landscape of human lives. 

<p align="center"><img src="clm_objective.png" alt="autoregressive" width="300"></p>
<p align="center">Figure 1: Illustration of conventional Auto-regressive generation: one token is generated at a time.</p>
In the following section, we'll introduce consistency large language models (CLLMs), a new family models developed with our proposed techniques to reduce inference latency with Jacobi decoding.

## Speculative decoding
Specualtive decoding is first introduced by [this paper](https://arxiv.org/abs/2211.17192). Put simply, speculative decoding recognizes that some tokens are straightforward to generate, while others are more challenging. To address this, we can utilize a streamlined 'draft' model for the easier tokens and a more comprehensive 'target' model for the complex ones.
Specifically, to ensure that speculative decoding produces identical output to the original generation method, the draft model proposes tokens which are then validated by the target model.

<p align="center"><img src="spec1.png" alt="Example-1" width="700"></p>

As shown in the picture above, the draft model proposes five tokens: `["I", "like", "cooking", "and", "traveling"]`. These are then forwarded to the target model for parallel verification. In this example, the third token, playing, was proposed inaccurately. As a result, only the first three tokens, `["I", "like", "playing"]`, are generated in this step.

<p align="center"><img src="spec2.png" alt="Example-2" width="800"></p>

For the second step, starting from the playing token, the draft model proposes a new set of tokens: `["piano", "and", "reading", "books"]`. Let's assume, fortunately, that all these tokens are accurately proposed and subsequently confirmed by the larger model. Additionally, the larger model produces an extra token, `<EOS>`, based on the last verified token `.`. The generation process concludes at this point since the end-of-string token (`<EOS>`) has been produced.

##### Why can speculative decoding reduce latency?
For the draft model, typically much smaller than the target model, it retains its autoregressive nature, generating tokens one at a time. Conversely, the target model can validate multiple tokens in a single forward pass (refer to [the paper](https://arxiv.org/abs/2211.17192) for more detail). Consequently, speculative decoding helps amortize the overhead of loading model weights and key-value caches. Originally, each token required accessing the weights and key-value cache, but now it's reduced to just one access per $k$ tokens, where $k$ represents the number of accepted tokens in each generation step.

### Observation & Online speculative decoding (OSD)
Based on the original speculative decoding, we have several interesting observations:
<p align="center">
<img src="analysis_c.png" alt="Architecture" width="150">         
<img src="analysis_k.png" alt="Architecture" width="150">
</p>

1. **Propose accuracy is important** The draft model must approximate the target model sufficiently to achieve a reduction in latency. We use the symbol $\alpha$ to represent the proposed accuracy, defined as the expected token acceptance rate. For instance, if the token acceptance rate is 0.7, it signifies that, on average, 70% of tokens proposed by the draft model will be accepted by the target model.
As illustrated in the figures above, for smaller values of $\alpha$, speculative decoding can even lead to performance degradation, as indicated by a speedup factor less than 1, especially when the draft model is of considerable size. Furthermore, the relationship between speedup and $\alpha$ exhibits a superlinear behavior; doubling the acceptance rate can result in a speedup exceeding 2x.
2. **The draft model knows the correct answer** The process of speculative decoding inherently detects inaccuracies in the smaller draft language model (LLM) and provides correct solutions for these inaccuracies. In the specific example provided above, we can pinpoint the occurrence of a proposal error when the `cooking` token is suggested, whereas the correct token should be `playing`. Additionally, we have access to the probability distribution associated with these two tokens. This essentially means that we gain valuable insights into the areas and strategies for refining the draft model, all without incurring any additional cost to get the label. 
3. **There are many spare FLOPs in the serving system** TODO

Based on the observations above, we propose the online speculative decoding (OSD) algorithm:
<p align="center"><img src="arch.png" alt="Architecture" width="800"></p>

For each prompt, the draft model suggests multiple tokens in a single step. The target model then verifies these tokens, accepting some and rejecting others. If the student proposes incorrect tokens, both the draft and target distributions are stored in a buffer. Once the buffer exceeds a specified threshold, the draft model is updated by calculating the loss between the draft and target distributions using various distance metrics.

## Experiments
**Online learning**: There are two important questions to anwer here: (1) Does the online algorithm increase the token acceptance rate? And is this enhancement comparable to the rates achieved in offline settings, which serve as an upper bound given their full access to data? (2) How quickly does the online algorithm increase the token acceptance rate, thereby indicating that the compact model has grasped the underlying distribution?

In this experiment, we pick LLaMA-160M as the draft model and Vicuna-7B as the target model. In the beginning, online speculative decoding yields a lower token acceptance rate in comparison to the offline distilled model. Nevertheless, these acceptance rates rise swiftly as the draft model is exposed to more data. We also annotate the token acceptance rate from the offline setting to highlight the potential peak performance that the online serving system could reach.
<p align="center">
<img src="online.png" alt="Online Learning" width="300"></p>
<p align="center">The x-axis represents the number of records that OSD has processed. Alpha is averaged over the most recent 50 records.</p>

**Distribution shift**
In this experiment, we want to know how quickly can OSD adapt to distribution shift. As shown below, OSD's alpha value dips notably at distribution boundaries, especially around 2K, 4K, and 6K records. This is anticipated since the draft model initially struggles when faced with a new distribution. However, the alpha value rebounds quickly as OSD processes more data, highlighting its adaptability to shifting query distributions. 
<p align="center"><img src="shift.png" alt="Distribution Shift" width="600"></p>

We also compared our results to those from a static setting. To ensure the draft model wasn't just memorizing data, we chose samples distinct from the online evaluation data. These samples correspond to 30%, 50%, 70%, and 100% of each dataset's online evaluation volume, at 0.6K, 1K, 1.4K, and 2K quantities respectively. As depicted, upon an initial shift in query distribution, OSD's performance aligns with or slightly trails the distillation with 30% data. However, it quickly catches up, matching or even surpassing performances seen with 70% to 100% data access. This highlights OSD's ability to rival models fully exposed to the query distribution, even without intimate knowledge of the underlying query dynamics.


**Arena dataset**
<p align="center"><img src="arena_language.png" alt="Architecture" width="300"> <img src="arena_class.png" alt="Architecture" width="300"></p>

We evaluate OSD on real LMSYS-chat conversations that span 4 months.
First, we categorize conversations based on the language and we focus on conversations among the top five languages, excluding English. For every chosen language, we use an independent LLaMA-160M to serve as our draft model. All draft models share the same Vicuna-7B as the target model. The token acceptance rate, averaged over the latest 100 requests, reveals that OSD's enhances rates by 0.1 to 0.2, even with under 2K data points. Notably, Japanese was the easiest while Portuguese was the toughest.
We also clustered English conversations by topics using the [fine-tuned distilled bert model]((https://huggingface.co/alimazhar-110/website_classification)), focusing on the top five. As shown above, acceptance rates are above 0.6 across topics, with Social and Computer discussions peaking near 0.8.

## Final words
We invite you to refer to the [OSD paper](TODO) for comprehensive details! While we plan to release the code to replicate the results presented above, it's important to note that this code is not intended for use in a production serving system; rather, it serves as a proof of concept for the idea. We are actively engaged in the development of a fully operational system, so please stay tuned for further updates!
