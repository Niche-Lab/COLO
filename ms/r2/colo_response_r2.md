 Reviewer 1: First of all, I'd like to thank the authors for their careful revision. Almost all of my concerns have been properly addressed!

There only remain a few minor points, which I'd prefer to also see addressed in a finally published version:

1) In the response of my second concern, the authors state that two example images of augmentations would have been added to the revised paragraph. These cannot be found in the manuscript. Please add them.



2) To address my comment regarding the statistical assessment of their results, the authors applied t-tests. However, it was not described if the conditions to safely use parametric hypothesis tests have been tested, too. Usually it would need to be tested for at least normal distribution of the sample obtained from the experiment repetitions (using QQ-Plots and/or Shapiro-Wilk tests). Furthermore, in the added description for Study 3, I found details missing, such as the number of experiment repetitions performed. Please add these details.

3) The authors also mentioned in their letter that they "removed the random mixup augmentation" in order to avoid ambiguity. This was not reflected in the manuscript and not totally clear to me. Clearly such a stochastic augmentation method constitutes another factor of randomness, but this could also be controlled using fixed seeds for each experiment repetition.

4) In their reply regarding my question about the set confidence threshold, the authors explain that they now removed the metrics precision and recall completely and restrict their results to mAP. This is too bad, since in my opinion insights into the performance of the models in these metrics along with a discussion on their importance for the use case at hand is clearly of value. I'd like to encourage the authors to include them again in their final version.

5) Furthermore, I would prefer to have for each study and the plots shown in the manuscript also a corresponding result table containing the numerical results, e.g., means with with standard deviation for each of the considered metrics (mAP50 and 50-95, recall, precision, training speed, etc.).

6) Lastly I recommend a final proof-read for cleaning up last typos, etc. (e.g., in line 175 `were' should be `are', caption of Fig. 2 `contains' --> `contain', `affect' --> `effect', line 246 `augmentation' --> `augmentation techniques' or `augmentations'?). Same for the bibliography, please look for general capitalization of the references and inclusion of access dates for [30-32] and specify [45] and [47] more clearly, please. Lastly the running title in the manuscript `A model generalization...' --> `A model generalization...'. In that regard I also wondered if the title might want to be subject of reconsideration: `... localizing indoor cows with COLO...' --> `with the COLO...'. And the term `indoor cows' appears a bit uncommon to me, even if I guess it refers to `localizing cows in indoor housing' or the like.

