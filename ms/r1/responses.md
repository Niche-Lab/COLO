Dear editors and reviewers,

We very much appreciate the professional and insightful feedback from the reviewer. We have carefully addressed the concerns and provided a point-to-point discussion below. 

Reviewer’s comments: underscored and blue font
Authors’ responses: black font

Reviewer 1: Summary:
+++++++
The manuscript reports on a study targeted at the exploration of machine learning model generalizability in animal husbandry settings, more precisely in cow detection under different circumstances.
To this end, a valuable new data set has been created, which is named the `Cow Localization (COLO) dataset'.
The dataset represents different sensor settings and environment conditions, ranging from top-view, over side-view angles, day and night imaging situations, to indoor and outdoor sceneries.
It then serves as basis for experimental verification of three research hypotheses about 1) the impacts of the abovementioned circumstances on model generalization, 2) the effect of model size on generalization, and, 3) the supporting effect on domain-dependent fine-tuning in contrast to off-the-shelf pre-trained model utilization.
In general, the paper is very well written and comes with a generally clear and straightforward structure.
The objectives are clearly stated and discussed with their implications on the targeted application domain of animal husbandry / precision livestock farming.
That said, after reading the manuscript, several questions remained unanswered along with some unclarities which I'd like to summarize below in the hope that they will aid the authors in improving their manuscript.

Authors' response: Thank you for your accuracy summary of our work and your suggestion. We have carefully addressed your comments and concern blow.

Major concerns:
+++++++++++
1) Restriction to YOLO model family:
- The set focus on YOLO models in version 8 and 9 as well as the adoption of default settings and usage of additional functions provided by Ultralytics framework (especially image augmentation), limits the significance and achievement of stated objective in my opinion. The paper would have benefitted strongly from a comparison of different model architectures next to single-stage YOLO models. Authors are encouraged to consider inclusion of Vision-Transformer-based object detection models such as RT-DETR and more conventional two-stage models e.g., based on Faster-R-CNN in their analysis. Furthermore, YOLOv8 and v9 are not representing the latest model versions anymore (why not opting for or comparing for v10 or v11?)

Authors' response: We appreciate your suggestion and agree with you that the previous studied models are not up-to-date. It is partly because the study was conducted in 2023 and these two were the most recent models at that time. We have now updated the manuscript to include the latest YOLOv11 and YOLOv12 models, as well as two size variants of RT-DETR models that leverage the attention mechanism to provide a more comprehensive benchmarking study. We do not consider Faster-R-CNN a suitable choice as it is slower and does not show competitive performance compared to other models based on the benchmarking COCO dataset as to date.

- Furthermore, the impact on auto-functions provided by the Ultralytics framework, especially the use of image augmentation, have not been considered in more detail. Thus, it is not clear to what extent the observations regarding the generalization capability originate from the model architecture or from the additional augmentation step. Authors should take this into consideration in their experiments.

Authors' response: Thank you for pointing out this concern, and we agree with this comment that the generalization capability directly come from the model characteristics might be confounded and overestimated due to the image augmentation implemented by the Ultralytics framework. However, as this study is focused on the practical use case of object detection models, where image augmentation is a standard practice. Hence, we still lean towards including the data augmentation step in our experiments and evaluation. To ensure readers are aware of this, we have added a description of the additional pre-processing steps in the methods section for clarity.

1) Concerns regarding the experimental setup:
- Still there is a restriction to only one data set by the authors, even if it is designed to explicitly contain different sensor settings. Nevertheless, I feel that hypotheses 1) and 3) should have been also evaluated on a `third-party' data set, which exhibits completely different settings, camera sensors, barn facilities, etc. This would clearly substantiate the claims and increase the reliability of the results.

Authors' response: Thank you for your suggestion. We have included a third-party dataset that is published by Inner Mongolia University (IMU) to address your suggestion. This dataset consists of three different camera angles with entirely different settings and barn facilities compared to the dataset collected at Virginia Tech.

- Hyperparameter table (Tab. 3) shows that the optimizer is automatically chosen and that much augmentation has been applied. This is functionality of the Ultralytics framework, not really belonging to the model itself. Since you have not controlled or systematically varied these additional functions, it remains unclear to what extent especially the augmentation influenced the robustness of the model against changed circumstances in the input images, which clearly has an effect on the investigated generalization capability of the models.

Authors' response: Thank you for pointing out this concern. We acknowledge that this was not clearly stated and have now added a description of the additional pre-processing steps in the methods section for clarity. All the training is performed by the Adam optimizer.

1) Presentation of the quantitative results:
- The intuition of Figure 3 and the single points is unclear: By what means are these points representing different model configurations? What is meant with configuration? Learned weights? Please explain the intuition of this figure in more detail.

Authors' response: Thank you for your comment. We intended to show the relationship between different metrics, as some metrics, such as mAP@0.5:0.95, might be difficult for readers to interpret in the non-technical community. We have redesigned the figure and focus on translating the metric mAP@0.5:0.95 to mAP@0.5, indicating that even with moderate performance, such as 0.6, in mAP@0.5:0.95, it still implies a good performance of 0.9 in mAP@0.5, which is more suitable for a case where counting is the only interest and location precision could be less of interest for farmers.

- I was confused by the results shown in Figure 5 where YOLOv8's performance often drops for the largest variants in most of the settings. Please elaborate more on this salient effect.

Authors' response: Thank you for pointing this out. Initially, we were also surprised by this observation and figured it might due to the overfitting of the complicated model. But after further investigation, the large model takes more updates to reach its optimal performance. Originally, we only trained the model for 100 epochs. This issue has been addressed by unifying the training period by a consistent number of steps (i.e., 2^13 = 8192 steps) instead of fixing the number of epochs, and with training size of 500 images and batch size of 16, the training period is now raised to approximately 256 epochs, with early-stopping windows of quarter of the training period. This change has led to a result where most large and complex models still outperform the smaller models in most cases.

- Learning curves in Figure 6 do not show full convergence in the investigated metrics. Have you inspected the loss curves? Was early stopping used? According to Tab. 3, the learning epochs are limited to 100, but was this sufficient for assuming convergence? Please clarify this in the revised manuscript.

Authors' response: Thank you for your comment. This issue has been addressed as described in the previous point. We have unified the training period by a consistent number of steps (i.e., 2^13 = 8192 steps) instead of fixing the number of epochs. And we ensure that the training is fully converged to obtain the representative estimate of generalization performance.

- You state at some places that „performance differences … were insignificant...". However, neither statistical hypothesis testing was applied, nor were tables given showing descriptive statistics about the observed results. This substantially limits the expressiveness and reliability of the reported results. I'd like to strongly encourage the authors to carefully reconsider the stochastic effects induced in the experiments through utilizing automated functions of the Ultralytics library (e.g., random mixup augmentation). 

Authors' response: Thank you for your comment. We have added the descriptive statistics to show the significance of the performance differences. We also have added brief descriptions of how the data augmentation would affect the estimated performance and use standard devication to show the stochastic effects in the experiments.
   
- You state on page 8 that „each training sample size was repeated 50 times with different random seeds", but it is not clear what exactly was varied here (subsampling?). Furthermore, figure 2 is unclear. It states to show the cross-validation configurations, but it appears not to be a real cross-validation (which would have been favorable). Why are for instance in the Top2Side setting (upper middle) the sizes of the rectangles for train and test equally sized? Does each green test rectangle always stand for the fixed 100 images? The idea of the figure is good, but the details need to be elaborated further to increase comprehension of the reader.

Authors' response: Thank you for your comment. We have redesigned the figure and have added more details to clarify the cross-validation configurations and the representation of the rectangles in the figure. 

- Lastly, the authors are strongly encouraged to explicitly elaborate on the current limitations of their experimental study which result from the experimental design choices and decisions felt.

Authors' response: Thank you for your comment. We have ensured that the limitations are clearly stated in the revised manuscript. We acknowlege that the current study is limited to object detection tasks, in which the obtained conclusions may not necessarily be applicable to more complicated tasks such as image segmentation or captioning. This study also only focuses on indoor barn settings, the readers may need to be cautious when applying the conclusions to outdoor settings. However, this study provides a practical example of how small models could outperform larger models that is ten times larger in size in certain conditions, which is beneficial for 

Additional minor points and open questions:
++++++++++++++++++++++++++++
- The introduction section is quite long. The authors might consider to split it up into introduction and background. Furthermore I was a bit surprised by the way the authors structured the introduction section I by using subheadings. This supported my impression that splitting up into separate introduction and background sections might be an option.

Authors' response: Thank you for the careful review. We have split the introduction into two sections: introduction to ellaborate the motivation of the study and background to provide essential literacy review.

- page 3 line 46-47: Please double check the sentence structure.

Authors' response: Thank you. We have revised the sentence structure and added references to support the statement.

- page 5: I really liked the style of presenting the implications of the key hypotheses for the investigated application domain have been discussed at this place. You could even go a step further and already state here, how you attempt to verify the hypotheses experimentally.

Authors' response: Thank you for your comment. We have extended this section to describe how we statistically test the hypotheses and the implications of the key hypotheses for the investigated application domain.

- page 7 / line 5: I was wondering who annotated the data and what measures for quality assurance have been applied. Please briefly touch upon this aspect.

Authors' response: Thank you for your comment. The annotations were completed by the authors and they are visually verified by the authors to ensure that the bounding boxes are minimally defined to enclose the entire cow body, including the tail and four legs. When the cows are occluded, the bounding box is extended to human's best guess to include the occluded parts. We have added this information in the revised manuscript.

- page 7 / line 39: Why did you choose a confidence threshold of 0.25? Is this an application-specific choice? Has it been found to be appropriate for cow detection? Please justify your choice briefly.

Authors' response: Thank you for your comment. A low confidence threshold results in a higher number of detections and higher sensitivity (i.e., true positive rate), which is more suitable for the cow counting task where all instances are more likely to be counted, although with a risk of overcounting. We acknoledge that this could create unnecessary ambiguity, as providing mAP values that is confidence-independent is more comprehensive. Hence, we discard the precision and recall metrics and only report the mAP metrics in the revised manuscript. We have also added a description of the intepretation of the mAP metrics in the revised manuscript.

- page 8 / line 47: m for million could be confused with m for medium size YOLO model. —> use mio. instead?

Authors' response: Thank you for your comment. To make the message clearer, we have remove the model size 'm' and only consider the small and large model size. Hence the confusion should be resolved.
 
- page 10 / Tab. 2: Please try to better map the table entries to the experiments.

Authors' response: Thank you for your comment and it is a great suggestion. We have revised the table accordingly.

- page 15 / line 57-58: Why exactly this baseline training time? Please justify briefly.

Authors' response: Thank you for your comment. The baseline training time is defined by the minimum training time of all studied combination (i.e., model size and sample size). We have added this information in the revised manuscript.

- page 16 / line 50: Please add a reference for supporting the statement that FPS higher than 30 is essential for real-time inference. An appropriate inference efficiency is highly application dependent.

Authors' response: Thank you for your comment. We have added a reference to support the statement.

- page 17 / line 23: What is exactly meant with `adaptive methods for enhancing model generalization'? And what do you consider `advanced data augmentation techniques'? Please give examples and cite according works.

Authors' response: Thank you for your comment. We have extended this statement for the clarification. What we intended to say is that since changing the camera angle is the major factor affecting the model generalization in our study, an image augmentation method that simulates different view angles in 3D space could be beneficial for this challenge. We have added a reference to support this statement.

- The authors might consider to add the hyperparameter table to the main text. It is important for judging on the experimental setup and since it is the only information provided in the appendix it could be integrated in the materials and methods section detailing the experimental setup.

Authors' response: Thank you for your comment. We have moved the hyperparameter table to the main text in the methods section.

- The references section should receive a careful revision: Please check for the last access dates for each URL. Check furthermore for the consistency in the provided information. For instance Chien-Yao Wang et al. 2024 is missing relevant information such as the journal/venue.

Authors' response: Thank you for your comment. We have revised the references section and ensured that all the references are consistent and complete.


Recommendation:
++++++++++++
Overall, the authors tackle a commendable research questions often neglected in research works applying AI to a particular domain. The topic of the work falls in the scope of the journal.
Despite my criticized points as listed above, I feel that the paper is not that far away from being publishable, however one round of careful revision is needed in my opinion.
