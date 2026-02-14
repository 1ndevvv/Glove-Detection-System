# Part 2: Reasoning-Based Questions

## Q1: Choosing the Right Approach

**Scenario**: Identifying whether a product is missing its label on an assembly line. The products are visually similar except for the label.

**Question**: Would you use classification, detection, or segmentation? Why? What would be your fallback if the first approach doesn't work?

### Answer:

I would use object detection rather than simple classification because we need to localize and identify individual hands in the image. In a factory environment, multiple workers or hands may appear in a single frame, so classification alone would not provide spatial information. Detection allows us to draw bounding boxes around each hand and classify them as gloved_hand or bare_hand, which is important for monitoring compliance. It also makes the system more interpretable, since supervisors can visually verify detections. If detection does not perform well, my fallback would be segmentation to more precisely analyze the hand region, especially in cases where gloves are thin or similar in color to skin

---

## Q2: Debugging a Poorly Performing Model

**Scenario**: You trained a model on 1000 images, but it performs poorly on new images from the factory.

**Question**: Design a small experiment or checklist to debug the issue. What would you test or visualize?

### Answer:

First, I would check for domain shift by comparing the training dataset with the new factory images, focusing on lighting, camera angles, motion blur, and background differences. I would visualize predictions and categorize errors into false positives and false negatives to identify patterns, such as missed bare hands in low light. Next, I would review the train-validation split to ensure there was no data leakage or imbalance between gloved and bare classes. I would also examine the confusion matrix and precision-recall curves to understand which class is underperforming. Finally, I would test retraining the model with additional factory-specific images or stronger augmentation to improve generalization.

---

## Q3: Accuracy vs Real Risk

**Scenario**: Your model has 98% accuracy but still misses 1 out of 10 defective products.

**Question**: Is accuracy the right metric in this case? What would you look at instead and why?

### Answer:

No i think accuracy is not the right metric in this safety compliance system because most workers may be wearing gloves, making the dataset imbalanced. Even with 98% overall accuracy, missing 1 out of 10 bare_hand cases means the model has a high false negative rate for unsafe behavior. In this context, failing to detect a bare hand is a serious safety risk, so recall for the bare_hand class is more important than overall accuracy. I would prioritize recall and evaluate the confusion matrix to measure how many unsafe cases are being missed. Additionally, I would consider using an F-beta score that emphasizes recall, since minimizing missed safety violations is critical.

---

## Q4: Annotation Edge Cases

**Scenario**: You're labeling data, but many images contain blurry or partially visible objects.

**Question**: Should these be kept in the dataset? Why or why not? What trade-offs are you considering?

### Answer:

I would keep some of the blurry or partially visible hands in the dataset because they reflect real-world factory conditions such as motion blur or occlusion. Removing them might improve validation accuracy artificially but reduce the model’s robustness in deployment. However, it is important that annotations remain accurate and consistent to avoid introducing label noise. The trade-off is between dataset cleanliness and real-world generalization. Including challenging examples helps the model learn to handle difficult scenarios, but excessive noisy labels could negatively affect training performance.

---

## Summary of Approach

Throughout these questions, my reasoning follows key principles:

1. **Business Context First**: Metrics and decisions must align with real-world costs and risks
2. **Data-Centric Mindset**: Most model failures stem from data issues, not architecture
3. **Systematic Debugging**: Use structured checklists and visualization to identify root causes
4. **Robustness Over Perfection**: Models should handle real-world messiness, not just clean test sets
5. **Cost-Aware Decisions**: Different error types have different consequences that must guide metric selection

These principles reflect the practical realities of deploying computer vision in production environments where reliability and business impact matter more than academic benchmark scores.

---

**Author**: Devanshu Shah   
**Contact**: devanshusah123499@gmail.com
