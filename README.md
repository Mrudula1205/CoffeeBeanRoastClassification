# Coffee Bean Roast Classification

## Introduction
 Coffee is one of the most beloved beverages worldwide, with its quality heavily influenced by the characteristics of the beans used. Coffee bean classification, which involves grading and sorting beans based on attributes like size, color, and defects, plays a critical role in ensuring consistent quality and flavor. Traditionally, this process has been labor-intensive and subjective, but advancements in technology have paved the way for automated coffee bean classification systems. These applications offer numerous benefits, such as:
 
- Grading and Sorting: Coffee beans are graded based on their size, shape, colour, and
 defects. Automated classification can ensure consistent quality and remove human
 biases.
- Automation: Automated classification systems save time and labour costs compared to
 manual sorting and grading. 
- Scalability: Large quantities of beans can be processed quickly, making the system
 scalable for industrial applications. 
- Improved Blending: By classifying beans based on specific attributes, producers can
 create tailored coffee blends with consistent flavour profiles. 
- Specialty Coffee: Identifying unique bean varieties or superior grades can help cater to
 speciality coffee markets. 
- CustomerTrust: Transparent grading and classification can instil confidence in
 customers about the quality of the coffee they purchase

## Dataset
The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224). It
contains a total of 1600 images belonging to 4 classes- Light, Green, Medium and Dark. Each class has 400 images

## Model Training
The dataset was trained with 3 CNN architectures- ResNet-152 V2, VGG-16 and Inception V3. Each of the base models trained on ImageNet was used, followed by a pooling layer, a dense layer with 1024 neurons and ReLU activation function, a dropout layer and a dense output layer with sigmoid activation function. The models were trained with the Adam optimizer and categorical cross entropy loss function, for 25 epochs

## Results

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Inception V3  | 0.97     | 0.97      | 0.97   | 0.97     |
| ResNet-152 V2 | 0.96     | 0.91      | 0.96   | 0.92     |
| VGG-16        | 0.96     | 0.91      | 0.96   | 0.91     |

## Deployment
[Streamlit App](https://coffeeroastclassification.streamlit.app/)
