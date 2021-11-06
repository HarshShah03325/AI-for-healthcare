# AI-FOR-HEALTHCARE

The importance of Artificial Intelligence in Healthcare is increasing significantly. This project provides the implementation of deep learning and machine learning
techniques to improve traditional healthcare systems. The primary aim of health-related AI applications is to analyze relationships between prevention or treatment techniques and patient outcomes. The 3 steps of patient management include:
- Medical diagnosis.
- Medical prognosis.
- Medical treatment.

AI algorithms can also be used to analyze large amounts of data through electronic health records for disease prevention and diagnosis. Additionally, hospitals are looking to AI software to support operational initiatives that increase cost saving, improve patient satisfaction, and satisfy their staffing and workforce needs.

## Challenges before AI-for-healthcare

### Data collection
In order to effectively train Machine Learning and use AI in healthcare, massive amounts of data must be gathered Acquiring this data, however, comes at the cost of patient privacy in most cases and is not well received publicly. For example, a survey conducted in the UK estimated that 63% of the population is uncomfortable with sharing their personal data in order to improve artificial intelligence technology. The scarcity of real, accessible patient data is a hindrance that deters the progress of developing and deploying more artificial intelligence in healthcare. 

### Automation
According to a recent study, AI can replace up to 35% of jobs in the UK within the next 10 to 20 years. However, of these jobs, it was concluded that AI has not eliminated any healthcare jobs so far. Though if AI were to automate healthcare related jobs, the jobs most susceptible to automation would be those dealing with digital information, radiology, and pathology, as opposed to those dealing with doctor to patient interaction.
Automation can provide benefits alongside doctors as well. It is expected that doctors who take advantage of AI in healthcare will provide greater quality healthcare than doctors and medical establishments who do not. AI will likely not completely replace healthcare workers but rather give them more time to attend to their patients. AI may avert healthcare worker burnout and cognitive overload.

### Bias
Since AI makes decisions solely on the data it receives as input, it is important that this data represents accurate patient demographics. In a hospital setting, patients do not have full knowledge of how predictive algorithms are created or calibrated. Therefore, these medical establishments can unfairly code their algorithms to discriminate against minorities and prioritize profits rather than providing optimal care.
There can also be unintended bias in these algorithms that can exacerbate social and healthcare inequities. Since AI’s decisions are a direct reflection of its input data, the data it receives must have accurate representation of patient demographics. These biases are able to be eliminated through careful implementation and a methodical collection of representative data. 

## Following are some ideas that have the potential to improve traditional healthcare systems:

# [Brain tumor segmentation using MRI](https://github.com/HarshShah03325/AI-for-healthcare/tree/main/MRI%20Segmentation)
![](main_assets/heading.png)

## Description
- The objective of the project is to use deep learning to diagnose tumor from MRI images.
- The project uses a 3D U-Net model able to diagnose 4 labels - background, edema, enhancing tumor and non-enhancing tumor.
- Soft dice loss is used as loss-function to optimize and offset the low performance of other traditonal optimizers due to heavy class imbalance.
- The predictions are done on patch level for a sub volume of the MRI. Finally, we combine the result of patches to obtain a full MRI scan result.


# [Chest X-Ray Analysis](https://github.com/HarshShah03325/AI-for-healthcare/tree/main/Chest%20X-Ray%20Analysis)
Diagnose 14 pathologies on Chest X-Ray using Deep Learning. Perform diagnostic interpretation using GradCAM Method

![](main_assets/xray-header-image.png)

## Description
- The objective of the project is to use a deep learning model to diagnose pathologies from Chest X-Rays.
- The project uses a pretrained DenseNet-121 model able to diagnose 14 labels such as Cardiomegaly, Mass, Pneumothorax or Edema. In other words, this single model can provide binary classification predictions for each of the 14 labeled pathologies.
- Weight normalization is performed to offset the low prevalence of the abnormalities among the dataset of X-Rays (class imbalance).
- Finally the GradCAM technique is used to highlight and visualize where the model is looking, which area of interest is used to make the prediction. This is a tool which can be helpful for discovery of markers, error analysis, training and even in deployment.

# [Risk prediction Models](https://github.com/HarshShah03325/AI-for-healthcare/tree/main/Risk%20Models)

## Description:
- The objetive of this project is to predict the 10-year risk of death of individuals based on 18 different medical factors such as age, gender, systolic blood pressure, BMI etc.
- Two types of models were used : Linear model and random forest classifier model.
- Finally, I compared different methods for both models using concordance index(c_index).


