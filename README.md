# GLAUCOMA DETECTION USING VISION TRANSFORMERS ON FUNDUS IMAGES
## A TAKE ON THE AIROGS LITE CHALLENGE

This is the code repository for our solution approach to the [AIROGS Lite challenge](https://airogs-lite.grand-challenge.org). Please note that this was done as coursework after the challenge ended, and was therefore not submitted to the challenge. A 4-page paper outlining our approach and our results can be found [here](https://github.com/valbuc/AIROGSLite-ViT/blob/main/GlaucomaDetectViT.pdf).


## Instructions:

* Download development data from: https://zenodo.org/record/7056954
* Download test data from: https://zenodo.org/record/7178671
* Optional
  * Use `resize_square.ipynb` to generate images of 640 x 640 pixels for fine-tuning yolo on fovea detection
  * Fine-tune yolo using `YOLOv5_notebook.ipynb`
* Use `lossless_od_crops_using_yolo_predictions.ipynb` to crop images for the classification model
* Train the classifier using `classifier.py`. Setting the predict_data_dir to the downladed and extracted test data will generate predictions
* Run `ensemble_metrics.ipynb` to generate performance metrics  
* Run `submission.ipynb` to generate a submission file 
