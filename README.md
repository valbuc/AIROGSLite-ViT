# AIROGSLite-AI4MI-VU-2022
Code repo for the Glaucoma AIROGS Lite challenge
https://airogs-lite.grand-challenge.org/

## General outline:

* Download development data from: https://zenodo.org/record/7056954
* Download test data from: https://zenodo.org/record/7178671
* Optional
  * using resize_square.ipynb generate images size 640 for yolo
  * train yolo model
* use lossless_od_crops_using_yolo_predictions.ipynb to generate crops for the classifer 
* train the classifier. Setting the predict_data_dir to the downladed and extracted test data will generate predictions
* run ensemble_metrics.ipynb to generate performance metrics for the dev test set  
* run submission.ipynb to generate a submission file 
* ...
* profit

## Examples
Classifier training command lines along with the generated log files (excluding the model checkpoints) can be found in ./experiments/

