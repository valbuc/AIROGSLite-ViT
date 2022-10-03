# package imports
import json
from src.crop_retina import crop_retina
from src.yolo import train_yolo, predict_yolo
from src.unet import train_unet, predict_unet
from src.crop_od import crop_od
from src.vit import train_vit, predict_vit
from src.evaluate import evaluate
import pickle
import logging


with open("pipeline_config.json") as fp:
    config = json.load(fp)

# setting experiment name
exp_name = config["Experiment Name"]
logging.info(f"Initiated pipeline for: {exp_name}")


# asserting correct configuration file
assert config["Crop Retina"]["Method"] in [
    "Diameter",
    "Skip",
], "Invalid method given for 'Crop Retina'"
assert config["OD Detection"]["Method"] in [
    "Yolo",
    "Unet",
    "Skip",
], "Invalid method given for 'Od Detection'"
assert config["Crop OD"]["Method"] in [
    "Factor",
    "Skip",
], "Invalid method given for 'Crop OD'"

# assert skipping is used in right order

# if no directory data/models exists make one


# shuffling is not done again as different computers may do it differently
if __name__() == "__main__":
    # 1. crop square around field of view - [Diameter (treshold), Skip]
    logging.info("Cropping Retina")

    if config["Crop Retina"]["Method"] == "Diameter":
        logging.info("    Using Method Diameter")

        treshold = config["Crop Retina"]["Threshold"]
        inpath = config["Crop Retina"]["Original Images Path"]
        outpath = config["Crop Retina"]["Retina Path"]
        crop_retina(inpath, outpath, treshold)

        logging.info("   Done")
    else:
        logging.info("    Skipping")
        logging.info("    Done")

    # Possibly add extra step to combine annotations - but will we change this part ever?

    # 2. run od detection - return center and width of OD - [YOlO, Unet, Skip]
    logging.info("Detecting ODs")
    inpath_retinas = config["OD Detection"]["Retina Path"]
    inpath_labels = config["OD Detection"]["Label Path"]
    outpath_predictions = config["OD Detection"]["Prediction Path"]

    if config["OD Detection"]["Method"] == "Yolo":
        logging.info("    Using Yolo")

        yolo_model = train_yolo(inpath_retinas, inpath_labels)
        pickle.dump(yolo_model, open(f"data/models/yolo_{exp_name}.pkl", "wb"))
        predict_yolo(yolo_model, inpath_retinas, outpath_predictions)

        logging.info("    Done")
    elif config["OD Detection"]["Method"] == "Unet":
        logging.info("    Using Unet")

        unet_model = train_unet(inpath_retinas, inpath_labels)
        pickle.dump(unet_model, open(f"data/models/unet_{exp_name}.pkl", "wb"))
        predict_unet(unet_model, inpath_retinas, outpath_predictions)

        logging.info("    Done")
    else:
        logging.info("    Skipping")
        logging.info("    Done")

    # 3. crop out OD using a certain factor - use square image if no OD is detected [Factor, Skip]
    logging.info("Cropping ODs")
    images_path = config["Crop OD"]["Original Images Path"]
    predictions_path = config["Crop OD"]["Prediction Path"]
    outpath_ods = config["Crop OD"]["ODs Path"]

    if config["Crop OD"]["Method"] == "Factor":
        logging.info("    Using Unet")

        factor = config["Crop OD"]["Factor"]
        crop_od(images_path, predictions_path, outpath_ods, factor)

        logging.info("    Done")
    else:
        logging.info("    Skipping")
        logging.info("    Done")

    # 4. train prediction model [ViT]
    logging.info("Classifying Glaucoma")
    inpath_ods = config["Glaucoma Classification"]["ODs Path"]
    inpath_labels = config["Glaucoma Classification"]["Label Path"]
    outpath_predictions = config["Glaucoma Classification"]["Prediction Path"]

    if config["Glaucoma Classification"]["Method"] == "Yolo":
        logging.info("    Using Yolo")

        vit_model = train_vit(inpath_ods, inpath_labels)
        pickle.dump(vit_model, open(f"data/models/vit_{exp_name}.pkl", "wb"))
        predict_vit(vit_model, inpath_ods, outpath_predictions)

        logging.info("    Done")
    else:
        logging.info("    Skipping")
        logging.info("    Done")

    # 5. evaluate - this has no options as it should always be the same
    logging.info("Evaluating")
    true_labels_path = config["Evaluation"]["True Path"]
    prediction_path = config["Evaluation"]["Prediction Path"]

    evaluate(true_labels_path, prediction_path)
