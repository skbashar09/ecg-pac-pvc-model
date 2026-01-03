import json
import numpy as np
import argparse

from cnn_lstm_classifier_helper import EcgDecisionClass, LSTM_CNNClassifier


def load_json_test_data(file_path, data_type):
    """
    Loads data from a JSON file.
    Handles slight format differences between 'CPSC test' and 'wearable pvc' data.
    """
    # Load the annotations from the JSON file
    with open(file_path, "r") as f:
        all_annotations = json.load(f)

    segment_count = 0
    ecg_data_out = []
    print("----------------------------------------------------")
    print(f"Loading and processing {data_type} data from: {file_path}")

    for annotation in all_annotations:
        if data_type == "cpsc":
            # CPSC data format: [ecg_V, ecg_II, annotated segments, file ID/name]
            if "999" not in str(annotation[2]):
                segment_count += len(annotation[2])
                segment_select = annotation[2]
                ecg_V = np.array(annotation[0])
                ecg_II = np.array(annotation[1])
                ecg_data_out.append(
                    (
                        ecg_V[segment_select, :],
                        ecg_II[segment_select, :],
                        annotation[2],
                        annotation[3],
                    )
                )

        elif data_type == "wearable":
            # Wearable data format: [ecg_V, annotated, patient ID]
            if "999" not in str(annotation[1]):
                segment_count += len(annotation[1])
                segment_select = annotation[1]
                ecg_V = np.array(annotation[0])
                ecg_data_out.append(
                    (
                        ecg_V[segment_select, :],
                        annotation[1],
                        annotation[2],
                    )
                )

    print("------------------------------------------------------")
    print(
        f"Total annotation files = {len(all_annotations)} and Total ECG segments = {segment_count}\n"
    )
    return ecg_data_out


def main():
    parser = argparse.ArgumentParser(description="Evaluate the pre-trained ECG model.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cpsc_nsr", "cpsc_pac", "cpsc_pvc", "wearable_pvc"],
        help="The name of the test dataset to evaluate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model_pre_trained_weight.pth",
        help="Path to the pre-trained model file (.pth).",
    )
    args = parser.parse_args()

    # Making dictionary to store data set configurations

    dataset_configs = {
        "cpsc_nsr": {
            "file": "test_data/test_ecg_nsr_cpsc.json",
            "lead": 1,
            "type": "cpsc",
        },
        "cpsc_pac": {
            "file": "test_data/test_ecg_pac_cpsc.json",
            "lead": 0,
            "type": "cpsc",
        },
        "cpsc_pvc": {
            "file": "test_data/test_ecg_pvc_cpsc.json",
            "lead": 1,
            "type": "cpsc",
        },
        "wearable_pvc": {
            "file": "test_data/test_ecg_pvc_wearable.json",
            "lead": 0,
            "type": "wearable",
        },
    }

    config = dataset_configs[args.dataset]
    test_ecg_selected = load_json_test_data(config["file"], config["type"])
    ecg_lead_select = config["lead"]

    ############### loading model ############
    print(f"Loading model from: {args.model_path}")
    ecg_model = EcgDecisionClass(args.model_path)
    ###########################################

    sum_total, sum_nsr, sum_pac, sum_pvc, sum_other = 0, 0, 0, 0, 0

    for m in range(0, len(test_ecg_selected)):
        ecg_signal_segment = test_ecg_selected[m][ecg_lead_select]
        ecg_decision_out, _ = ecg_model.ecg_abnormal_decision(ecg_signal_segment)

        sum_total += len(ecg_decision_out)
        sum_nsr += np.sum(ecg_decision_out == 0)
        sum_pac += np.sum(ecg_decision_out == 1)
        sum_pvc += np.sum(ecg_decision_out == 2)
        sum_other += np.sum(ecg_decision_out == 3)

    print(f"----------------- RESULTS FOR: {args.dataset} --------------")
    print(f"Sum total = {sum_total}")
    print(f"NSR total = {sum_nsr}")
    print(f"PAC total = {sum_pac}")
    print(f"PVC total = {sum_pvc}")
    print(f"Other total = {sum_other}")
    print("-------------------------------------------")


if __name__ == "__main__":
    main()
