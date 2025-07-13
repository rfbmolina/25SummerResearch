import os
import sys
from dotenv import load_dotenv

from src.data_loader import load_data
from src.split_data import split_data
from src.pipeline import run_pipeline 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

# Dictionary mapping classifier names to their respective classes
classifiers_dict = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "Balanced Random Forest": BalancedRandomForestClassifier,
    "XGBoost": XGBClassifier,
    "EasyEnsemble": EasyEnsembleClassifier,
}

# Dictionary mapping sampler names to their respective classes
samplers_dict = { 
    "Random Over Sampler": RandomOverSampler,
    "Random Under Sampler": RandomUnderSampler,
    "SMOTE": SMOTE,
    "Borderline SMOTE": BorderlineSMOTE,
    "ADASYN": ADASYN,
    "Cluster Centroids": ClusterCentroids,
    "None": None,
}

def choose_option_from_dict(prompt, options_dict):
    
    #Displays numbered options from a dictionary and prompts user to select one.
    #Parameters:
        #prompt: Instructional message to display to the user.
        #options_dict: A dictionary whose keys represent the options.

    #Returns:
        #The selected key from the dictionary.
    
    keys = list(options_dict.keys())
    print(prompt)
    print(', '.join(f"({i}) {k}" for i, k in enumerate(keys, start=1)))

    while True:
        try:
            user_input = int(input("Enter the number corresponding to your choice: "))
            if 1 <= user_input <= len(keys):
                return keys[user_input - 1]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    name = None 
    file_path = None
    classifier = None
    sampler = None
    random_state = 0 

    # Check if using .env defaults
    default_setup = False
    if len(sys.argv) == 2:
        if sys.argv[1] != "-default_values":
            print("Wrong parameter usage. Try:\npython run.py [-default_values]")
            exit(1)
        default_setup = True
        load_dotenv()

    if default_setup:
        name = os.environ.get("NAME")
        file_path = os.environ.get("FILE_PATH")
        selected_classifier = os.environ.get("CLASSIFIER")
        selected_sampler = os.environ.get("IMBALANCE_HANDLER")
        random_state = os.environ.get("RANDOM_STATE")

        # Validate that all values are provided
        if not all([name, file_path, selected_classifier, selected_sampler, random_state]):
            print("Missing required environment variables. Please ensure all of the following are set:")
            print("  - NAME")
            print("  - FILE_PATH")
            print("  - CLASSIFIER")
            print("  - IMBALANCE_HANDLER")
            print("  - RANDOM_STATE")
            exit(1)

        # Validate classifier and sampler values
        if selected_classifier not in classifiers_dict:
            print(f"\nInvalid CLASSIFIER: '{selected_classifier}'")
            print("Valid CLASSIFIER options are:")
            for k in classifiers_dict.keys():
                print(f"  - {k}")
            exit(1)

        if selected_sampler not in samplers_dict:
            print(f"\nInvalid IMBALANCE_HANDLER: '{selected_sampler}'")
            print("Valid IMBALANCE_HANDLER options are:")
            for k in samplers_dict.keys():
                print(f"  - {k}")
            exit(1)

        try:
            random_state = int(random_state)
        except ValueError:
            print(f"\nInvalid RANDOM_STATE '{random_state}'")
            print("RANDOM_STATE must be an integer.")
            exit(1)

    else: 
        # Pipeline name
        name = input("Choose a name for your pipeline: ")

        # File path
        file_path = input("Please enter the path to your data: ")

        # Choose classifier
        selected_classifier = choose_option_from_dict("Choose a classifier:", classifiers_dict)

        # Choose sampler
        selected_sampler = choose_option_from_dict("Choose an imbalanced data handler (sampler):", samplers_dict)

        # Random state
        while True:
            random_state = input("Please choose a random state (any number >= 1): ")
            try:
                random_state = int(random_state)
                break 
            except ValueError:
                print(f"Invalid random state: '{random_state}'")

    # Initialize classifier
    classifier_cls = classifiers_dict[selected_classifier] 
    classifier = classifier_cls()

    # Initialize sampler
    if selected_sampler != "None":
        sampler_cls = samplers_dict[selected_sampler] 
        sampler = sampler_cls()
    else:
        sampler = None 

    # Display configuration
    print("\n############## Pipeline Configuration ##############")
    print(f"Pipeline name: {name}")
    print(f"File path: {file_path}")
    print(f"Classifier: {selected_classifier} -> {classifier}")
    print(f"Sampler: {selected_sampler} -> {sampler}")
    print(f"Random state: {random_state}")
    print("#####################################################\n")

    X, y, groups = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y, groups, random_state) 

    if selected_classifier == "XGBoost":
        y_train=(y_train == 1).astype(int)  # Convert -1/1 - 0/1
        y_test=(y_test == 1).astype(int)

    run_pipeline(
        X_tr=X_train,
        X_te=X_test,
        y_tr=y_train,
        y_te=y_test,
        classifier=classifier,
        sampler=sampler,
        name=name,
        random_state=random_state,
        # ncomponents=50
    )

if __name__ == "__main__":
    main()
