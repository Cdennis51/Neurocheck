from transformers import pipeline

# def predict(preprocessed_image):

    # Create a pipeline
    # classifier = pipeline("image-classification", model="DHEIVER/Alzheimer-MRI")

    # Get label mappings from the model
    # try:
        # id2label = model.config.id2label
        # label2id = model.config.label2id

        # print(f"Number of classes: {len(id2label)}")
        # print(f"Classes: {id2label}")

    # except:
        # print("Failed to get label mappings from the model.")

    # Predict the label
    # result = classifier(preprocessed_image)

    # return result[0]




# Load once at module level
classifier = pipeline("image-classification", model="DHEIVER/Alzheimer-MRI")

def predict(preprocessed_image):
    # Access model config
    try:
        model = classifier.model
        id2label = model.config.id2label
        label2id = model.config.label2id

        print(f"Number of classes: {len(id2label)}")
        print(f"Classes: {id2label}")
    except Exception as e:
        print(f"Failed to get label mappings: {e}")

    # Predict
    result = classifier(preprocessed_image)
    return result[0]
