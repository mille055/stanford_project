import torch
import torch.nn.functional as F
import numpy as np
import pydicom
from torch.utils.data import DataLoader

#local imports
from cnn.cnn_model import MyModel
from cnn.cnn_data_loaders import get_data_loaders, dataset_sizes, data_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_pix_model(model,test_loader,device):
    model = model.to(device)
    # Turn autograd off
    with torch.no_grad():

        # Set the model to evaluation mode
        model.eval()

        # Set up lists to store true and predicted values
        y_true = []
        test_preds = []

        # Calculate the predictions on the test set and add to list
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # Feed inputs through model to get raw scores
            logits = model.forward(inputs)
            # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
            probs = F.softmax(logits,dim=1)
            # Get discrete predictions using argmax
            preds = np.argmax(probs.cpu().numpy(),axis=1)
            # Add predictions and actuals to lists
            test_preds.extend(preds)
            y_true.extend(labels.cpu())

        # Calculate the accuracy
        test_preds = np.array(test_preds)
        y_true = np.array(y_true)
        test_acc = np.sum(test_preds == y_true)/y_true.shape[0]
        
        # Recall for each class
        recall_vals = []
        for i in range(10):
            class_idx = np.argwhere(y_true==i)
            total = len(class_idx)
            correct = np.sum(test_preds[class_idx]==i)
            recall = correct / total
            recall_vals.append(recall)
    
    return test_acc,recall_vals


def image_to_tensor(filepath, transforms, device=device):
    
    ds = pydicom.dcmread(filepath)
    img = np.array(ds.pixel_array, dtype=np.float32)
    img = img[np.newaxis]
    img = torch.from_numpy(np.asarray(img))
    input_tensor = data_transforms['test'](img)

    # Add a batch dimension to the input tensor
    input_tensor = input_tensor.unsqueeze(0)
    print('changing input_tensor to shape', input_tensor.shape)
    # Move the input tensor to the appropriate device
    input_tensor = input_tensor.to(device)

    return input_tensor

def main():
    # Create instances of model, criterion, optimizer, and scheduler
    ####

    
    # Get data loaders
    batch_size = 8
    #train, val, test need to be imported
    
    train_loader, val_loader, test_loader, dataset_sizes = get_data_loaders(train, val, test, batch_size)

    # Perform inference on the test dataset
    test_acc, recall_vals = test_pix_model(model, test_loader, device)
    print("Test accuracy:", test_acc)
    print("Recall values:", recall_vals)

    # Perform inference on a single image
    image_path = "path/to/your/image.dcm"
    input_tensor = image_to_tensor(image_path, device)
    with torch.no_grad():
        model_ft.eval()
        output = model_ft(input_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
    print("Prediction for the single image:", pred.item())

if __name__ == "__main__":
    main()