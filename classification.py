import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, TensorDataset

# Check if 'data' directory exists. If not, create directories for three classes.
if not os.path.exists('data'):
    os.makedirs('data/class1')
    os.makedirs('data/class2')
    os.makedirs('data/class3')

# Function to collect data using the webcam.
def collect_data(class_name, num_samples=10):
    # Initialize the webcam.
    cap = cv2.VideoCapture(0)
    count = 0
    print('Collecting', class_name)
    while True:
        # Read frames from the webcam.
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show the current frame for data collection.
        cv2.imshow('Collect Data', frame)

        # When 'c' key is pressed, save the current frame as an image.
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(f'data/{class_name}/{class_name}_{count}.jpg', frame)
            print(f'sample {count+1}')
            count += 1
            # Stop collection after reaching the desired number of samples.
            if count >= num_samples:
                break

    # Release the webcam and close OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()

# Collect samples for each class.
collect_data('class1', 10)
collect_data('class2', 10)
collect_data('class3', 10)

# Setup device for PyTorch (use CUDA if available).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet-50 model
model = resnet50(pretrained=True)
# Freeze all layers in the model. Training only the final layer.
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer to classify 3 classes.
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

# Define the transformations for preprocessing the images.
data_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Convert image to PIL Image format.
    transforms.Resize((224, 224)),  # Resize to fit ResNet50's input size.

    # Apply random augmentations to increase dataset variety.
    transforms.RandomVerticalFlip(),  # Randomly flip the image vertically.
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop and resize the image.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change image brightness, contrast, etc.
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image.

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Normalize according to values suitable for ResNet50.
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Lists to hold processed images and their labels.
data = []
labels = []

# Read and preprocess images from each class directory.
for class_name in ['class1', 'class2', 'class3']:
    for filename in os.listdir(f'data/{class_name}'):
        img = cv2.imread(f'data/{class_name}/{filename}')
        img = data_transforms(img)
        data.append(img)
        # Assign a numeric label based on class.
        if class_name == 'class1':
            labels.append(0)
        elif class_name == 'class2':
            labels.append(1)
        else:
            labels.append(2)

# Convert data and labels to PyTorch tensors.
data = torch.stack(data)
labels = torch.tensor(labels)

# Create a PyTorch dataset and data loader.
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop for the model.
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero out any gradient from the previous step.
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Compute the gradients.
        optimizer.step()  # Update the weights.

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Use the trained model for live predictions using webcam feed.
cap = cv2.VideoCapture(0)

# Save the model's state dictionary
model_path = 'model_state.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
model.load_state_dict(torch.load(model_path))

model.eval()  # Set the model to evaluation mode if using for inference

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame and get predictions.
    # Transform the image frame: increases the dimensions of the tensor 
    # by adding an extra dimension at the 0th position to create
    # a batch of one image for ResNet50
    input_img = data_transforms(frame).unsqueeze(0).to(device)
    with torch.no_grad():  # It doesn't need to keep track of gradients.
        output = model(input_img)
        _, pred = torch.max(output, 1)  # Get the predicted class

    # Map numeric predictions back to class names.
    if pred == 0:
        label = 'class1'
    elif pred == 1:
        label = 'class2'
    else:
        label = 'class3'
    
    # Display the predictions on the frame.
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow('Predictions', frame)

    # Exit loop on pressing 'q' key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources.
cap.release()
cv2.destroyAllWindows()
