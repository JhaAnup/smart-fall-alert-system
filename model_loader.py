# Model Loader Module
# Loads the pre-trained CNN-LSTM fall detection model


import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    # CNN-LSTM architecture for fall detection
    # Input: (batch, seq_len=30, features=99) means 30 frames of 99 features
    # Output: (batch, 2) - Fall/Non-Fall classification means 2 classes

    def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layers (bidirectional)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True  # This doubles the output size
        )
        
        # Fully connected layers
        # Note: bidirectional LSTM outputs hidden_size * 2
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        
        # CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # For bidirectional LSTM, h_n has shape (num_layers * 2, batch, hidden_size)
        # We need to concatenate the final forward and backward hidden states
        # h_n[-2] is the last forward layer, h_n[-1] is the last backward layer
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        x = torch.cat([h_forward, h_backward], dim=1)  # Concatenate to get (batch, hidden_size*2)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_model(model_path='best_fall_detector.pth', device='cpu'):
    # Load the trained fall detection model
    # model_path: Path to the saved model file
    # device: Device to load model on ('cpu' or 'cuda')
    
    # Initialize model
    model = CNNLSTMModel(input_size=99, hidden_size=128, num_layers=2, num_classes=2)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def predict(model, pose_sequence, device='cpu', threshold=0.95):
    # Make prediction on pose sequence
    # model: Loaded PyTorch model
    # pose_sequence: Tensor of shape (1, 30, 99) or (30, 99) means 30 frames of 99 features
    # device: Device for inference
    # threshold: Confidence threshold for fall detection 
    
    # Returns:
    # dictionary: {
    #     'prediction': 'Fall Detected' or 'Normal Activity',
    #     'confidence': float,
    #     'probabilities': [prob_normal, prob_fall] means 2 classes [normal, fall]
    # }
    
    model.eval()
    
    with torch.no_grad():
        # Ensure correct shape
        if pose_sequence.dim() == 2:
            pose_sequence = pose_sequence.unsqueeze(0)  # Add batch dimension
        
        pose_sequence = pose_sequence.to(device)
        
        # Forward pass
        outputs = model(pose_sequence)
        
        # Get probabilities
        probs = torch.softmax(outputs, dim=1)
        prob_normal = probs[0][0].item()
        prob_fall = probs[0][1].item()
        
        # Prediction
        if prob_fall >= threshold:
            prediction = "Fall Detected"
            confidence = prob_fall
        else:
            prediction = "Normal Activity"
            confidence = prob_normal
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': [prob_normal, prob_fall]
        }
