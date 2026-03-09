import torch

def validate_distilled(device, student_model, teacher_model, dataloader, criterion):
    running_loss_val = 0.0
    correct_predictions_val = 0
    total_predictions_val = 0
    
    student_model.eval()
    teacher_model.eval()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = student_model(inputs)
            loss = criterion(inputs, outputs, labels)
            running_loss_val += loss.item()

            outputs = torch.softmax(outputs, 1)
            _, predicted = torch.max(outputs, 1)
            
            total_predictions_val += labels.size(0)
            correct_predictions_val += (predicted == labels).sum().item()
                               
    epoch_loss_val = running_loss_val / len(dataloader)
    epoch_accuracy_val = (correct_predictions_val / total_predictions_val) * 100
    return epoch_loss_val, epoch_accuracy_val

def predict_batch(device, model, batch, binary=True):
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        outputs = model(batch)
        if binary:
            predicted = torch.round(outputs)
        else:
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu(), outputs.cpu()