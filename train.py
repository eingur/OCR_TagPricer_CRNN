import torch.optim as optim
from tqdm import tqdm
from preprocessing import encode_text_batch, compute_loss, decode_predictions, correct_prediction, preprocess_image, decode_sample
from torch.nn import CTCLoss
import torch 
import numpy as np
import pandas as pd

# Обучение модели
def train_model(model, train_loader, num_epochs=10, lr=0.001, device = 'cuda:0', PATH_TO_SAVE = 'checkpoints/CRNN.pth'):
    model.train()
    
    criterion = CTCLoss(blank=0, zero_infinity = True)  # CTC Loss #zero_infinity!
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)
    epoch_loss_list = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to(device)  # Перемещаем на GPU
            #labels = [torch.tensor([int(c) for c in label]).to(device) for label in labels]  # Преобразуем строки в числа
            # Инициализация градиентов
            optimizer.zero_grad()

            output = model(images) 

            input_lengths = torch.full(size=(output.size(0),), fill_value=output.size(1), dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(label) for label in labels]).to(device)
            logits = output.log_softmax(2).permute(1,0,2)
            # loss = criterion(logits, torch.cat(labels), input_lengths, target_lengths)
            loss = compute_loss(criterion, labels, logits, input_lengths, target_lengths)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                print("FUCK")
                continue
            # Обратный проход
            loss.backward()
            optimizer.step()
            epoch_loss_list.append(loss.item())

            running_loss += loss.item()
        lr_scheduler.step(np.mean(epoch_loss_list))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    torch.save(model, PATH_TO_SAVE)

def predict_one_model(model,file_path,device = 'cuda:0'):
    img = preprocess_image(file_path).unsqueeze(0)
    with torch.no_grad():
        output = model(img) 
        logits = output.log_softmax(2).argmax(2).to(device)
        predictions = decode_sample(logits.to(device).numpy()[0])
    return predictions



def eval_model(model, test_loader,device = 'cuda:0'):
    model.eval()
    results_test = pd.DataFrame(columns=['actual', 'prediction'])    
    criterion = CTCLoss(blank=0, zero_infinity=True)  # CTC Loss
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)  # Перемещаем на GPU
            output = model(images) 
            logits = output.log_softmax(2).argmax(2).to(device)
            predictions = decode_predictions(logits.to(device))
            df = pd.DataFrame(columns=['actual', 'prediction'])
            df['actual'] = labels
            df['prediction'] = predictions
            results_test = pd.concat([results_test, df])
    results_test = results_test.reset_index(drop=True)
    return  results_test