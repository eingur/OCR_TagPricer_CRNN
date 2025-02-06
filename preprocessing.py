from torch.nn import CTCLoss
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


vocabulary = ["-"] + [str(x) for x in range(10)]
idx2char = {k:v for k,v in enumerate(vocabulary)}
char2idx = {v:k for k,v in idx2char.items()}

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    # img = cv2.cvtColor(img cv2.COLOR_)#RGB

    # Применение аугментации
    augmentations = A.Compose([
        A.Resize(256, 256, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    augmented_image = augmentations(image=img)['image']

    return augmented_image
        
def encode_text_batch(text_batch):
    
    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)
    
    return text_batch_targets
    
def compute_loss(criterion, text_batch, text_batch_logits, input_lengths, target_lengths):
    text_batch_targets = encode_text_batch(text_batch)
    loss = criterion(text_batch_logits, text_batch_targets, input_lengths, target_lengths)

    return loss

#у нас возвращается последовательность  [5-6-3-22-9] для числа 5639, нужно убрать дубли которые попали в несколько окон сегментации
def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)

def correct_prediction(word):
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word

def decode_sample(text_tokens):
    text = [idx2char[int(idx)] for idx in text_tokens]
    text = "".join(text)
    text = correct_prediction(text)
    return text
    
def decode_predictions(text_batch_tokens):

    text_batch_tokens = text_batch_tokens.numpy() # [batch_size, T]
    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = decode_sample(text_tokens)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new
