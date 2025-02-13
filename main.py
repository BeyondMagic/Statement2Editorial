# Aqui é para testar se o dispositivo está disponível para usar a GPU.
# E import as bibliotecas necessárias para rodar o código.
# Fizemos assim para também podermos rodar o código localmente, e no Google Colab.
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0)
print(f"Arquitetura de dispositivo: {device} e nome do dispositivo: {device_name}" )

# Já que juntamos todos os dados em um CSV com duas colunas: statement do problema E editorial;
# a gente tem que utilizar uma biblioteca específica, com a função load_dataset para carregar o arquivo.
from datasets import load_dataset
