"""
Binary Classifier Model using BERT.

Copyright (c) 2024 Hideyuki Inada.  All rights reserved.

References
[1] https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel
[2] https://huggingface.co/google-bert/bert-base-uncased
"""
import os
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    """
    A BERT-based classifier for binary classification tasks.

    Args:
        pretrained_model_name (str): The name of the pre-trained BERT model to use. Default is 'bert-base-uncased'.

    Attributes:
        bert (BertModel): The BERT model used for encoding input sequences.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for classification.
        sigmoid (nn.Sigmoid): Sigmoid activation function for output probabilities.
    """

    def __init__(self, pretrained_model_name: str = 'google-bert/bert-base-uncased'):
        """
        Initializes the BertClassifier model.

        Args:
            pretrained_model_name (str): The name of the pre-trained BERT model to use. Default is 'bert-base-uncased'.
        """
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name, local_files_only=local_files_only_value)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.

        Args:
            input_ids (Tensor): Tensor containing input IDs for the BERT model.
            attention_mask (Tensor): Tensor containing attention masks to avoid attending to padding tokens.

            BERT's output is a tuple with:
              outputs[0]: last_hidden_state (batch_size, sequence_length, hidden_size)
              outputs[1]: pooler_output (batch_size, hidden_size)

        Returns:
            Tensor: Output probabilities from the model after applying sigmoid activation.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        avg_output = outputs[0].mean(dim=1)
        concat_output = torch.concat((pooled_output, avg_output), dim=1)
        dropout_output = self.drop(concat_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba

