from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn as nn

class DialogueActSlotValueModel(nn.Module):
    def __init__(self, num_acts, num_slots, num_values):
        super(DialogueActSlotValueModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.act_classifier = nn.Linear(self.bert.config.hidden_size, num_acts)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)
        self.value_classifier = nn.Linear(self.bert.config.hidden_size, num_values)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        act_logits = self.act_classifier(pooled_output)
        slot_logits = self.slot_classifier(pooled_output)
        value_logits = self.value_classifier(pooled_output)
        return act_logits, slot_logits, value_logits
