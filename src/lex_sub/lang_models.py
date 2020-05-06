import torch
from transformers import BertTokenizer, BertForMaskedLM


class MaskedLM(LexSubBase):
    def __init__(self, vocab=None, LM=None, tokenizer=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        if vocab != None:
            super().__init__(vocab)
        self.LM = LM
        self.tokenizer = tokenizer
        self.device = device
        if self.LM is None:
            name = 'bert-base-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(name)
            self.LM = BertForMaskedLM.from_pretrained(name).to('cuda')
            
    def substitution_score(self, sentence, target_index, subs):
        target = sentence[target_index]
        x_in = torch.tensor([self.tokenizer.convert_tokens_to_ids(sentence)],
                            device=self.device)
        sentence[target_index] = '[MASK]'
        mask_in = torch.tensor([self.tokenizer.convert_tokens_to_ids(sentence)],
                               device=self.device)
        sentence[target_index] = target

        sub_ids = self.tokenizer.convert_tokens_to_ids(subs)
        with torch.no_grad():
            preds = self.LM(mask_in, masked_lm_labels=x_in)[1][0,target_index]

        indices = torch.argsort(preds[sub_ids], descending=True)
        ordered_subs = [subs[i] for i in indices]
        return ordered_subs

    def get_candidates(self):
        pass

    def _get_candidates(self, sentence, target_index, n_substitutes=10):
        target = sentence[target_index]
        x_in = torch.tensor([self.tokenizer.convert_tokens_to_ids(sentence)],
                            device=self.device)
        sentence[target_index] = '[MASK]'
        mask_in = torch.tensor([self.tokenizer.convert_tokens_to_ids(sentence)],
                               device=self.device)
        sentence[target_index] = word

        with torch.no_grad():
            preds = self.LM(mask_in, masked_lm_labels=x_in)[1][0,target_index]
            _, top_k_index = torch.topk(preds, n_substitutes)
            return self.tokenizer.convert_ids_to_tokens(top_k_index.tolist())