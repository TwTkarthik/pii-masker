import torch
import unittest
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
import os

class PIIMasker:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "output_model", "deberta3base_1024")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def mask_pii(self, input_text):
        # Keep character offsets so predictions can be mapped back to the
        # original text.  The previous implementation zipped predictions
        # with tokenizer.tokenize(), which omitted [CLS]/[SEP] while the
        # predictions still included them.  That shifted every label by one
        # token (for example, labelling "My" as an address).
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()

        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        predicted_labels = [
            self.model.config.id2label[pred.item()]
            for pred in predictions[0]
        ]

        # Convert token-level labels into character spans.  Special tokens
        # have an empty (0, 0) offset and are therefore ignored naturally.
        token_spans = []
        for (start, end), label in zip(offsets, predicted_labels):
            if end <= start:
                continue
            label = label.replace('_STUDENT', '')
            if label == 'O':
                continue

            entity_type = label.split('-', 1)[-1]
            token_spans.append((start, end, entity_type))

        # Join adjacent subword tokens belonging to the same entity.  A one
        # character gap allows normal whitespace inside names and addresses.
        entities = []
        for start, end, entity_type in token_spans:
            if (
                entities
                and entities[-1]['type'] == entity_type
                and start <= entities[-1]['end'] + 1
            ):
                entities[-1]['end'] = end
            else:
                entities.append({'start': start, 'end': end, 'type': entity_type})

        # SSNs are handled deterministically as well.  The original code
        # reported them in pii_dict but left the sensitive value visible in
        # masked_text; replace any overlapping model spans with an SSN span.
        ssn_pattern = r'\b(?:\d{3}-\d{2}-\d{4}|\d{9})\b'
        for match in re.finditer(ssn_pattern, input_text):
            entities = [
                entity for entity in entities
                if entity['end'] <= match.start() or entity['start'] >= match.end()
            ]
            entities.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'SSN',
            })
        entities.sort(key=lambda entity: entity['start'])

        pii_dict = {}
        masked_parts = []
        cursor = 0
        for entity in entities:
            start, end, entity_type = entity['start'], entity['end'], entity['type']
            # SentencePiece offsets can include the whitespace immediately
            # before the first subword. Keep that whitespace in the original
            # text rather than inside the detected PII value.
            while start < end and input_text[start].isspace():
                start += 1
            while end > start and input_text[end - 1].isspace():
                end -= 1
            if start >= end:
                continue
            masked_parts.append(input_text[cursor:start])
            masked_parts.append(f"[B-{entity_type}]")
            pii_dict.setdefault(f"B-{entity_type}", []).append(input_text[start:end])
            cursor = end

        masked_parts.append(input_text[cursor:])
        masked_text = ''.join(masked_parts)

        # Add deterministic SSN matches that are handled separately by the
        # original project.
        pii_dict.update(self.extract_ssn(input_text))

        return masked_text, pii_dict

    @staticmethod
    def extract_ssn(input_string):
        ssn_pattern = r'\b(\d{3}-\d{2}-\d{4}|\d{9})\b'
        ssn_dict = {}
        matches = re.findall(ssn_pattern, input_string)
        for ssn in matches:
            ssn_dict[ssn] = 'SSN'  # Directly use the matched SSN
        return ssn_dict

if __name__ == "__main__":
    class TestPIIMasker(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            # 在测试开始前加载模型
            cls.masker = PIIMasker()
            
        def test_mask_pii(self):
            input_text = "My name is John Doe and my SSN is 123-45-6789."
            masked_text, _ = self.masker.mask_pii(input_text)

            # 检查输出是否包含预期的掩码文本
            self.assertIn("[B-NAME]", masked_text)  # 假设掩码为 [NAME]

        def test_extract_ssn(self):
            input_string = "My SSN is 987654320."
            ssn_dict = self.masker.extract_ssn(input_string)

            # 检查提取的 SSN 是否正确
            self.assertIn("987654320", ssn_dict)  

    unittest.main()
