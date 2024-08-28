import json

import torch
from torch.utils.data import Dataset


class CustomRefDataset(Dataset):
    def __init__(self, fname, tokenizer, mode="eval"):
        IGNORE_INDEX=-100
        MAX_SEQ_LENGTH=tokenizer.model_max_length
        self.inp = []
        self.trg = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']

                if len(str(utterance)) > 1:
                    chat.append(f"화자{speaker}: {utterance}")

            chat = "\n".join(chat)

            reference_id_list = inp["reference_id"]
            references = ["[Utterance References]"]
            for cvt in inp['conversation']:
                if cvt["utterance_id"] in reference_id_list:
                    speaker = cvt['speaker']
                    utterance = cvt['utterance']
                    if len(str(utterance)) > 1:
                        references.append(f"화자{speaker}: {utterance}")
            references = "\n".join(references)

            question = f"[Question]\n위 대화의 특정 발화인 `Utterance References`에 대한 {inp['category']}"
            if (ord(inp['category'][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question = "로"
            question += " 올바른 지문은?"
                
            chat = chat + "\n\n" + references + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if mode == "train":
                if source[0].shape[0] > MAX_SEQ_LENGTH:
                    continue  # Skip this sample if it exceeds the maximum sequence length

            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
            else:
                target = ""
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]


class CustomRefOfficialTermDataset(Dataset):
    def __init__(self, fname, tokenizer, mode="eval"):
        IGNORE_INDEX=-100
        MAX_SEQ_LENGTH=tokenizer.model_max_length
        self.inp = []
        self.trg = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        OfficialTerm = {
            "원인": "원인(cause)",
            "후행사건": "후행 사건(subsequent event)",
            "전제": "전제 조건(prerequisite)",
            "동기": "내적 동기(motivation)",
            "반응": "감정 반응(emotional reaction)",
        }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                
                if len(str(utterance)) > 1:
                    chat.append(f"화자{speaker}: {utterance}")

            chat = "\n".join(chat)

            reference_id_list = inp["reference_id"]
            references = ["[Utterance References]"]
            for cvt in inp['conversation']:
                if cvt["utterance_id"] in reference_id_list:
                    speaker = cvt['speaker']
                    utterance = cvt['utterance']
                    if len(str(utterance)) > 1:
                        references.append(f"화자{speaker}: {utterance}")
            references = "\n".join(references)

            question = f"[Question]\n위 대화의 특정 발화인 `Utterance References`에 대한 **{OfficialTerm[inp['category']]}**"
            if (ord(OfficialTerm[inp['category']][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question = "로"
            question += " 올바른 지문은?"
                
            chat = chat + "\n\n" + references + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if mode == "train":
                if source[0].shape[0] > MAX_SEQ_LENGTH:
                    continue  # Skip this sample if it exceeds the maximum sequence length

            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
            else:
                target = ""
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]


class CustomRefDefinitionDataset(Dataset):
    def __init__(self, fname, tokenizer, mode="eval"):
        IGNORE_INDEX=-100
        MAX_SEQ_LENGTH=tokenizer.model_max_length
        self.inp = []
        self.trg = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        Definition = {
            "원인": "대화의 사건을 유발하는 사건",
            "후행사건": "대화 이후에 일어날 수 있는 사건",
            "전제": "대화의 사건을 가능하게 하는 상태 혹은 사건",
            "동기": "대화를 일으키는 '화자'의 감정이나 기본 욕구",
            "반응": "대화 사건에 대해 '청자'가 보일 수 있는 감정 반응",
        }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']

                if len(str(utterance)) > 1:
                    chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            reference_id_list = inp["reference_id"]
            references = ["[Utterance References]"]
            for cvt in inp['conversation']:
                if cvt["utterance_id"] in reference_id_list:
                    speaker = cvt['speaker']
                    utterance = cvt['utterance']
                    if len(str(utterance)) > 1:
                        references.append(f"화자{speaker}: {utterance}")
            references = "\n".join(references)

            question = f"[Question]\n주어진 대화의 내용과 문맥으로 미루어 보았을 때, 특정 발화인 `Utterance References`에 대해 **{Definition[inp['category']]}**"
            if (ord(Definition[inp['category']][-1]) - ord("가")) % 28 > 0:
                question += "을"
            else:
                question = "를"
            question += " 가장 잘 설명하는 문장은 무엇인가?"
                
            chat = chat + "\n\n" + references + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if mode == "train":
                if source[0].shape[0] > MAX_SEQ_LENGTH:
                    continue  # Skip this sample if it exceeds the maximum sequence length

            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
            else:
                target = ""
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]


class CustomRefInstructionDataset(Dataset):
    def __init__(self, fname, tokenizer, mode="eval"):
        IGNORE_INDEX=-100
        MAX_SEQ_LENGTH=tokenizer.model_max_length
        self.inp = []
        self.trg = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        Definition = {
            "원인": "대화의 사건을 유발하는 사건",
            "후행사건": "대화 이후에 일어날 수 있는 사건",
            "전제": "대화의 사건을 가능하게 하는 상태 혹은 사건",
            "동기": "대화를 일으키는 '화자'의 감정이나 기본 욕구",
            "반응": "대화 사건에 대해 '청자'가 보일 수 있는 감정 반응",
        }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            instruction = f"[Instruction]\n주어진 대화의 내용과 문맥을 바탕으로, 특정 발화인 `Utterance References`에 대한 **{Definition[inp['category']]}**"
            if (ord(Definition[inp['category']][-1]) - ord("가")) % 28 > 0:
                instruction += "을"
            else:
                instruction = "를"
            instruction += " 가장 잘 설명하는 문장을 선택하라."

            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']

                if len(str(utterance)) > 1:
                    chat.append(f"화자{speaker}: {utterance}")

            chat = "\n".join(chat)

            reference_id_list = inp["reference_id"]
            references = ["[Utterance References]"]
            for cvt in inp['conversation']:
                if cvt["utterance_id"] in reference_id_list:
                    speaker = cvt['speaker']
                    utterance = cvt['utterance']
                    if len(str(utterance)) > 1:
                        references.append(f"화자{speaker}: {utterance}")
            references = "\n".join(references)

            chat = instruction + "\n\n" + chat  + "\n\n" + references + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if mode == "train":
                if source[0].shape[0] > MAX_SEQ_LENGTH:
                    continue  # Skip this sample if it exceeds the maximum sequence length

            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
            else:
                target = ""
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]


class SystemRefOfficialTermDataset(Dataset):
    def __init__(self, fname, tokenizer, mode="eval"):
        IGNORE_INDEX=-100
        MAX_SEQ_LENGTH=tokenizer.model_max_length
        self.inp = []
        self.trg = []
        self.label = []

        PROMPT = '''당신은 단일 선택 질문(single-choice questions)에 정확하게 답변하는 데 도움을 주기 위해 훈련된 고급 언어 모델입니다.
이제부터 당신은 '대화 맥락 추론' 과제를 수행해야 합니다. 이 과제는 입력으로 주어진 `대화 (Conversation)`를 바탕으로, `특정된 대상 발화 (Utterance References)`로부터 `추론문 유형`에 가장 적절한 추론문을 선택하는 것입니다. 각 대화에 대해 세 개의 추론문 후보가 제공되며, 당신은 해당 대화를 정확하게 파악하고, 제시된 특정 대상 발화에 대해, 세 가지 추론문 중에서 추론문 유형에 가장 적합한 하나의 정답을 선택해야 합니다.

다음 지침을 주의 깊게 따르세요:

1. **대화 내용을 주의 깊게 읽으세요**: 대화의 전체적인 흐름과 구조를 이해하세요.
2. **대상 발화를 이해하세요**: 대화 중 특정된 발화가 무엇을 의미하는지 파악하세요.
3. **추론문 유형을 고려하세요**: 주어진 추론문이 '원인(cause)', '후행 사건(subsequent event)', '전제 조건(prerequisite)', '내적 동기(motivation)', '감정 반응(emotional reaction)' 중 어떤 유형인지 확인하세요.
4. **가장 적절한 추론문을 선택하세요**: 세 개의 추론문 후보(`Option`) 중 대상 발화에 가장 적합한 하나를 선택하세요.

항상 다음 형식으로 답변을 제공하세요:
- 선택한 추론문에 해당하는 문자 (A, B, C).

작업을 진행하세요.'''

        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        OfficialTerm = {
            "원인": "원인(cause)",
            "후행사건": "후행 사건(subsequent event)",
            "전제": "전제 조건(prerequisite)",
            "동기": "내적 동기(motivation)",
            "반응": "감정 반응(emotional reaction)",
        }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                
                if len(str(utterance)) > 1:
                    chat.append(f"화자 {speaker}: {utterance}")
                    
            chat = "\n".join(chat)

            reference_id_list = inp["reference_id"]
            references = ["[Utterance References]"]
            for cvt in inp['conversation']:
                if cvt["utterance_id"] in reference_id_list:
                    speaker = cvt['speaker']
                    utterance = cvt['utterance']
                    if len(str(utterance)) > 1:
                        references.append(f"화자{speaker}: {utterance}")
            references = "\n".join(references)

            question = f"[Question]\n위 대화의 특정 발화인 `Utterance References`에 대한 **{OfficialTerm[inp['category']]}**"

            if (ord(OfficialTerm[inp['category']][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question = "로"
            question += " 올바른 추론문은?"
                
            chat = chat + "\n\n" + references + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if mode == "train":
                if source[0].shape[0] > MAX_SEQ_LENGTH:
                    continue  # Skip this sample if it exceeds the maximum sequence length

            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
            else:
                target = ""
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]



class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
