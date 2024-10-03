import argparse
import config
import LogUtils as utils
import json


def chunk_list(token_text, chunk_size, pre_last_token):
    sp_list = []
    tokens = token_text[0]
    texts = token_text[1]
    per_senid_nums = []
    for i in range(0, len(tokens), chunk_size):
        end = i + chunk_size
        s_token = tokens[i:end]
        str_start = s_token[0][1] - pre_last_token
        str_end = s_token[-1][2] - pre_last_token
        s_text = texts[str_start: str_end]
        sp_list.append([s_token, s_text])
        per_senid_nums.append(len(s_token))
    return sp_list, per_senid_nums


# a = [i for i in range(1000)]  # 示例列表
def chunk_text(chunk_l, text):
    text_list = []
    for chunk in chunk_l:
        start = chunk[0]
        end = chunk[-1]
        text_s = text[start, end]
        text_list.append(text_s)


def compute_pre_sent_nums(sentence_idx_nums, curr_idx):
    pre_total_sent_nums = 0
    for idx in sentence_idx_nums:
        if idx < curr_idx:
            pre_total_sent_nums += sentence_idx_nums[idx]
        else:
            break
    return pre_total_sent_nums


def split_sentences(sentences_olds, chunk_size):
    bep_sents = {}
    current_idx = 0
    sentences_new = []
    pre_last_token = 0

    total_sent_nums_new = []
    for idx, tokens in enumerate(sentences_olds):
        if idx > 0:
            pre_last_token = sentences_olds[idx - 1][0][-1][2]

        chunk_l, per_senid_nums = chunk_list(tokens, chunk_size, pre_last_token)
        total_sent_nums_new.append(per_senid_nums)

        bep_sent_len = len(chunk_l)
        bep_sents[idx] = [current_idx, current_idx + bep_sent_len - 1]
        current_idx = current_idx + bep_sent_len
        sentences_new += chunk_l
    return sentences_new, bep_sents, total_sent_nums_new


def compute_total_token_count(sentences_old):
    total_count = 0
    for sentence in sentences_old:
        total_count += len(sentence[0])
    return total_count


# Construct a mapping of the position for the tokens and sentence ids between the new and old sentences, after the split.
# For example, the old sentence 100, split into ten. Then in the new sentence, the position of token 55 is ,[5, 4], the 5th position of the sixth sentence.
def find_new_sent_idx(total_sent_nums_new, search_token_id, new_sent_idx=None, new_token_id=None):
    cur_total_token_count = 0
    cur_total_sent_count = 0
    pre_total_sent_id = 0

    for idx, origin_sent in enumerate(total_sent_nums_new):
        if idx != 0:
            pre_total_sent_id = pre_total_sent_id + 1
        for s_sent_idx, split_sent_nums in enumerate(origin_sent):
            next_total_count = cur_total_token_count + split_sent_nums
            if s_sent_idx != 0:
                pre_total_sent_id = pre_total_sent_id + 1
            if search_token_id < next_total_count:
                current_token_position = search_token_id - cur_total_token_count
                return pre_total_sent_id, current_token_position
                break
            else:
                cur_total_token_count = cur_total_token_count + split_sent_nums

    return None, None


def map_old_po_to_new(total_token_count, total_sent_nums_new):
    old2new_map = {}
    for token_id_old in range(total_token_count):
        new_sent_idx, new_token_id = find_new_sent_idx(total_sent_nums_new, token_id_old)
        old2new_map[token_id_old] = (new_sent_idx, new_token_id)
    return old2new_map


def sent_idx_nums_map(new_sentences, sentences_new):
    sentence_idx_nums_new = {}
    for idx, tokens in enumerate(sentences_new):
        filtered_tokens = [token[0] for token in tokens[0]]
        sentence_text = tokens[1]
        sentence = [filtered_tokens, sentence_text]
        new_sentences.append(sentence)
        sentence_idx_nums_new[idx] = len(tokens[0])
    return sentence_idx_nums_new


def build_ent_info(entity_mentions, old2new_map):
    enid_eninfo = {}
    for entity_mention in entity_mentions:
        en_info = {}
        enid = entity_mention["id"]
        # sent_idx1 = entity_mention["sent_idx"]
        entity_type = entity_mention["entity_type"]
        text = entity_mention["text"]
        en_start = entity_mention["start"]
        en_end = entity_mention["end"] - 1

        new_sentidx1, curr_sta_position = old2new_map[en_start]
        new_sentidx2, curr_ed_position = old2new_map[en_end]
        if new_sentidx1 != new_sentidx2:
            print("two sentid is different")

        en_info["sent_idx"] = new_sentidx1
        en_info["entity_type"] = entity_type
        en_info["text"] = text
        en_info["start"] = curr_sta_position
        en_info["end"] = curr_ed_position
        enid_eninfo[enid] = en_info
    return enid_eninfo


def build_new_arguments(event_mentions, old2new_map, enid_eninfo):
    new_triggers = []
    arguments_new = []
    for event_mention in event_mentions:
        new_trigger = {}
        event_type = event_mention["event_type"]
        trigger = event_mention["trigger"]
        arguments = event_mention["arguments"]
        start = trigger["start"]
        end = trigger["end"] - 1
        tri_text = trigger["text"]

        new_sentid1, curr_sta_position = old2new_map[start]
        new_sentid2, curr_ed_position = old2new_map[end]
        if new_sentid1 != new_sentid2:
            print("two sentid is different")

        new_trigger["text"] = tri_text
        new_trigger["start"] = curr_sta_position
        new_trigger["end"] = curr_ed_position
        new_trigger["event_type"] = event_type
        new_trigger["sent_idx"] = new_sentid1

        for argument in arguments:
            # argument_new = {}
            role = argument["role"]
            arg_text = argument["text"]
            entity_id = argument["entity_id"]
            en_info = enid_eninfo[entity_id]
            arg_sent_idx = en_info["sent_idx"]
            arg_start = en_info["start"]
            arg_end = en_info["end"]

            argument_new = {"arg_text": arg_text, "arg_start": arg_start, "arg_end": arg_end,
                            "ent_id": entity_id, "sent_idx": arg_sent_idx, "role": role,
                            "tri_start": curr_sta_position, "tri_end": curr_ed_position, "event_type": event_type,
                            "tri_sent_idx": new_sentid1, "tri_text": tri_text}

            arguments_new.append(argument_new)

        new_triggers.append(new_trigger)


def build_new_sentencess(new_sentences, new_triggers, arguments_new, doc_id):
    doc_sentences = {}
    for idx, new_sentence in enumerate(new_sentences):
        doc_sentence = {}
        trigger_mentions = []
        for new_trigger in new_triggers:
            sent_idx = new_trigger["sent_idx"]
            if idx == sent_idx:
                trigger_mention = {"text": new_trigger["text"], "start": new_trigger["start"],
                                   "end": new_trigger["end"], "event_type": new_trigger["event_type"],
                                   "sent_idx": new_trigger["sent_idx"]}
                trigger_mentions.append(trigger_mention)
        doc_sentence["trigger_mentions"] = trigger_mentions

        args_sentences = []
        for arg_new in arguments_new:
            arg_sent_idx = arg_new["sent_idx"]
            if idx == arg_sent_idx:
                args_sentences.append(arg_new)
        doc_sentence["argument_mentions"] = args_sentences
        doc_sentence["sentence"] = new_sentence
        doc_sentences[doc_id + "-sent_idx-" + str(idx)] = doc_sentence
    return doc_sentences


def docs2sen(docs, dataset_name, chunk_size):
    docs_sentences = []

    for doc in docs:
        doc_id = doc["doc_id"]
        sentences_old = doc["sentences"]
        new_sentences = []
        entity_mentions = doc["entity_mentions"]
        event_mentions = doc["event_mentions"]

        sentence_idx_nums_old = {}
        for idx, tokens in enumerate(sentences_old):
            sentence_idx_nums_old[idx] = len(tokens[0])
        total_token_count = compute_total_token_count(sentences_old)

        sentences_new, bep_sents, total_sent_nums_new = split_sentences(sentences_old, chunk_size)
        old2new_map = map_old_po_to_new(total_token_count, total_sent_nums_new)
        enid_eninfo = build_ent_info(entity_mentions, old2new_map)
        new_triggers, arguments_new = build_new_arguments(event_mentions, old2new_map, enid_eninfo)
        doc_sentences = build_new_sentencess(new_sentences, new_triggers, arguments_new, doc_id)
        docs_sentences.append(doc_sentences)
    # return docs_sentences
    with open("./data/wikievents/{}_transfer_split.jsonl".format(dataset_name), "w+") as f:
        json.dump(docs_sentences, f)
    print("The new splitted dataset has been written in the json file")


if __name__ == '__main__':
    # define input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/wikievents.json')
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    # log settings
    config = config.Config(args)
    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    with open("./data/{}/train.jsonl".format(config.dataset), "r", encoding="utf-8") as f:
        train_data = [json.loads(x) for x in f.readlines()]
    with open("./data/{}/dev.jsonl".format(config.dataset), "r", encoding="utf-8") as f:
        dev_data = [json.loads(x) for x in f.readlines()]
    with open("./data/{}/test.jsonl".format(config.dataset), "r", encoding="utf-8") as f:
        test_data = [json.loads(x) for x in f.readlines()]

    datasets = {"train": train_data, "dev": dev_data, "test": test_data}
    for dataset_name in datasets.keys():
        dataset = datasets.get(dataset_name)
        docs2sen(dataset, dataset_name, 300)
