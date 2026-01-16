import nltk
import os
import json

nltk.download('punkt', quiet=True)


def get_last_sentence_nltk(text):
    if not text:
        return ""
    sentences = nltk.tokenize.sent_tokenize(text)
    return sentences[-1] if sentences else ""


def process_semeval_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            data = json.loads(line)

            text_keys = ['anchor_text', 'text_a', 'text_b']
            for key in text_keys:
                if key in data:
                    data[key] = get_last_sentence_nltk(data[key])

            f_out.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    root_path = './SemEval2026-Task_4-test-v1'
    input_file = os.path.join(root_path, 'test_track_a.jsonl')
    output_file = os.path.join(root_path, 'test_track_a_processed.jsonl')
    # root_path = './SemEval2026-Task_4-sample-v1'
    # input_file = os.path.join(root_path, 'sample_track_a.jsonl')
    # output_file = os.path.join(root_path, 'sample_track_a_processed.jsonl')
    # root_path = './SemEval2026-Task_4-dev-v1'
    # input_file = os.path.join(root_path, 'dev_track_a.jsonl')
    # output_file = os.path.join(root_path, 'dev_track_a_processed.jsonl')

    if os.path.exists(input_file):
        process_semeval_data(input_file, output_file)
        print(f"Processing complete. Saved to: {output_file}")
    else:
        print("Input file not found.")