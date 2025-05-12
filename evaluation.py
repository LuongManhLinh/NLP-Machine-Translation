import sacrebleu
from laonlp.tokenize import word_tokenize as lo_tokenize
from vi_tokenize import vi_tokenize
import threading
import ctypes
import evaluate

def run_with_timeout(func, timeout, *args, **kwargs):
    result = [None]
    exception = [None]
    has_returned = threading.Event()
    
    # Function to be run in the thread
    def target_func():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
        finally:
            has_returned.set()
    
    # Create and start the thread
    thread = threading.Thread(target=target_func)
    thread.daemon = True
    thread.start()
    
    # Wait for the function to complete or for timeout
    has_returned.wait(timeout)
    
    if has_returned.is_set():
        # Function completed in time
        if exception[0]:
            raise exception[0]
        return result[0]
    else:
        # Function did not complete in time, terminate it
        tid = thread.ident
        if tid:
            # This is a low-level approach to forcibly terminate a thread
            # It's not generally recommended due to potential resource leaks,
            # but is one of the few ways to actually kill a thread in Python
            if hasattr(ctypes, 'pythonapi'):
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(tid),
                    ctypes.py_object(SystemExit)
                )
                if res == 0:
                    raise ValueError("Invalid thread ID")
                elif res != 1:
                    # If more than one thread was affected, restore the state
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
                    raise SystemError("PyThreadState_SetAsyncExc failed")
        return None

        
def tokenize_text(tokenize_fn, text):
    """A custome tokenize function useful in the case of tokenize_fn running for too long"""
    return run_with_timeout(tokenize_fn, 2, text)


def calculate_bleu(reference_file, candidate_file, lang):
    with open(reference_file, 'r', encoding='utf-8') as ref_file:
        references = ref_file.readlines()
    
    with open(candidate_file, 'r', encoding='utf-8') as cand_file:
        candidates = cand_file.readlines()

    if len(references) != len(candidates):
        raise ValueError(f"Number of references ({len(references)}) does not match number of candidates ({len(candidates)})")
    
    if lang == 'vi':
        tokenize_fn = vi_tokenize
    elif lang == 'lo':
        tokenize_fn = lo_tokenize
    else:
        raise ValueError(f"Unsupported language: {lang}")

    filtered_references = []
    filtered_candidates = []

    for idx in range(len(references)):
        ref = references[idx].strip()
        cand = candidates[idx].strip()

        tokenized_ref = tokenize_text(tokenize_fn, ref)
        tokenized_cand = tokenize_text(tokenize_fn, cand)

        if tokenized_ref is not None and tokenized_cand is not None:
            filtered_references.append([' '.join(tokenized_ref)])
            filtered_candidates.append(' '.join(tokenized_cand))
        else:
            print(f"Skipping line {idx + 1} due to timeout or error in tokenization.")

    bleu_score_1 = sacrebleu.corpus_bleu(filtered_candidates, filtered_references).score
    chrf_score_1 = sacrebleu.corpus_chrf(candidates, references, word_order=0).score

    bleu = evaluate.load("bleu")
    bleu_score_2 = bleu.compute(predictions=filtered_candidates, references=filtered_references)['bleu']

    chrf = evaluate.load("chrf")
    chrf_score_2 = chrf.compute(predictions=candidates, references=references)['score']
    return bleu_score_1, bleu_score_2, chrf_score_1, chrf_score_2


if __name__ == "__main__":
    lang = 'vi'  # Change to 'vi' for Vietnamese
    file_idx = 1  # Change to the desired file index
    candidate_file = f'results/{lang}/pred{file_idx}.txt'
    reference_file = f'results/{lang}/label{file_idx}.txt'

    bleu_score_1, bleu_score_2, chrf_score_1, chrf_score_2 = calculate_bleu(reference_file, candidate_file, lang)
    print(f"BLEU score by sacrebleu: {bleu_score_1}")
    print(f"BLEU score by evaluate:  {bleu_score_2}")
    print(f"CHRF score by sacrebleu: {chrf_score_1}")
    print(f"CHRF score by evaluate:  {chrf_score_2}")


