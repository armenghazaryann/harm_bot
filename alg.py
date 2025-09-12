import pdfplumber, re, json, argparse, os

def pdf_to_text(pdf_path):
    pages=[]
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
    return "\n".join(pages)

def collapse_whitespace(s):
    return re.sub(r"\s+", " ", s).strip()

def find_candidate_labels(text, max_words=7):
    labels=[]
    for m in re.finditer(r":", text):
        start=m.start()
        nl = text.rfind("\n", 0, start)
        seg_start = nl+1 if nl!=-1 else 0
        seg = text[seg_start:start].strip()
        if not seg: continue
        words = seg.split()
        if len(words) > max_words: continue
        # count "name-like" tokens (start with uppercase or digits/abbrev)
        name_like = sum(1 for w in words if re.match(r"^[A-Z0-9\(\)\-][A-Za-z0-9\.\-&()]*$", w))
        if name_like / len(words) >= 0.5:
            labels.append(seg)
    # deduplicate while preserving order
    seen=set(); uniq=[]
    for l in labels:
        if l not in seen:
            seen.add(l); uniq.append(l)
    return uniq

def segment_by_labels(text, labels):
    if not labels: return []
    labels_sorted = sorted(labels, key=lambda x:-len(x))
    pattern = re.compile(r"(?:" + "|".join(re.escape(l) for l in labels_sorted) + r")\s*:", flags=re.M)
    dialogues=[]
    last_pos=0; last_label=None
    for m in pattern.finditer(text):
        label_text = re.sub(r"\s*:$","", m.group(0)).strip()
        if last_label is not None:
            speech = text[last_pos:m.start()]
            dialogues.append({"speaker": last_label.strip(), "speech": collapse_whitespace(speech)})
        last_label = label_text
        last_pos = m.end()
    if last_label is not None and last_pos < len(text):
        dialogues.append({"speaker": last_label.strip(), "speech": collapse_whitespace(text[last_pos:])})
    return dialogues

def speaker_looks_like_sentence(s):
    words=s.split()
    if len(words) <= 5: return False
    lower_start = sum(1 for w in words if w and w[0].islower())
    common_sent_words = {"the","and","of","to","that","is","we","our","for","with","in","on","ability","will"}
    comm = sum(1 for w in words if w.lower() in common_sent_words)
    if lower_start/len(words) > 0.25 or comm >= 2:
        return True
    return False

def repair_dialogues(dialogues):
    repaired = []
    merged = 0
    for d in dialogues:
        if speaker_looks_like_sentence(d["speaker"]):
            # merge into previous if exists
            if repaired:
                repaired[-1]["speech"] = collapse_whitespace(repaired[-1]["speech"] + " " + d["speaker"] + " " + d["speech"])
                merged += 1
            else:
                # no previous => create Unknown
                repaired.append({"speaker":"Unknown","speech":collapse_whitespace(d["speaker"]+" "+d["speech"])})
                merged += 1
        else:
            repaired.append(d)
    return repaired, merged

def main(pdf_path, out_json, out_json_repaired=None):
    text = pdf_to_text(pdf_path)
    labels = find_candidate_labels(text)
    dialogues = segment_by_labels(text, labels)
    # fallback: if nothing found, fall back to line-based pattern
    if not dialogues:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        sp = re.compile(r"^([A-Z][A-Za-z0-9\s\.,&\-]{0,80}?):\s*(.*)$")
        current_speaker=None; current_speech=[]
        for line in lines:
            m = sp.match(line)
            if m:
                if current_speaker:
                    dialogues.append({"speaker":current_speaker,"speech":collapse_whitespace(" ".join(current_speech))})
                current_speaker = m.group(1).strip()
                current_speech = [m.group(2).strip()] if m.group(2).strip() else []
            else:
                if current_speaker:
                    current_speech.append(line)
        if current_speaker:
            dialogues.append({"speaker":current_speaker,"speech":collapse_whitespace(" ".join(current_speech))})

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(dialogues)} pairs to {out_json}")

    if out_json_repaired:
        repaired, merged = repair_dialogues(dialogues)
        with open(out_json_repaired, "w", encoding="utf-8") as f:
            json.dump(repaired, f, ensure_ascii=False, indent=2)
        print(f"Wrote repaired {len(repaired)} pairs to {out_json_repaired}  (merged {merged} suspicious entries)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-repaired", default=None)
    args = parser.parse_args()
    pdf_path = args.pdf_path
    out = args.out or os.path.splitext(os.path.basename(pdf_path))[0] + "_dialogues.json"
    out_repaired = args.out_repaired
    main(pdf_path, out, out_repaired)

